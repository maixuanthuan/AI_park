"""Test kiểm config (T17-T19): mỗi loại lỗi đều bị bắt, config đang giao phải hợp lệ, detector không có mặc định ngầm."""
import copy
import os

import pytest
import yaml

from run_tamper import validate_config
from tamper_detector import TamperDetector

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def load(name):
    with open(os.path.join(ROOT, name)) as f:
        return yaml.safe_load(f)


def base_cfg():
    cfg = load("config.yaml")
    cfg["cameras"] = [{"id": "cam_a", "url": "rtsp://10.0.0.1/live"}]
    return cfg


def _set(path, value):
    def f(cfg):
        *parents, last = path
        d = cfg
        for p in parents:
            d = d[p]
        d[last] = value
    return f


def _del(path):
    def f(cfg):
        *parents, last = path
        d = cfg
        for p in parents:
            d = d[p]
        del d[last]
    return f


# (tên ca, cách làm hỏng config, đoạn chữ phải có trong câu báo lỗi)
CASES = [
    ("thiếu mục detector", _del(["detector", "clear_sec"]), "detector.clear_sec: thiếu"),
    ("thiếu mục source", _del(["source", "sample_fps"]), "source.sample_fps: thiếu"),
    ("thiếu cả mục lớn", _del(["source"]), "source: thiếu"),
    ("thiếu stats_sec", _del(["stats_sec"]), "stats_sec: thiếu"),
    ("stats_sec bằng 0", _set(["stats_sec"], 0), "stats_sec = 0: phải trong khoảng"),
    ("mục lạ (gõ nhầm)", _set(["detector", "hold_secs"], 10), "detector.hold_secs: tên mục không tồn tại"),
    ("mục lạ ở gốc", _set(["out_dirr"], "x"), "out_dirr: tên mục không tồn tại"),
    ("số viết thành chuỗi", _set(["detector", "hold_sec"], "10"), "detector.hold_sec = '10': phải là số"),
    ("true/false không phải số", _set(["detector", "hold_sec"], True), "detector.hold_sec = True: phải là số"),
    ("số nguyên viết số lẻ", _set(["detector", "grid"], 8.5), "detector.grid = 8.5: phải là số nguyên"),
    ("hwaccel không phải true/false", _set(["source", "hwaccel"], "yes"), "source.hwaccel = 'yes': phải là true hoặc false"),
    ("nhầm 60 thay vì 0.6", _set(["detector", "obstruct_frac"], 60), "detector.obstruct_frac = 60: phải trong khoảng"),
    ("thời gian bằng 0", _set(["detector", "hold_sec"], 0), "detector.hold_sec = 0: phải trong khoảng"),
    ("vùng đệm ngược chiều (obstruct)", _set(["detector", "exit_obstruct"], 0.7), "exit_obstruct"),
    ("vùng đệm ngược chiều (edge_keep)", _set(["detector", "exit_edge_keep"], 0.2), "exit_edge_keep"),
    ("tối >= chói", _set(["detector", "dark_thr"], 250), "dark_thr"),
    ("osd_mask x1 >= x2", _set(["detector", "osd_mask"], [[0.5, 0, 0.2, 0.1]]), "detector.osd_mask"),
    ("osd_mask ngoài khung", _set(["detector", "osd_mask"], [[0, 0, 1.5, 0.1]]), "detector.osd_mask"),
    ("không có camera", _set(["cameras"], []), "cameras: phải có ít nhất 1 camera"),
    ("camera thiếu id", _set(["cameras"], [{"url": "rtsp://a/b"}]), "cameras[0].id: thiếu"),
    ("camera có cả url lẫn url_env", _set(["cameras"], [{"id": "a", "url": "rtsp://a/b", "url_env": "X"}]),
     "cameras[0]: phải có đúng một trong hai"),
    ("camera không có url nào", _set(["cameras"], [{"id": "a"}]), "cameras[0]: phải có đúng một trong hai"),
    ("mục lạ trong camera", _set(["cameras"], [{"id": "a", "url": "rtsp://a/b", "sample_fps": 2}]),
     "cameras[0].sample_fps: tên mục không tồn tại"),
    ("mục lạ trong ghi đè detector", _set(["cameras"], [{"id": "a", "url": "rtsp://a/b", "detector": {"hold_secs": 3}}]),
     "cameras[0].detector.hold_secs: tên mục không tồn tại"),
    ("ghi đè sai khoảng", _set(["cameras"], [{"id": "a", "url": "rtsp://a/b", "source": {"sample_fps": 0}}]),
     "cameras[0].source.sample_fps = 0: phải trong khoảng"),
    ("ghi đè làm vùng đệm ngược chiều", _set(["cameras"], [{"id": "a", "url": "rtsp://a/b", "detector": {"obstruct_frac": 0.3}}]),
     "cameras[0].detector: exit_obstruct"),
]


@pytest.mark.parametrize("name,mutate,expect", CASES, ids=[c[0] for c in CASES])
def test_t17_moi_loai_loi_config_deu_bi_bat(name, mutate, expect):
    """Tình huống thật: devops / dev sửa config và gõ sai (thiếu mục, gõ nhầm tên, sai kiểu, sai khoảng, trùng camera...).
    Giả lập: config.yaml đang giao, làm hỏng đúng một chỗ theo từng ca.
    Đỏ thì hỏng gì: lỗi đó lọt qua; hệ thống chạy với giá trị sai, hoặc chết giữa chừng lúc đang vận hành.
    """
    cfg = base_cfg()
    assert validate_config(copy.deepcopy(cfg), check_env=False) == []  # trước khi làm hỏng: hợp lệ
    mutate(cfg)
    errors = validate_config(cfg, check_env=False)
    assert any(expect in e for e in errors), f"không thấy '{expect}' trong {errors}"


def test_t17b_moi_truong_url_env_va_file_video(tmp_path, monkeypatch):
    """Tình huống thật: link camera chưa điền vào .env, hoặc đường dẫn video gõ sai.
    Giả lập: camera trỏ tới biến .env không tồn tại và tới file không có trên đĩa.
    Đỏ thì hỏng gì: chương trình chạy rồi thử kết nối mãi mãi, devops không biết sai ở đâu.
    """
    monkeypatch.delenv("CAM_KHONG_CO_URL", raising=False)
    monkeypatch.chdir(tmp_path)
    cfg = base_cfg()
    cfg["cameras"] = [{"id": "a", "url_env": "CAM_KHONG_CO_URL"}, {"id": "b", "url": "videos/khong_co.mp4"}]
    errors = validate_config(cfg, check_env=True)
    assert any("cameras[0].url_env = 'CAM_KHONG_CO_URL': chưa có trong .env" in e for e in errors), errors
    assert any("cameras[1].url = 'videos/khong_co.mp4': không tìm thấy file" in e for e in errors), errors

    (tmp_path / "videos").mkdir()
    (tmp_path / "videos" / "khong_co.mp4").write_bytes(b"")
    monkeypatch.setenv("CAM_KHONG_CO_URL", "rtsp://u:p@10.0.0.1/live")
    assert validate_config(cfg, check_env=True) == []


@pytest.mark.parametrize("name", ["config.yaml", "config.park.yaml"])
def test_t18_config_dang_giao_phai_hop_le(name):
    """Tình huống thật: giao một bản config mà chính nó đã sai.
    Giả lập: kiểm cấu trúc 2 file config trong repo (không kiểm .env và file video: đó là chuyện của máy chạy).
    Đỏ thì hỏng gì: người nhận chạy lên là bị từ chối khởi động, hoặc tệ hơn, chạy với giá trị sai.
    """
    assert validate_config(load(name), check_env=False) == []


def test_t19_detector_khong_co_mac_dinh_ngam(cfg):
    """Tình huống thật: gọi TamperDetector trực tiếp (vd từ bài toán khác) với config thiếu mục.
    Giả lập: config.yaml bỏ mục hold_sec rồi tạo TamperDetector.
    Đỏ thì hỏng gì: detector tự điền giá trị mặc định, chạy với con số không ai chọn.
    """
    c = cfg()
    del c["hold_sec"]
    with pytest.raises(ValueError, match="hold_sec"):
        TamperDetector(c)
