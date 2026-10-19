"""Test kiểm config (K1-K5): cameras.yaml (devops), zones.yaml (backend), profiles/overview.yaml (dev).

Config sai thì không khởi động, báo MỌI lỗi một lượt, chỉ đúng mục sai (như camera_tamper).
"""
import copy
import os

import pytest
import yaml

from config import validate_all

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PROFILE_DIR = os.path.join(ROOT, "profiles")


def load(name):
    with open(os.path.join(ROOT, name)) as f:
        return yaml.safe_load(f)


def base():
    """Một bộ config hợp lệ tối thiểu: 1 camera overview đếm, 1 vùng."""
    cams = {"cameras": [{"cam_id": "cam_a", "name": "Sân chơi", "url_env": "CAM_A_URL",
                         "profile": "overview", "functions": ["count"]}]}
    zones = {"zones": [{"zone_id": "z1", "cam_id": "cam_a", "function": "count",
                        "polygon": [[0, 0], [1, 0], [1, 1], [0, 1]], "ref_size": [1920, 1080],
                        "params": {"report_sec": 30}, "version": 1}]}
    return cams, zones, load("profiles/overview.yaml")


def errors_of(cams, zones, prof, check_env=False):
    return validate_all(cams, zones, prof, ROOT, PROFILE_DIR, check_env=check_env)


def _set(which, path, value):
    def f(cfgs):
        d = cfgs[which]
        *parents, last = path
        for p in parents:
            d = d[p]
        d[last] = value
    return f


def _del(which, path):
    def f(cfgs):
        d = cfgs[which]
        *parents, last = path
        for p in parents:
            d = d[p]
        del d[last]
    return f


C, Z, P = 0, 1, 2
CAM0, ZONE0 = ["cameras", 0], ["zones", 0]
CASES = [
    # cameras.yaml: devops điền
    ("camera thiếu cam_id", _del(C, CAM0 + ["cam_id"]), "cameras[0].cam_id: thiếu"),
    ("camera gõ nhầm tên mục", _set(C, CAM0 + ["fuctions"], ["count"]), "cameras[0].fuctions: tên mục không tồn tại"),
    ("camera có cả url lẫn url_env", _set(C, CAM0 + ["url"], "videos/a.mp4"), "cameras[0]: phải có đúng một trong hai"),
    ("profile không tồn tại", _set(C, CAM0 + ["profile"], "overveiw"), "cameras[0].profile = 'overveiw'"),
    ("chức năng không thuộc profile", _set(C, CAM0 + ["functions"], ["count", "face"]),
     "cameras[0].functions: 'face' không thuộc profile overview"),
    ("chức năng khai trùng", _set(C, CAM0 + ["functions"], ["count", "count"]), "cameras[0].functions: 'count' khai 2 lần"),
    ("không bật chức năng nào", _set(C, CAM0 + ["functions"], []), "cameras[0].functions: phải có ít nhất 1 chức năng"),
    # zones.yaml: backend vẽ
    ("vùng trỏ camera không có", _set(Z, ZONE0 + ["cam_id"], "cam_x"), "zones[0].cam_id = 'cam_x': không có trong cameras.yaml"),
    ("vùng của chức năng camera không bật", _set(Z, ZONE0 + ["function"], "pet_waste"),
     "zones[0].function = 'pet_waste': camera cam_a không bật chức năng này"),
    ("polygon dưới 3 điểm", _set(Z, ZONE0 + ["polygon"], [[0, 0], [1, 1]]), "zones[0].polygon"),
    ("polygon theo pixel thay vì tỉ lệ", _set(Z, ZONE0 + ["polygon"], [[0, 0], [1919, 0], [1919, 1079]]), "zones[0].polygon"),
    ("thiếu ref_size", _del(Z, ZONE0 + ["ref_size"]), "zones[0].ref_size: thiếu"),
    ("thiếu ngưỡng nghiệp vụ", _set(Z, ZONE0 + ["params"], {}), "zones[0].params.report_sec: thiếu"),
    ("ngưỡng nghiệp vụ sai kiểu", _set(Z, ZONE0 + ["params"], {"report_sec": "30"}), "zones[0].params.report_sec = '30': phải là số"),
    ("camera bật đếm mà không có vùng", _set(Z, ["zones"], []), "cam_a: bật count nhưng chưa có vùng count nào"),
    # profiles/overview.yaml: dev
    ("profile thiếu mục", _del(P, ["stats_sec"]), "stats_sec: thiếu"),
    ("profile gõ nhầm", _set(P, ["mux", "widht"], 1920), "mux.widht: tên mục không tồn tại"),
    ("lớp đếm không có trong labels của model", _set(P, ["count", "classes"], ["person", "dogs"]),
     "count.classes: 'dogs' không có trong labels"),
    ("tracker bật mà enable không phải true/false", _set(P, ["tracker", "enable"], "yes"), "tracker.enable = 'yes'"),
]


@pytest.mark.parametrize("name,mutate,expect", CASES, ids=[c[0] for c in CASES])
def test_k1_moi_loai_loi_config_deu_bi_bat(name, mutate, expect):
    """Tình huống thật: devops / backend / dev sửa config và gõ sai.
    Giả lập: bộ config hợp lệ, làm hỏng đúng một chỗ theo từng ca.
    Đỏ thì hỏng gì: lỗi lọt qua, hệ đếm sai hoặc chết giữa chừng lúc đang vận hành.
    """
    cfgs = list(base())
    assert errors_of(*copy.deepcopy(cfgs)) == []  # trước khi làm hỏng: hợp lệ
    mutate(cfgs)
    errors = errors_of(*cfgs)
    assert any(expect in e for e in errors), f"không thấy '{expect}' trong {errors}"


def test_k2_trung_cam_id_va_trung_zone_id():
    """Tình huống thật: dán trùng một camera / một vùng (gặp thật 2026-10-16 với camera_tamper).
    Giả lập: hai camera cùng cam_id, hai vùng cùng zone_id.
    Đỏ thì hỏng gì: báo cáo gửi 2 lần, backend không biết số nào đúng.
    """
    cams, zones, prof = base()
    cams["cameras"].append(copy.deepcopy(cams["cameras"][0]))
    zones["zones"].append(copy.deepcopy(zones["zones"][0]))
    errors = errors_of(cams, zones, prof)
    assert any("cameras[1].cam_id = 'cam_a': trùng với cameras[0]" in e for e in errors), errors
    assert any("zones[1].zone_id = 'z1': trùng với zones[0]" in e for e in errors), errors


def test_k3_moi_truong_url_env_va_file_video(tmp_path, monkeypatch):
    """Tình huống thật: link camera chưa điền vào .env, hoặc đường dẫn video gõ sai.
    Giả lập: một camera trỏ biến .env không có, một camera trỏ file không có.
    Đỏ thì hỏng gì: chương trình chạy rồi chờ mãi một camera không bao giờ có, devops không biết sai ở đâu.
    """
    monkeypatch.delenv("CAM_A_URL", raising=False)
    cams, zones, prof = base()
    prof["tracker"]["enable"] = False  # file config tracker chỉ có trong container DeepStream
    cams["cameras"].append({"cam_id": "cam_b", "name": "Video thử", "url": "videos/khong_co.mp4",
                            "profile": "overview", "functions": ["count"]})
    zones["zones"].append({**copy.deepcopy(zones["zones"][0]), "zone_id": "z2", "cam_id": "cam_b"})
    errors = errors_of(cams, zones, prof, check_env=True)
    assert any("cameras[0].url_env = 'CAM_A_URL': chưa có trong .env" in e for e in errors), errors
    assert any("cameras[1].url = 'videos/khong_co.mp4': không tìm thấy file" in e for e in errors), errors

    monkeypatch.setenv("CAM_A_URL", "rtsp://u:p@10.0.0.1/live")
    cams["cameras"][1]["url"] = __file__
    assert errors_of(cams, zones, prof, check_env=True) == []


def test_k4_nhieu_loi_bao_du_mot_luot():
    """Tình huống thật: config có nhiều chỗ sai ở cả 3 file.
    Giả lập: 1 lỗi camera, 1 lỗi vùng, 1 lỗi profile cùng lúc.
    Đỏ thì hỏng gì: sửa một lỗi, chạy lại mới thấy lỗi tiếp theo; mỗi vòng mất vài phút dựng pipeline.
    """
    cams, zones, prof = base()
    cams["cameras"][0]["profile"] = "x"
    zones["zones"][0]["version"] = 0
    del prof["mux"]
    errors = errors_of(cams, zones, prof)
    for part in ("cameras[0].profile", "zones[0].version", "mux: thiếu"):
        assert any(part in e for e in errors), (part, errors)


@pytest.mark.parametrize("cams,zones", [("cameras.yaml", "zones.yaml"), ("cameras.office.yaml", "zones.office.yaml")])
def test_k5_config_dang_giao_phai_hop_le(cams, zones):
    """Tình huống thật: giao một bộ config mẫu mà chính nó đã sai.
    Giả lập: kiểm từng bộ cameras + zones trong repo với profiles/overview.yaml (không kiểm .env / file video / container).
    Đỏ thì hỏng gì: người nhận chạy lên là bị từ chối khởi động.
    """
    assert validate_all(load(cams), load(zones), load("profiles/overview.yaml"),
                        ROOT, PROFILE_DIR, check_env=False) == []