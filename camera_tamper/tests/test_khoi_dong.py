"""Test hộp đen: chạy đúng chương trình run_tamper.py với config sai, hỏi đúng câu ngoài đời:
"nó có chịu khởi động không, và có nói rõ sai ở đâu không?"

Không cần camera hay GPU: config sai thì chương trình phải thoát trước khi mở camera nào.
"""
import os
import subprocess
import sys

import yaml

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TIMEOUT = 15  # config sai phải thoát trong vài giây; quá thời gian này = chương trình vẫn đang chạy


def run_with(tmp_path, mutate):
    """Lấy config.yaml đang giao, thay camera bằng 1 file video giả (có tồn tại), sửa theo `mutate`, rồi chạy."""
    with open(os.path.join(ROOT, "config.yaml")) as f:
        cfg = yaml.safe_load(f)
    (tmp_path / "cam.mp4").write_bytes(b"")  # file có thật nhưng rỗng: chương trình cũ sẽ thử mở mãi
    cfg["out_dir"] = str(tmp_path / "events")
    cfg["cameras"] = [{"id": "cam_a", "url": "cam.mp4"}]
    mutate(cfg)
    path = tmp_path / "config.yaml"
    path.write_text(yaml.safe_dump(cfg, allow_unicode=True))
    try:
        r = subprocess.run([sys.executable, os.path.join(ROOT, "run_tamper.py"), "--config", str(path)],
                           cwd=tmp_path, capture_output=True, text=True, timeout=TIMEOUT)
    except subprocess.TimeoutExpired:
        raise AssertionError(f"chương trình vẫn chạy sau {TIMEOUT}s với config sai (lẽ ra phải từ chối khởi động)")
    return r.returncode, r.stdout + r.stderr


def test_t14_go_nham_ten_muc_thi_khong_khoi_dong(tmp_path):
    """Tình huống thật: devops gõ "hold_secs: 10" (thừa chữ s) cho camera công viên.
    Giả lập: config đủ mục nhưng hold_sec đổi tên thành hold_secs.
    Đỏ thì hỏng gì: hệ thống âm thầm dùng giá trị khác (đo 2026-10-16: báo sau 3s thay vì 10s), chạy "xanh" mà sai.
    """
    def mutate(cfg):
        cfg["detector"]["hold_secs"] = cfg["detector"].pop("hold_sec")

    code, out = run_with(tmp_path, mutate)
    assert code == 2, out  # 2 = config sai, chưa mở camera nào (1 = đã chạy rồi mới chết: sai)
    assert "hold_secs" in out and "hold_sec" in out


def test_t15_trung_id_camera_thi_khong_khoi_dong(tmp_path):
    """Tình huống thật: dán trùng một camera vào danh sách (gặp thật ngày 2026-10-16 với park1_tamper).
    Giả lập: hai camera cùng id.
    Đỏ thì hỏng gì: mỗi cảnh báo gửi 2 lần, ảnh bằng chứng ghi đè lên nhau.
    """
    def mutate(cfg):
        cfg["cameras"].append({"id": "cam_a", "url": "cam.mp4"})

    code, out = run_with(tmp_path, mutate)
    assert code == 2, out  # 2 = config sai, chưa mở camera nào (1 = đã chạy rồi mới chết: sai)
    assert "cam_a" in out and "trùng" in out


def test_t16_nhieu_loi_thi_bao_du_mot_luot(tmp_path):
    """Tình huống thật: config có nhiều chỗ sai cùng lúc.
    Giả lập: 3 lỗi khác loại: thiếu mục, giá trị là chuỗi, nhầm 60 thay vì 0.6.
    Đỏ thì hỏng gì: sửa một lỗi, chạy lại mới thấy lỗi tiếp theo; mất thời gian, dễ bỏ cuộc giữa chừng.
    """
    def mutate(cfg):
        del cfg["detector"]["clear_sec"]
        cfg["detector"]["hold_sec"] = "10"
        cfg["detector"]["obstruct_frac"] = 60

    code, out = run_with(tmp_path, mutate)
    assert code == 2, out  # 2 = config sai, chưa mở camera nào (1 = đã chạy rồi mới chết: sai)
    for key in ("clear_sec", "hold_sec", "obstruct_frac"):
        assert f"detector.{key}" in out
