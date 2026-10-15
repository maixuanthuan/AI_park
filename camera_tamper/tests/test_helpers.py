"""Test các hàm phụ: che mật khẩu trong log (T11), báo thiếu biến trong .env (T12)."""
import pytest

from frame_source import mask
from run_tamper import resolve_url


def test_t11_log_khong_lo_mat_khau_camera():
    """Tình huống thật: ffmpeg báo lỗi kết nối, câu lỗi có chứa nguyên link camera kèm user/pass.
    Giả lập: câu lỗi kiểu ffmpeg chứa rtsp://admin:Secret123@192.168.1.10/...
    Đỏ thì hỏng gì: mật khẩu camera nằm trong log, ai đọc được log là vào được camera.
    """
    line = "[rtsp @ 0x5bd7] rtsp://admin:Secret123@192.168.1.10:554/Streaming/Channels/102: Connection refused"
    out = mask(line)
    assert "Secret123" not in out
    assert "admin" not in out
    assert "192.168.1.10" in out  # vẫn giữ địa chỉ để devops biết camera nào lỗi
    assert mask("rtsp://192.168.1.10:554/live") == "rtsp://192.168.1.10:554/live"  # không có user/pass thì giữ nguyên


def test_t11b_mat_khau_co_ky_tu_a_cung_khong_lo():
    """Tình huống thật: mật khẩu camera có ký tự @ (vd p@ss123), devops dán nguyên văn vào .env.
    Giả lập: câu lỗi kiểu ffmpeg chứa rtsp://admin:p@ss123@192.168.1.10/...
    Đỏ thì hỏng gì: một phần mật khẩu (phần sau chữ @ đầu tiên) nằm trong log.
    """
    line = "[rtsp @ 0x5bd7] rtsp://admin:p@ss123@192.168.1.10:554/Streaming/Channels/102: Connection refused"
    out = mask(line)
    assert "ss123" not in out
    assert "admin" not in out
    assert "192.168.1.10" in out


def test_t12_thieu_bien_env_bao_dung_ten_bien(monkeypatch):
    """Tình huống thật: devops khai báo camera trong config nhưng quên điền link vào .env.
    Giả lập: gọi resolve_url khi biến CAM_TEST_URL không tồn tại, rồi khi đã điền.
    Đỏ thì hỏng gì: chương trình chết với lỗi khó hiểu, devops không biết phải điền biến nào.
    """
    cam = {"id": "cam_test", "url_env": "CAM_TEST_URL"}
    monkeypatch.delenv("CAM_TEST_URL", raising=False)
    with pytest.raises(ValueError, match="CAM_TEST_URL"):
        resolve_url(cam)

    monkeypatch.setenv("CAM_TEST_URL", "rtsp://u:p@10.0.0.1/stream")
    assert resolve_url(cam) == "rtsp://u:p@10.0.0.1/stream"
