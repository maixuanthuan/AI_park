"""Test log vận hành (T20-T25): log là cho người vận hành / người tiếp nhận đọc, phải nói đúng và đủ, không rác.

T20-T22 chạy đúng chương trình thật với một camera chết (cổng 9 trên máy mình: luôn bị từ chối kết nối).
Không cần camera, GPU; có ffprobe hay không đều được (không có thì lỗi là "không tìm thấy ffprobe", vẫn là mất kết nối).
"""
import os
import random
import signal
import subprocess
import sys
import threading
import time

import numpy as np
import pytest
import yaml

import run_tamper
from frame_source import FFmpegSource

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def write_cfg(tmp_path, cameras, **over):
    with open(os.path.join(ROOT, "config.yaml")) as f:
        cfg = yaml.safe_load(f)
    cfg["out_dir"] = str(tmp_path / "events")
    cfg["stats_sec"] = 1
    cfg["source"].update(hwaccel=False, reconnect_sec=0.2, read_timeout=2)
    cfg["detector"].update(hold_sec=0.5)
    cfg["cameras"] = cameras
    for k, v in over.items():
        cfg[k] = v
    path = tmp_path / "config.yaml"
    path.write_text(yaml.safe_dump(cfg, allow_unicode=True))
    return path


@pytest.fixture(scope="module")
def dead_camera_log(tmp_path_factory):
    """Chạy chương trình 4.5 giây với 1 camera chết, rồi Ctrl+C. Trả về (mã thoát, các dòng log)."""
    tmp = tmp_path_factory.mktemp("dead")
    path = write_cfg(tmp, [{"id": "cam_mat", "url": "rtsp://127.0.0.1:9/live"}])
    p = subprocess.Popen([sys.executable, os.path.join(ROOT, "run_tamper.py"), "--config", str(path)],
                         cwd=tmp, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    time.sleep(4.5)
    p.send_signal(signal.SIGINT)
    out, _ = p.communicate(timeout=20)
    return p.returncode, out.splitlines()


def test_t20_mat_ket_noi_chi_bao_mot_dong(dead_camera_log):
    """Tình huống thật: camera mất điện cả đêm, chương trình thử kết nối lại mỗi vài giây.
    Giả lập: camera chết (từ chối kết nối), thử lại mỗi 0.2s trong 4.5s; lỗi ffprobe mỗi lần có địa chỉ bộ nhớ khác nhau.
    Đỏ thì hỏng gì: mỗi lần thử in 1 dòng (đo 2026-10-16: 8 dòng / 40s / camera, 50 camera một đêm ~430.000 dòng),
    log quan trọng bị chôn vùi.
    """
    code, lines = dead_camera_log
    assert code == 0, lines
    lost = [ln for ln in lines if "cam_mat: mất kết nối" in ln]
    assert len(lost) == 1, lines
    retries = [ln for ln in lines if "thống kê" in ln]
    assert any(int(ln.split("thử kết nối lại ")[1].split()[0]) >= 3 for ln in retries), lines  # đã thử lại nhiều lần thật


def test_t21_thong_ke_dinh_ky_ke_ca_khi_bang_0(dead_camera_log):
    """Tình huống thật: hệ chạy cả ngày, người vận hành cần biết nó còn sống hay đang treo.
    Giả lập: stats_sec = 1, chạy 4.5s, không có frame nào (camera chết).
    Đỏ thì hỏng gì: log im lặng hàng giờ, không phân biệt được "không có chuyện gì" với "chương trình treo".
    """
    code, lines = dead_camera_log
    stats = [ln for ln in lines if "thống kê" in ln]
    assert len(stats) >= 3, lines
    assert all("1 camera" in ln and "0 frame" in ln for ln in stats), stats


def test_t22_tong_ket_dem_ca_canh_bao_mat_tin_hieu(dead_camera_log):
    """Tình huống thật: cuối buổi chạy, đọc dòng tổng kết để báo cáo số cảnh báo.
    Giả lập: camera chết đủ lâu để phát cảnh báo no_signal, rồi dừng chương trình.
    Đỏ thì hỏng gì: tổng kết ghi "0 cảnh báo" trong khi đã có cảnh báo (gặp thật 2026-10-16), số liệu báo cáo sai.
    """
    code, lines = dead_camera_log
    assert any("tamper_start" in ln and "no_signal" in ln for ln in lines), lines
    summary = [ln for ln in lines if "cam_mat: dừng giữa chừng" in ln]
    assert len(summary) == 1 and "1 cảnh báo" in summary[0] and "no_signal" in summary[0], lines


class FakeSource:
    """Nguồn camera giả: 3 lần mở đầu thất bại (lỗi mỗi lần có địa chỉ bộ nhớ khác nhau), rồi có hình 6 frame."""
    stop = None

    def __init__(self, url, fps, out_width, hwaccel, read_timeout, name):
        self.is_file, self.ts, self.last_error = False, 0.0, None
        self.n_open = self.n_read = 0
        rng = np.random.default_rng(0)
        self.frame = rng.integers(0, 256, (480, 640, 3), np.uint8)

    def open(self):
        self.n_open += 1
        if self.n_open <= 3:
            self.last_error = f"[tcp @ 0x{random.getrandbits(40):x}] Connection refused"
            return False
        self.last_error = None
        return True

    def read(self):
        self.n_read += 1
        if self.n_read > 6:
            FakeSource.stop.set()
            return False, None
        self.ts = time.time()
        return True, self.frame

    def close(self):
        pass


def test_t23_co_hinh_lai_bao_mot_dong_kem_thoi_gian_mat(monkeypatch, caplog, cfg):
    """Tình huống thật: camera mất kết nối một lúc rồi có lại.
    Giả lập: nguồn giả thất bại 3 lần (lỗi khác nhau chỉ ở địa chỉ bộ nhớ), lần 4 có hình.
    Đỏ thì hỏng gì: không biết camera đã có lại chưa, mất bao lâu; hoặc mỗi lần thử lại in 1 dòng mới.
    """
    monkeypatch.setattr(run_tamper, "FFmpegSource", FakeSource)
    stop = threading.Event()
    FakeSource.stop = stop
    src_cfg = {"hwaccel": False, "out_width": 0, "read_timeout": 2, "sample_fps": 2, "reconnect_sec": 0.01}
    stats = run_tamper.Stats(["fake"])
    caplog.set_level("INFO", logger="tamper")
    run_tamper.run_camera({"id": "fake"}, "rtsp://x/y", cfg(), src_cfg, None, stop, [], {}, False, stats)

    msgs = [r.getMessage() for r in caplog.records]
    assert len([m for m in msgs if "fake: mất kết nối" in m]) == 1, msgs
    back = [m for m in msgs if "fake: có hình lại" in m]
    assert len(back) == 1 and "đã thử 3 lần" in back[0], msgs


class FakeVideo(FakeSource):
    """File video giả: mở được ngay, 4 frame rồi hết (không lỗi)."""

    def __init__(self, *args, **kw):
        super().__init__(*args, **kw)
        self.is_file = True

    def open(self):
        self.last_error = None
        return True

    def read(self):
        self.n_read += 1
        if self.n_read > 4:
            return False, None
        self.ts = self.n_read / 2
        return True, self.frame


def test_t26_het_video_thi_thong_ke_ghi_het_video(monkeypatch, cfg):
    """Tình huống thật: chạy nhiều video cùng lúc, video ngắn hết trước, video dài còn chạy.
    Giả lập: nguồn file giả 4 frame rồi hết.
    Đỏ thì hỏng gì: dòng thống kê vẫn ghi camera đó "ok" (thấy 2026-10-16), người đọc tưởng nó còn đang chạy.
    """
    monkeypatch.setattr(run_tamper, "FFmpegSource", FakeVideo)
    src_cfg = {"hwaccel": False, "out_width": 0, "read_timeout": 2, "sample_fps": 2, "reconnect_sec": 0.01}
    stats = run_tamper.Stats(["vid"])
    run_tamper.run_camera({"id": "vid"}, "v.mp4", cfg(), src_cfg, None, threading.Event(), [], {}, False, stats)
    line = stats.report(1)
    assert "hết video 1" in line and "4 frame" in line, line


def _spawn_fake_ffmpeg(src, script):
    """Thay ffmpeg bằng một tiến trình Python nhỏ ghi ra byte giả; frame 2x1 pixel = 6 byte."""
    src.ow, src.oh, src.frame_bytes = 2, 1, 6
    src._spawn([sys.executable, "-c", script])


def test_t24_ffmpeg_dung_binh_thuong_thi_im_lang_chet_bat_thuong_thi_noi_ro(caplog):
    """Tình huống thật: hết video (ffmpeg thoát mã 0), hoặc mình chủ động tắt (Ctrl+C), hoặc ffmpeg tự chết vì lỗi.
    Giả lập: thay ffmpeg bằng tiến trình Python: thoát 0 / bị tắt giữa chừng / thoát mã 3 kèm câu lỗi có mật khẩu.
    Đỏ thì hỏng gì: dòng "ffmpeg dừng (code=0):" trống rỗng gây hiểu nhầm là lỗi (gặp thật 2026-10-15/16);
    hoặc ffmpeg chết thật mà không biết vì sao; hoặc lộ mật khẩu.
    """
    caplog.set_level("INFO", logger="tamper")
    new = lambda: FFmpegSource("rtsp://x/y", fps=2, out_width=0, hwaccel=False, read_timeout=10, name="t")

    src = new()  # 1. hết video: 2 frame rồi thoát mã 0
    _spawn_fake_ffmpeg(src, "import sys; sys.stdout.buffer.write(b'x' * 12)")
    assert [src.read()[0] for _ in range(3)] == [True, True, False]
    assert src.last_error is None

    src = new()  # 2. mình chủ động tắt giữa chừng
    _spawn_fake_ffmpeg(src, "import sys, time; sys.stdout.buffer.write(b'x' * 6); sys.stdout.flush(); time.sleep(30)")
    assert src.read()[0] is True
    threading.Timer(0.3, src.close).start()
    assert src.read()[0] is False
    assert src.last_error is None

    # 3. tự chết vì lỗi, câu lỗi (có mật khẩu) tới SAU khi tiến trình đã thoát: dựng chắc chắn bằng một tiến trình
    # con giữ ống stderr, ghi câu lỗi muộn 0.3s. Không đợi đọc hết stderr là mất đúng câu quan trọng nhất.
    # (Thử cách "in thật nhiều dòng rồi chạy nhiều lần" ngày 2026-10-16: không bao giờ lộ lỗi, nên bỏ.)
    src = new()
    _spawn_fake_ffmpeg(src, "import subprocess, sys; subprocess.Popen([sys.executable, '-c', "
                            "\"import sys, time; time.sleep(0.3); sys.stderr.write('boom rtsp://admin:Secret1@1.2.3.4/x\\\\n')\"], "
                            "stdout=subprocess.DEVNULL); sys.exit(3)")
    assert src.read()[0] is False
    assert "code=3" in src.last_error and "boom" in src.last_error and "Secret1" not in src.last_error

    assert not [r for r in caplog.records if "ffmpeg dừng" in r.getMessage()]  # nguồn đọc không tự in log nữa


def test_t25_loi_code_trong_thread_camera_thi_dung_toan_bo_ma_khac_0(tmp_path, monkeypatch, caplog):
    """Tình huống thật: một lỗi code (bug) làm thread của một camera chết.
    Giả lập: thay hàm chạy camera bằng hàm ném lỗi, chạy main() với config hợp lệ.
    Đỏ thì hỏng gì: chương trình thoát mã 0 như tắt bình thường (gặp thật ở T16 trên code cũ), Docker không khởi động
    lại, lỗi bị che; hoặc không có traceback để dev sửa.
    """
    (tmp_path / "cam.mp4").write_bytes(b"")
    path = write_cfg(tmp_path, [{"id": "cam_a", "url": "cam.mp4"}])
    monkeypatch.chdir(tmp_path)

    def boom(cam, *args):
        raise RuntimeError("bug giả trong code")

    monkeypatch.setattr(run_tamper, "run_camera", boom)
    caplog.set_level("INFO", logger="tamper")
    assert run_tamper.main(["--config", str(path)]) == 1
    fatal = [r for r in caplog.records if r.levelname == "ERROR" and "cam_a" in r.getMessage()]
    assert fatal and fatal[0].exc_info is not None  # có traceback đầy đủ
