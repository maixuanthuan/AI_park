"""Đọc camera / file video bằng ffmpeg subprocess, giải mã trên GPU (NVDEC) hoặc CPU.

read() trả về frame BGR uint8 (h, w, 3), đã giảm xuống `fps` frame/s ngay trong ffmpeg.
GPU: -hwaccel cuda -> fps -> (scale_cuda) -> hwdownload -> bgr24 qua pipe.
"""
import json
import logging
import os
import re
import signal
import subprocess
import threading
import time
from collections import deque

import numpy as np

log = logging.getLogger("tamper")

# Che tới dấu @ CUỐI CÙNG trước dấu / đầu tiên: mật khẩu dán nguyên văn có thể chứa @ (vd p@ss123).
# Bản cũ dừng ở @ đầu tiên -> lộ phần sau (phát hiện 2026-10-15, khoá bằng test T11b).
_CRED = re.compile(r"//[^/\s]*@")


def mask(s):
    """Che user:pass trong URL trước khi in log."""
    return _CRED.sub("//***@", s)


def _tail(lines, n=5):
    lines = [l for l in lines if "no version information" not in l]
    return mask(" | ".join(lines[-n:]))


class FFmpegSource:
    def __init__(self, url, fps, out_width, hwaccel, read_timeout, name):
        self.url = url
        self.is_file = os.path.isfile(url)
        self.fps = float(fps)
        self.out_width = int(out_width or 0)  # 0 = giữ độ phân giải gốc
        self.hwaccel = bool(hwaccel)
        self.read_timeout = float(read_timeout)
        self.name = name
        self.proc = None
        self.err = deque(maxlen=20)
        self.ow = self.oh = self.frame_bytes = 0
        self.frame_idx = 0
        self.ts = 0.0
        self.last_t = 0.0
        # Lý do lần mở / đọc gần nhất thất bại (None = không lỗi). Nguồn đọc KHÔNG tự in log:
        # runner quyết định in gì, in lúc nào (mỗi sự cố 2 dòng), xem run_tamper.run_camera.
        self.last_error = None
        self.closing = False      # mình chủ động tắt (Ctrl+C / dừng chương trình): ffmpeg chết là bình thường
        self.timed_out = False    # watchdog đã tắt ffmpeg vì quá read_timeout không có frame
        self._drain_t = None

    def _probe(self):
        cmd = ["ffprobe", "-v", "error"]
        if not self.is_file:
            cmd += ["-rtsp_transport", "tcp"]
        cmd += ["-select_streams", "v:0", "-show_entries", "stream=codec_name,width,height",
                "-of", "json", self.url]
        out = subprocess.run(cmd, capture_output=True, timeout=self.read_timeout, check=True).stdout
        s = json.loads(out)["streams"][0]
        return s["codec_name"], int(s["width"]), int(s["height"])

    def open(self):
        self.last_error, self.closing, self.timed_out = None, False, False
        try:
            codec, w, h = self._probe()
        except subprocess.CalledProcessError as e:
            self.last_error = "ffprobe: " + (_tail(e.stderr.decode(errors="replace").splitlines()) or f"mã {e.returncode}")
            return False
        except Exception as e:
            self.last_error = f"ffprobe: {type(e).__name__}: {mask(str(e))}"
            return False

        if self.out_width and self.out_width < w:
            ow, oh = self.out_width, int(round(h * self.out_width / w / 2)) * 2
        else:
            ow, oh = w, h
        self.ow, self.oh, self.frame_bytes = ow, oh, ow * oh * 3

        cmd = ["ffmpeg", "-hide_banner", "-loglevel", "error", "-nostdin"]
        if not self.is_file:
            cmd += ["-rtsp_transport", "tcp"]
        # fps đứng đầu: bỏ frame thừa ngay trên GPU, chỉ scale + chép về CPU đúng số frame cần
        vf = [f"fps={self.fps}"]
        if self.hwaccel:
            cmd += ["-hwaccel", "cuda", "-hwaccel_output_format", "cuda"]
            if (ow, oh) != (w, h):
                vf.append(f"scale_cuda={ow}:{oh}")
            vf += ["hwdownload", "format=nv12"]
        elif (ow, oh) != (w, h):
            vf.append(f"scale={ow}:{oh}")
        cmd += ["-i", self.url, "-an", "-vf", ",".join(vf), "-pix_fmt", "bgr24", "-f", "rawvideo", "-"]

        self._spawn(cmd)
        log.info("%s: %s %dx%d -> %dx%d @%sfps, decode=%s", self.name, codec, w, h, ow, oh,
                 self.fps, "gpu" if self.hwaccel else "cpu")
        return True

    def read(self):
        proc = self.proc
        if proc is None:
            return False, None
        buf = proc.stdout.read(self.frame_bytes)
        if len(buf) < self.frame_bytes:
            proc.wait()
            if self._drain_t is not None:
                self._drain_t.join(timeout=1)  # đợi đọc hết stderr, không thì câu lỗi bị cụt
            if self.timed_out:
                self.last_error = f"quá {self.read_timeout:g}s không có frame, đã tắt ffmpeg"
            elif self.closing or proc.returncode == 0:
                self.last_error = None  # mình chủ động tắt, hoặc hết video: không phải lỗi
            else:
                tail = _tail(self.err)
                self.last_error = f"ffmpeg dừng (code={proc.returncode})" + (f": {tail}" if tail else "")
            return False, None
        self.last_t = time.time()
        self.frame_idx += 1
        self.ts = self.frame_idx / self.fps if self.is_file else self.last_t
        return True, np.frombuffer(buf, np.uint8).reshape(self.oh, self.ow, 3).copy()

    def close(self):
        self.closing = True
        proc, self.proc = self.proc, None
        if proc is not None:
            self._kill(proc)

    # ---------- internal ----------
    def _spawn(self, cmd):
        self.err.clear()
        self.proc = subprocess.Popen(cmd, stdin=subprocess.DEVNULL, stdout=subprocess.PIPE,
                                     stderr=subprocess.PIPE, start_new_session=True)
        self.last_t = time.time()
        self._drain_t = threading.Thread(target=self._drain, args=(self.proc,), daemon=True)
        self._drain_t.start()
        threading.Thread(target=self._watchdog, args=(self.proc,), daemon=True).start()

    def _drain(self, proc):
        for line in proc.stderr:
            self.err.append(line.decode(errors="replace").rstrip())

    def _watchdog(self, proc):
        while proc.poll() is None:
            time.sleep(1.0)
            if time.time() - self.last_t > self.read_timeout:
                self.timed_out = True
                self._kill(proc)
                break

    @staticmethod
    def _kill(proc):
        if proc.poll() is not None:
            return
        try:
            os.killpg(proc.pid, signal.SIGTERM)
            proc.wait(timeout=3)
        except subprocess.TimeoutExpired:
            os.killpg(proc.pid, signal.SIGKILL)
            proc.wait()
        except ProcessLookupError:
            pass
