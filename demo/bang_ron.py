"""Module bang_ron (PoC): có băng rôn / biển ĐANG ĐƯỢC NGƯỜI CẦM trong khung, giữ đủ lâu -> cảnh báo. Không IO.

Đầu vào: FrameMeta có đối tượng lớp "banner" (độ tin trong conf) và "person" (cùng model YOLOE, xem run_demo.py).
- Mỗi frame: có băng rôn (conf >= banner_conf) "gần người" không (luật thử 2026-10-21, cùng luật tools/thu_bang_ron.py):
  chồng ngang với một khung người, cạnh dưới không cao hơn đỉnh đầu người quá nửa chiều cao người.
  Băng rôn treo cố định trên cột / cây thường cao hơn hẳn đầu người -> không tính (clip phố Đà Nẵng: 0% ở ngưỡng 0.3).
- Báo khi QUÁ NỬA số frame trong hold_sec giây gần nhất có băng rôn gần người (cùng kiểu trung vị của crowd.py:
  model nhận chập chờn, thử clip biểu tình 2026-10-21: có băng rôn 38% thời gian nhưng liên tục dài nhất chỉ 4s).
- Hết khi không quá nửa trong clear_sec giây, chỉ xét dữ liệu từ sau lúc báo (bài học crowd D10).
"""
from collections import deque

from meta import Text

BANNER, PERSON = "banner", "person"


def near_person(b, persons):
    """Băng rôn b gần người không. b, persons: (x1, y1, x2, y2), cùng hệ toạ độ."""
    for p in persons:
        h = p[3] - p[1]
        if min(b[2], p[2]) > max(b[0], p[0]) and b[3] >= p[1] - 0.5 * h and b[1] <= p[3]:
            return True
    return False


class BannerModule:
    """Băng rôn cầm tay cho MỘT nguồn, cả khung hình (PoC chưa có vùng)."""

    def __init__(self, cam_id, banner_conf, hold_sec, clear_sec, row=0):
        self.row = row  # dòng chữ thứ mấy trên video (dưới chữ của count / crowd cùng nguồn)
        self.cam_id, self.banner_conf, self.hold, self.clear = cam_id, banner_conf, hold_sec, clear_sec
        self.keep = max(hold_sec, clear_sec)
        self.samples = deque()  # (ts, 0 / 1 có băng rôn gần người)
        self.alarm, self.since, self.live = False, None, 0

    def _frac(self, ts, sec):
        if not self.samples or ts - self.samples[0][0] < sec:
            return None  # chưa đủ dữ liệu phủ cửa sổ
        v = [x for t, x in self.samples if t >= ts - sec]
        return sum(v) / len(v)

    def update(self, meta):
        if meta.cam_id != self.cam_id:
            raise ValueError(f"frame của {meta.cam_id} đưa vào module băng rôn của {self.cam_id}")
        persons = [o.bbox for o in meta.objects if o.cls == PERSON]
        held = [o for o in meta.objects if o.cls == BANNER and o.conf >= self.banner_conf and near_person(o.bbox, persons)]
        self.live = len(held)
        if self.samples and meta.ts - self.samples[-1][0] > self.keep:
            self.samples.clear()  # mất hình lâu: mẫu cũ không còn nói gì về hiện tại
        self.samples.append((meta.ts, 1 if held else 0))
        while len(self.samples) > 1 and self.samples[1][0] <= meta.ts - self.keep:
            self.samples.popleft()
        if self.alarm and meta.ts - self.since < self.clear:
            return []
        sec = self.clear if self.alarm else self.hold
        f = self._frac(meta.ts, sec)
        if f is None or (f > 0.5) == self.alarm:
            return []
        self.alarm, self.since = not self.alarm, meta.ts
        return [dict(event_type="alert_start" if self.alarm else "alert_end", function="bang_ron", cam_id=self.cam_id,
                     zone_id=None, ts=meta.ts,
                     payload=dict(window_sec=sec, ti_le_co_bang_ron=round(f, 2), banner_conf=self.banner_conf))]

    def flush(self):
        return []

    def draw(self):
        text = f"bang ron dang cam: {self.live}" + (" -> CANH BAO" if self.alarm else "")
        return [Text(0.0, 0.0, self.row, text, "alert" if self.alarm else "info")]