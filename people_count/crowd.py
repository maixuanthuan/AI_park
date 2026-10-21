"""Module crowd: ĐÁM ĐÔNG = số người trong vùng >= min_people, giữ đủ lâu. Không phụ thuộc DeepStream.

Không đoán ý định từ hình ảnh: chỉ dùng số người trong vùng (cùng cách đếm với module count: chân trong vùng).
Chống nhiễu bằng TRUNG VỊ số người trong cửa sổ trượt: model sót / thừa vài người ở vài frame (thấy 2026-10-21 trên
video demo: 8 -> 10 -> 7 giữa các frame liền nhau) không làm cảnh báo bật tắt liên tục. Xét từng frame thì một frame
sót người là đếm lại từ đầu, đám đông thật có thể không bao giờ được báo.
- Đang bình thường: trung vị hold_sec giây gần nhất >= min_people -> alert_start.
- Đang báo động: trung vị clear_sec giây gần nhất < min_people -> alert_end. Chỉ xét khi đã báo đủ clear_sec giây:
  cửa sổ "hết" chỉ chứa dữ liệu TỪ SAU lúc báo. Bản đầu xét ngay, cửa sổ còn nửa phần số thấp từ trước lúc báo
  -> báo rồi tắt sau 3.7s (demo park1 2026-10-21, 82.3 -> 86.0). Hệ quả: cảnh báo kéo dài ít nhất clear_sec.
- Chưa đủ dữ liệu phủ cửa sổ (mới chạy, vừa có hình lại sau mất hình lâu) -> chưa quyết định gì.
Giới hạn đã biết: dừng chương trình lúc đang báo động thì không gửi alert_end (chưa thật sự hết đông);
mất hình lúc đang báo động thì giữ trạng thái, có hình lại đủ clear_sec mới xét kết thúc.
"""
import statistics
from collections import deque

from meta import Line, Text
from zones import anchor, contains

CLASS = "person"  # đám đông là đám đông NGƯỜI: định nghĩa của chức năng, không phải ngưỡng


class _Zone:
    def __init__(self, z):
        self.z = z
        p = z["params"]
        self.min_people, self.hold, self.clear = p["min_people"], p["hold_sec"], p["clear_sec"]
        self.keep = max(self.hold, self.clear)
        self.samples = deque()  # (ts, số người), phủ ít nhất `keep` giây gần nhất
        self.alarm = False
        self.since = None  # lúc bắt đầu báo động
        self.live = None  # số người frame gần nhất, chỉ để vẽ

    def add(self, ts, n):
        if self.samples and ts - self.samples[-1][0] > self.keep:
            self.samples.clear()  # mất hình lâu hơn cửa sổ: mẫu cũ không còn nói gì về hiện tại
        self.samples.append((ts, n))
        self.live = n
        # bỏ mẫu cũ nhưng GIỮ 1 mẫu ở / trước mốc ts - keep, để biết dữ liệu đã phủ đủ cửa sổ
        while len(self.samples) > 1 and self.samples[1][0] <= ts - self.keep:
            self.samples.popleft()

    def window(self, ts, sec):
        """(trung vị, max) số người trong `sec` giây gần nhất; None nếu dữ liệu chưa phủ đủ `sec` giây."""
        if ts - self.samples[0][0] < sec:
            return None
        v = [n for t, n in self.samples if t >= ts - sec]
        return float(statistics.median(v)), max(v)


class CrowdModule:
    """Đám đông cho MỘT camera, mọi vùng crowd của camera đó."""

    def __init__(self, cam_id, zones):
        self.cam_id = cam_id
        self.zones = [_Zone(z) for z in zones if z["cam_id"] == cam_id and z["function"] == "crowd"]

    def update(self, meta):
        """meta: FrameMeta của camera này. Trả về danh sách event alert_start / alert_end (thường rỗng)."""
        if meta.cam_id != self.cam_id:
            raise ValueError(f"frame của {meta.cam_id} đưa vào module đám đông của {self.cam_id}")
        feet = [anchor(o.bbox) for o in meta.objects if o.cls == CLASS]
        events = []
        for zs in self.zones:
            zs.add(meta.ts, sum(1 for x, y in feet if contains(zs.z["polygon"], x, y)))
            if zs.alarm and meta.ts - zs.since < zs.clear:
                continue  # chưa đủ clear_sec dữ liệu từ sau lúc báo
            sec = zs.clear if zs.alarm else zs.hold
            w = zs.window(meta.ts, sec)
            if w is None or (w[0] >= zs.min_people) == zs.alarm:
                continue
            zs.alarm, zs.since = not zs.alarm, meta.ts
            events.append(dict(event_type="alert_start" if zs.alarm else "alert_end", function="crowd",
                               cam_id=self.cam_id, zone_id=zs.z["zone_id"], ts=meta.ts,
                               payload=dict(min_people=zs.min_people, window_sec=sec, median=w[0], max=w[1])))
        return events

    def flush(self):
        """Dừng chương trình: không có gì để gửi (xem giới hạn ở đầu file)."""
        return []

    def draw(self):
        """Viền vùng (đỏ khi đang báo động) + số người frame đang hiện / ngưỡng. Dòng 1 (row=1): vùng count
        cùng chỗ thì dòng 0 là của count, không đè nhau."""
        out = []
        for zs in self.zones:
            poly, kind = zs.z["polygon"], "alert" if zs.alarm else "zone"
            out += [Line(*poly[i], *poly[(i + 1) % len(poly)], kind) for i in range(len(poly))]
            x, y = min(p[0] for p in poly), min(p[1] for p in poly)
            n = "-" if zs.live is None else zs.live
            text = f"{zs.z['zone_id']} | dam dong: {n}/{zs.min_people} nguoi" + (" -> CANH BAO" if zs.alarm else "")
            out.append(Text(x, y, 1, text, "alert" if zs.alarm else "info"))
        return out