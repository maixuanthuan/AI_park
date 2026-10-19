"""Module count: đếm HIỆN DIỆN (bao nhiêu người / chó / mèo đang ở trong vùng). Không phụ thuộc DeepStream.

Mỗi frame: đếm số đối tượng theo lớp có điểm chân trong từng vùng. Hết cửa sổ report_sec của vùng: trả về
1 báo cáo trung vị + max theo lớp. Trung vị chống nhiễu (model sót / thừa vài frame không làm lệch số),
max giữ đỉnh. Không dùng track_id: đếm hiện diện không có khái niệm "đếm trùng" (mỗi frame một người một khung).
"""
import statistics

from zones import anchor, contains


class _Window:
    """Cửa sổ của một vùng: số đếm từng frame theo lớp. Mọi lớp đều được ghi ở MỌI frame (0 nếu không thấy):
    bản thử ds_count.py chỉ ghi lớp từ lúc nó xuất hiện -> lớp hiếm bị trung vị cao hơn thật (chứng minh 2026-10-19)."""

    def __init__(self, classes):
        self.classes = classes
        self._reset()

    def _reset(self):
        self.start = self.last = None
        self.samples = {c: [] for c in self.classes}

    def add(self, ts, n):
        if self.start is None:
            self.start = ts
        for c in self.classes:
            self.samples[c].append(n[c])
        self.last = ts

    def close(self):
        frames = len(self.samples[self.classes[0]])
        out = dict(window_start=self.start, window_end=self.last, frames=frames,
                   # median luôn float: số frame chẵn ra x.5, lẻ ra số nguyên -> kiểu không ổn định cho backend (thấy 2026-10-19)
                   counts={c: {"median": float(statistics.median(v)), "max": max(v)} for c, v in self.samples.items()})
        self._reset()
        return out


class CountModule:
    """Đếm hiện diện cho MỘT camera, mọi vùng count của camera đó."""

    def __init__(self, cam_id, zones, classes):
        self.cam_id = cam_id
        self.classes = tuple(classes)
        self.zones = [z for z in zones if z["cam_id"] == cam_id and z["function"] == "count"]
        self.win = {z["zone_id"]: _Window(self.classes) for z in self.zones}

    def update(self, meta):
        """meta: FrameMeta của camera này. Trả về danh sách báo cáo (thường rỗng)."""
        if meta.cam_id != self.cam_id:
            raise ValueError(f"frame của {meta.cam_id} đưa vào module đếm của {self.cam_id}")
        feet = [(o.cls, anchor(o.bbox)) for o in meta.objects if o.cls in self.classes]
        events = []
        for z in self.zones:
            w, report_sec = self.win[z["zone_id"]], z["params"]["report_sec"]
            # mất hình lâu hơn 1 cửa sổ: đóng cửa sổ cũ tại frame cuối, không kéo dài qua khoảng không có frame
            if w.last is not None and meta.ts - w.last > report_sec:
                events.append(self._report(z, w))
            n = dict.fromkeys(self.classes, 0)
            for cls, (x, y) in feet:
                if contains(z["polygon"], x, y):
                    n[cls] += 1
            w.add(meta.ts, n)
            if meta.ts - w.start >= report_sec:
                events.append(self._report(z, w))
        return events

    def flush(self):
        """Dừng chương trình / hết video: báo cáo phần cửa sổ còn dở (có trường frames để biết nó ngắn)."""
        return [self._report(z, self.win[z["zone_id"]]) for z in self.zones if self.win[z["zone_id"]].last is not None]

    def _report(self, z, w):
        payload = w.close()
        return dict(event_type="report", function="count", cam_id=self.cam_id, zone_id=z["zone_id"],
                    ts=payload["window_end"], payload=payload)