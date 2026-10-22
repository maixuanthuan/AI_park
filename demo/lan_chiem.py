"""Module lan_chiem (PoC): vật lấn chiếm (xe máy, ghế, xe đẩy...) ĐỨNG YÊN trong khung quá dwell_sec -> cảnh báo. Không IO.

Logic B (Thuận chốt 2026-10-21), KHÔNG dùng tracker: dwell tính bằng phút, mã tracker không giữ nổi lâu như vậy
(người đi qua che khuất là đổi mã, đếm lại từ đầu).
- Mỗi sample_sec giây lấy 1 mẫu (vật đứng yên hàng phút, không cần xét mọi frame).
- Mỗi khung vật được khớp với "ứng viên" CÙNG LỚP đã thấy trước đó nếu trùng chỗ (IoU >= iou). Không khớp -> ứng viên mới.
- Ứng viên ĐỨNG YÊN khi đã tồn tại >= dwell_sec và có mặt ở QUÁ NỬA số mẫu trong dwell_sec giây gần nhất
  (chịu được người đi qua che khuất, model sót vài lần: cùng kiểu "quá nửa thời gian" của crowd / bang_ron).
  Xe chạy qua: mỗi mẫu một chỗ khác -> không bao giờ đủ -> không báo.
- Có >= 1 vật đứng yên -> alert_start (kèm danh sách vật); không còn vật nào -> alert_end. Danh sách đổi giữa chừng
  không bắn event mới (một quán cóc không thành mười mấy tin).
- Vào và ra KHÁC ngưỡng: vào = quá nửa số mẫu (như trên); ra = KHÔNG THẤY LẠI LẦN NÀO trong dwell_sec (ứng viên bị xoá).
  Bản đầu dùng cùng một ngưỡng "quá nửa" cho cả hai: vật đứng yên mà model nhận sát nửa số lần (đổi nhãn, bị che)
  -> báo / hết mỗi 5s, 174 event / 15 phút cho 1 cái ghế (giả lập 2026-10-22). Chỉ áp bài học D10 của crowd (xét hết
  bằng dữ liệu sau lúc báo) vẫn còn 43 event; tách ngưỡng ra -> 1 event. Giá: vật dọn đi thật thì dwell_sec sau mới hết.
- Ứng viên không thấy lại quá dwell_sec thì xoá (bộ nhớ không phình theo số xe chạy qua).
Giới hạn: camera phải đứng yên (camera xoay / quay tay thì mọi thứ "di chuyển"); vùng = cả khung (PoC), thực tế phải
vẽ vùng vỉa hè và tránh đồ cố định hợp lệ (ghế đá, bồn cây); hai vật đỗ chồng đúng chỗ nhau tính là một.
"""
from collections import Counter, deque

from meta import Line, Text


def iou(a, b):
    w, h = min(a[2], b[2]) - max(a[0], b[0]), min(a[3], b[3]) - max(a[1], b[1])
    if w <= 0 or h <= 0:
        return 0.0
    inter = w * h
    return inter / ((a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter)


class LanChiemModule:
    """Lấn chiếm cho MỘT nguồn, cả khung hình (PoC chưa có vùng)."""

    def __init__(self, cam_id, classes, dwell_sec, sample_sec=5.0, iou_thr=0.5, row=0):
        self.cam_id, self.classes, self.dwell, self.sample = cam_id, set(classes), dwell_sec, sample_sec
        self.iou_thr, self.row = iou_thr, row
        self.samples = deque()  # thời điểm các mẫu trong dwell_sec gần nhất
        self.cands = []         # [{"cls", "bbox", "first", "hits": deque(ts)}]
        self.alarm, self.static = False, []

    def update(self, meta):
        if meta.cam_id != self.cam_id:
            raise ValueError(f"frame của {meta.cam_id} đưa vào module lấn chiếm của {self.cam_id}")
        ts = meta.ts
        if self.samples and ts - self.samples[-1] < self.sample:
            return []
        if self.samples and ts - self.samples[-1] > self.dwell:
            self.samples.clear()  # mất hình lâu hơn dwell: mẫu cũ không còn nói gì về hiện tại
            self.cands = []
        self.samples.append(ts)
        while self.samples and self.samples[0] < ts - self.dwell:
            self.samples.popleft()

        free = list(self.cands)
        for o in sorted((o for o in meta.objects if o.cls in self.classes), key=lambda o: -o.conf):
            best = max((c for c in free if c["cls"] == o.cls), key=lambda c: iou(c["bbox"], o.bbox), default=None)
            if best is not None and iou(best["bbox"], o.bbox) >= self.iou_thr:
                free.remove(best)
                best["bbox"] = o.bbox
                best["hits"].append(ts)
            else:
                self.cands.append({"cls": o.cls, "bbox": o.bbox, "first": ts, "hits": deque([ts])})
        for c in self.cands:
            while c["hits"] and c["hits"][0] < ts - self.dwell:
                c["hits"].popleft()
        self.cands = [c for c in self.cands if c["hits"]]  # không thấy lại quá dwell_sec: xoá
        n = len(self.samples)
        for c in self.cands:  # đã đứng yên thì giữ tới khi bị xoá (không thấy lại lần nào trong dwell_sec), xem đầu file
            if ts - c["first"] >= self.dwell and len(c["hits"]) > n / 2:
                c["locked"] = True
        self.static = [c for c in self.cands if c.get("locked")]

        if bool(self.static) == self.alarm:
            return []
        self.alarm = not self.alarm
        payload = dict(dwell_sec=self.dwell)
        if self.alarm:
            payload["vat"] = [dict(cls=c["cls"], bbox=[round(v, 3) for v in c["bbox"]], tu_luc=c["first"])
                              for c in self.static]
        return [dict(event_type="alert_start" if self.alarm else "alert_end", function="lan_chiem",
                     cam_id=self.cam_id, zone_id=None, ts=ts, payload=payload)]

    def flush(self):
        return []

    def draw(self):
        out = []
        for c in self.static:  # khung đỏ dày quanh vật đứng yên
            x1, y1, x2, y2 = c["bbox"]
            out += [Line(x1, y1, x2, y1, "alert"), Line(x2, y1, x2, y2, "alert"),
                    Line(x2, y2, x1, y2, "alert"), Line(x1, y2, x1, y1, "alert")]
        kinds = Counter(c["cls"] for c in self.static)
        text = f"lan chiem: {len(self.static)} vat dung yen > {self.dwell:g}s" + \
            (" (" + ", ".join(f"{k} {v}" for k, v in sorted(kinds.items())) + ") -> CANH BAO" if self.static else "")
        out.append(Text(0.0, 0.0, self.row, text, "alert" if self.alarm else "info"))
        return out