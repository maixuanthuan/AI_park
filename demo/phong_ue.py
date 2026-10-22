"""Module phong_ue (PoC): động vật NGỒI XỔM (dáng phóng uế) giữ đủ lâu -> cảnh báo. Không IO.

Yêu cầu (Thuận chốt 2026-10-22, đề xuất BA tư vấn khách): chỉ cần biết ĐỘNG VẬT đang phóng uế (không phân biệt chó / mèo,
không phân biệt tiểu / đại tiện) + ảnh bằng chứng. Chó đực nhấc chân đi tiểu: hông không hạ -> KHÔNG bắt được.
Đầu vào: FrameMeta có đối tượng lớp "animal" (detector DeepLabCut chỉ biết "động vật bốn chân") kèm keypoint (Obj.kp, model SuperAnimal-Quadruped của DeepLabCut, xem run_demo.py).
Định nghĩa (đo 2026-10-22 trên 1 clip camera chuông cửa, chó đen ~100 px, đáp án phóng uế 16-22s):
- Chiều cao hông = (y trung bình chân sau - y gốc đuôi) / chiều dài thân (cổ -> gốc đuôi), tính bằng PIXEL
  (toạ độ tỉ lệ 0..1 thì trục x, y khác thước khi khung không vuông). Chia cho thân: chó to / nhỏ, gần / xa so được nhau.
  Ngồi xổm -> hông hạ: trung vị 0.54 lúc phóng uế vs 0.64 lúc đứng / đi cùng cỡ; tách theo từng frame 0.80 (1.0 = tách hết).
- Frame "ngồi xổm" khi có ít nhất 1 con vật đo được có chiều cao hông < ratio_thr. Đo được = cổ, gốc đuôi và >= 1 chân sau
  có độ tin >= kp_conf, thân >= min_body_px (chó quá nhỏ thì keypoint nhảy, không tin).
- Báo khi QUÁ NỬA số frame trong hold_sec giây gần nhất là ngồi xổm; hết khi không quá nửa trong clear_sec giây, chỉ xét
  dữ liệu từ sau lúc báo (cùng luật bang_ron / crowd, bài học D10).
  Thử trên clip đo: ngưỡng 0.56, cửa sổ 3-4s -> báo đúng 1 lần ~18-22.5s; ngưỡng 0.58 đã báo nhầm lúc chó cúi ngửi (10.5s).
  -> biên an toàn HẸP, ngưỡng mới đúng cho 1 clip: phải kiểm trên nhiều clip (co / khong) trước khi tin.
Giới hạn đã biết: ngưỡng đo trên 1 clip chó, góc camera đổi là phải đo lại (góc cao nhìn xuống làm hông "dẹt");
mèo ngồi nghỉ hông vốn thấp -> có thể báo nhầm, chưa có clip mèo; vùng = cả khung (PoC).
"""
import math
import statistics
from collections import deque

from meta import Line, Text

ANIMAL = "animal"
NECK, TAIL = "neck_base", "tail_base"
BACK_PAWS = ("back_left_paw", "back_right_paw")


def hip_ratio(o, w, h, kp_conf, min_body_px):
    """Chiều cao hông / thân của 1 con vật (Obj có kp), hoặc None nếu không đo được. w, h: kích thước khung (pixel)."""
    kp = o.kp or {}
    ok = {k: (v[0] * w, v[1] * h) for k, v in kp.items() if v[2] >= kp_conf}
    paws = [ok[p] for p in BACK_PAWS if p in ok]
    if NECK not in ok or TAIL not in ok or not paws:
        return None
    body = math.dist(ok[NECK], ok[TAIL])
    if body < min_body_px:
        return None
    return (sum(p[1] for p in paws) / len(paws) - ok[TAIL][1]) / body


class PhongUeModule:
    """Phóng uế cho MỘT nguồn, cả khung hình (PoC chưa có vùng)."""

    def __init__(self, cam_id, ratio_thr, hold_sec, clear_sec, kp_conf=0.3, min_body_px=40, row=0):
        self.cam_id, self.thr, self.hold, self.clear = cam_id, ratio_thr, hold_sec, clear_sec
        self.kp_conf, self.min_body, self.row = kp_conf, min_body_px, row
        self.keep = max(hold_sec, clear_sec)
        self.samples = deque()  # (ts, 0 / 1 có chó ngồi xổm)
        self.alarm, self.since = False, None
        self.animals = []       # [(Obj, chiều cao hông)] frame gần nhất, chỉ để vẽ
        self.wh = (1, 1)        # kích thước khung frame gần nhất, chỉ để vẽ (vạch ngưỡng tính bằng pixel)

    def _frac(self, ts, sec):
        if not self.samples or ts - self.samples[0][0] < sec:
            return None  # chưa đủ dữ liệu phủ cửa sổ
        v = [x for t, x in self.samples if t >= ts - sec]
        return sum(v) / len(v)

    def update(self, meta):
        if meta.cam_id != self.cam_id:
            raise ValueError(f"frame của {meta.cam_id} đưa vào module phóng uế của {self.cam_id}")
        self.wh = (meta.src_w, meta.src_h)
        self.animals = [(o, hip_ratio(o, meta.src_w, meta.src_h, self.kp_conf, self.min_body))
                        for o in meta.objects if o.cls == ANIMAL]
        squat = any(r is not None and r < self.thr for _, r in self.animals)
        if self.samples and meta.ts - self.samples[-1][0] > self.keep:
            self.samples.clear()  # mất hình lâu: mẫu cũ không còn nói gì về hiện tại
        self.samples.append((meta.ts, 1 if squat else 0))
        while len(self.samples) > 1 and self.samples[1][0] <= meta.ts - self.keep:
            self.samples.popleft()
        if self.alarm and meta.ts - self.since < self.clear:
            return []
        sec = self.clear if self.alarm else self.hold
        f = self._frac(meta.ts, sec)
        if f is None or (f > 0.5) == self.alarm:
            return []
        self.alarm, self.since = not self.alarm, meta.ts
        payload = dict(window_sec=sec, ti_le_ngoi_xom=round(f, 2), nguong_hong_than=self.thr)
        if self.alarm:
            r = [float(r) for _, r in self.animals if r is not None]
            payload["hong_than"] = round(statistics.median(r), 2) if r else None
        return [dict(event_type="alert_start" if self.alarm else "alert_end", function="phong_ue", cam_id=self.cam_id,
                     zone_id=None, ts=meta.ts, payload=payload)]

    def flush(self):
        return []

    def draw(self):
        """Vẽ để NHÌN HÌNH LÀ THẤY LUẬT (Thuận 2026-10-22: bản đầu vẽ đường chéo gốc đuôi -> chân, không khớp phép đo).
        Mỗi con vật đo được: C = cổ, D = gốc đuôi, P = chân sau (trung bình 2 chân).
          - đường trắng C-D: THƯỚC (chiều dài thân = 1)
          - thanh đứng từ D xuống ngang mức P: CHIỀU CAO HÔNG (đúng thứ được đo), vàng = đứng, đỏ = ngồi xổm
          - vạch trắng ngang trên thanh đứng ở độ sâu ngưỡng x thân dưới D: chân sau QUA vạch = đứng, CHƯA tới vạch = ngồi xổm
          - chữ dưới con vật: "hong 0.51 < 0.56 NGOI XOM" / "hong 0.70 DUNG"
        Dòng trên cùng: % frame ngồi xổm trong cửa sổ hold_sec (quá 50% thì báo), để thấy cả luật thời gian."""
        w, h = self.wh
        out = []
        for o, r in self.animals:
            x1, y2 = o.bbox[0], o.bbox[3]
            if r is None:
                out.append(Text(x1, y2, 0, "khong do duoc (diem mo / qua nho)", "info"))
                continue
            kp, kind = o.kp, "alert" if r < self.thr else "zone"
            (cx, cy), (dx, dy) = kp[NECK][:2], kp[TAIL][:2]
            paws = [kp[p] for p in BACK_PAWS if p in kp and kp[p][2] >= self.kp_conf]
            px, py = sum(p[0] for p in paws) / len(paws), sum(p[1] for p in paws) / len(paws)
            body = math.dist((cx * w, cy * h), (dx * w, dy * h))
            tick, mark = 0.15 * body / w, dy + self.thr * body / h   # nửa vạch ngang (tỉ lệ x); vạch ngưỡng (tỉ lệ y)
            out += [Line(cx, cy, dx, dy, "info"),                                       # thước thân
                    Line(dx, dy, dx, py, kind),                                         # chiều cao hông
                    Line(dx - tick, py, dx + tick, py, kind),                           # mức chân sau
                    Line(dx - tick, mark, dx + tick, mark, "info"),                     # vạch ngưỡng
                    Line(dx, py, px, py, kind)]                                         # nối sang chân sau thật
            out += [Text(x, y, -1, t, "info") for x, y, t in ((cx, cy, "C"), (dx, dy, "D"), (px, py, "P"))]
            out.append(Text(x1, y2, 0, f"hong {r:.2f} < {self.thr:g} NGOI XOM" if r < self.thr
                            else f"hong {r:.2f} DUNG", kind))
        ts = self.samples[-1][0] if self.samples else 0.0
        v = [x for t, x in self.samples if t >= ts - self.hold]
        pct = f"{100 * sum(v) / len(v):.0f}%" if v else "-"
        text = f"phong ue: ngoi xom {pct} / {self.hold:g}s (bao khi > 50%)" + (" -> CANH BAO" if self.alarm else "")
        out.append(Text(0.0, 0.0, self.row, text, "alert" if self.alarm else "info"))
        return out
