"""Camera tamper detection: obstructed (che / mờ / tối / lóa), scene_changed (xoay lệch / cảnh bị thay),
frozen, no_signal, baseline_invalid (lúc khởi động cảnh không đủ chi tiết, có thể đang bị che).

Chỉ dùng opencv + numpy, chạy CPU, không cần model.
Lõi không có IO: đưa frame + timestamp vào update(), nhận về event (dict) hoặc None.
"""
import hashlib
from collections import Counter

import cv2
import numpy as np

HOLD = "_hold"  # frame nằm trong vùng đệm: vẫn tính là bất thường nhưng không bỏ phiếu nhãn


class TamperDetector:
    DEFAULTS = dict(
        width=320,             # resize về chiều ngang này trước khi tính
        grid=8,                # chia lưới grid x grid ô
        osd_mask=[],           # vùng bỏ qua (chữ OSD đồng hồ...), list [x1, y1, x2, y2] theo tỉ lệ 0..1 của khung hình
        warmup_sec=10.0,       # thời gian học ảnh gốc (chỉ tính các frame hợp lệ, liên tục)
        baseline_min_cells=3,  # lúc học: frame phải có >= số ô có chi tiết này mới được học (0 = tắt kiểm tra)
        hold_sec=3.0,          # bất thường (bất kể loại) giữ liên tục bao lâu mới báo
        clear_sec=5.0,         # bình thường lại bao lâu mới báo hết
        ref_alpha=0.02,        # tốc độ cập nhật ảnh gốc (EMA) khi trạng thái bình thường
        cell_tex_min=30.0,     # ô có Laplacian var >= ngưỡng này mới tính là có chi tiết (bỏ ô trơn: trời, tường)
        cell_drop_ratio=0.25,  # ô bị coi là mất chi tiết nếu var < ref * ratio
        obstruct_frac=0.6,     # tỉ lệ ô mất chi tiết >= ngưỡng -> obstructed
        dark_thr=15.0,         # mean < ngưỡng -> obstructed (đặt 0 để tắt nếu camera không có IR)
        bright_thr=240.0,      # mean > ngưỡng -> obstructed (bị rọi đèn)
        min_ref_edges=300,     # ảnh gốc ít cạnh hơn thì bỏ qua check scene_changed; lúc học: frame phải đủ số cạnh này
        edge_keep_thr=0.35,    # tỉ lệ cạnh ảnh gốc còn khớp < ngưỡng -> scene_changed
        exit_obstruct=0.35,    # đang bất thường: chỉ coi là hết khi obstruct < ngưỡng này
        exit_edge_keep=0.6,    # ... và edge_keep > ngưỡng này (vùng đệm, tránh nhảy qua lại quanh ngưỡng)
        frozen_sec=10.0,       # frame giống hệt nhau liên tục bao lâu -> frozen
        relearn_sec=600.0,     # scene_changed kéo dài bao lâu thì học lại cảnh mới (0 = tắt)
    )

    def __init__(self, cfg=None):
        c = dict(self.DEFAULTS)
        c.update(cfg or {})
        self.cfg = c
        self.kernel = np.ones((5, 5), np.uint8)
        self.reset()

    def reset(self):
        self.ready = False         # đã học xong ảnh gốc chưa
        self.learn_since = None    # thời điểm bắt đầu chuỗi frame hợp lệ đang học
        self.ref_gray = None
        self.ref_cell_var = None
        self.ref_edges = None
        self.n_warm = 0
        self.last_hash = None
        self.same_since = None
        self.cand_abn = False      # ứng viên hiện tại: bất thường hay bình thường
        self.cand_since = None
        self.reasons = Counter()   # đếm loại sự cố trong lúc chờ đủ hold_sec
        self.state = None          # None = bình thường, ngược lại là reason
        self.state_since = None
        self.metrics = {}
        self.cell_info = None      # (textured, dropped) theo lưới, chỉ dùng để vẽ debug
        self.mask = None           # True = pixel thuộc vùng osd_mask (tạo theo kích thước ảnh đã resize)

    def status(self):
        """Trạng thái để log / hiển thị: warmup | ok | <reason>."""
        if self.state:
            return self.state
        return "ok" if self.ready else "warmup"

    # ---------- feature ----------
    def _prep(self, frame):
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) if frame.ndim == 3 else frame
        w = self.cfg["width"]
        h = int(round(gray.shape[0] * w / gray.shape[1]))
        gray = cv2.resize(gray, (w, h), interpolation=cv2.INTER_AREA)
        return cv2.GaussianBlur(gray, (3, 3), 0)

    def _build_mask(self, shape):
        if self.mask is None or self.mask.shape != shape:
            h, w = shape
            m = np.zeros(shape, bool)
            for x1, y1, x2, y2 in self.cfg["osd_mask"]:
                m[int(y1 * h):int(np.ceil(y2 * h)), int(x1 * w):int(np.ceil(x2 * w))] = True
            self.mask = m if m.any() else np.zeros((0, 0), bool)  # rỗng = không có vùng bỏ qua
        return self.mask

    def _cell_var(self, gray):
        lap = cv2.Laplacian(gray, cv2.CV_32F)
        if self.mask.size:
            lap[self.mask] = 0
        g = self.cfg["grid"]
        ch, cw = gray.shape[0] // g, gray.shape[1] // g
        return lap[:ch * g, :cw * g].reshape(g, ch, g, cw).var(axis=(1, 3))

    def _edges(self, gray_u8):
        e = cv2.Canny(gray_u8, 50, 150) > 0
        if self.mask.size:
            e &= ~self.mask
        return e

    def _learn(self, gray, cell_var, alpha):
        g = gray.astype(np.float32)
        if self.ref_gray is None:
            self.ref_gray = g
            self.ref_cell_var = cell_var.astype(np.float32).copy()
        else:
            self.ref_gray += alpha * (g - self.ref_gray)
            self.ref_cell_var += alpha * (cell_var - self.ref_cell_var)
        self.ref_edges = self._edges(self.ref_gray.astype(np.uint8))

    # ---------- api ----------
    def update(self, frame, ts):
        c = self.cfg
        gray = self._prep(frame)
        self._build_mask(gray.shape)

        # hash bỏ vùng OSD: đồng hồ chạy giây không làm hỏng việc phát hiện frozen
        hg = gray
        if self.mask.size:
            hg = gray.copy()
            hg[self.mask] = 0
        h = hashlib.md5(hg.tobytes()).digest()
        if h == self.last_hash:
            if self.same_since is None:
                self.same_since = ts
        else:
            self.same_since = None
        self.last_hash = h

        cell_var = self._cell_var(gray)
        n_tex_abs = int((cell_var >= c["cell_tex_min"]).sum())

        if not self.ready:
            return self._warmup(gray, cell_var, n_tex_abs, ts)

        mean = float(gray.mean())
        textured = self.ref_cell_var >= c["cell_tex_min"]
        n_tex = int(textured.sum())
        dropped = (cell_var < self.ref_cell_var * c["cell_drop_ratio"]) & textured
        obstruct = float(dropped.sum()) / n_tex if n_tex else 0.0
        self.cell_info = (textured, dropped)

        n_ref = int(self.ref_edges.sum())
        if n_ref >= c["min_ref_edges"]:
            cur = cv2.dilate(self._edges(gray).astype(np.uint8), self.kernel) > 0
            edge_keep = float((self.ref_edges & cur).sum()) / n_ref
        else:
            edge_keep = 1.0

        self.metrics = dict(mean=round(mean, 1), obstruct=round(obstruct, 2),
                            edge_keep=round(edge_keep, 2), n_tex=n_tex, n_tex_abs=n_tex_abs)

        # Gán nhãn LUÔN theo ngưỡng vào. Vùng đệm (ngưỡng thoát lỏng hơn) chỉ GIỮ trạng thái bất thường,
        # không bỏ phiếu nhãn. Bản cũ gán nhãn bằng ngưỡng thoát: xoay camera (obstruct 0.37 >= 0.35) bị xếp
        # "obstructed" 179/180 phiếu và không bao giờ học lại (phát hiện 2026-10-15, khoá bằng test T5, T10).
        sticky = self.cand_abn or self.state is not None
        if self.same_since is not None and ts - self.same_since >= c["frozen_sec"]:
            raw = "frozen"
        elif mean < c["dark_thr"] or mean > c["bright_thr"] or obstruct >= c["obstruct_frac"]:
            raw = "obstructed"
        elif edge_keep < c["edge_keep_thr"]:
            raw = "scene_changed"
        elif sticky and (obstruct >= c["exit_obstruct"] or edge_keep < c["exit_edge_keep"]):
            raw = HOLD  # đang bất thường, chưa đủ sạch để thoát
        else:
            raw = None
        return self._debounce(raw, ts, gray, cell_var)

    def update_no_frame(self, ts):
        """Gọi khi mất stream / không mở được camera."""
        self.last_hash = None
        self.same_since = None
        self.metrics = {}
        return self._debounce("no_signal", ts, None, None)

    # ---------- internal ----------
    def _warmup(self, gray, cell_var, n_tex_abs, ts):
        """Học ảnh gốc, chỉ từ frame trông như cảnh thật (đủ chi tiết). Không hợp lệ -> chờ, báo baseline_invalid."""
        c = self.cfg
        n_edges = int(self._edges(gray).sum())
        self.metrics = dict(n_tex_abs=n_tex_abs, edges=n_edges)
        valid = c["baseline_min_cells"] <= 0 or (
            n_tex_abs >= c["baseline_min_cells"] and n_edges >= c["min_ref_edges"])

        if valid:
            if self.learn_since is None:
                self.learn_since, self.n_warm, self.ref_gray = ts, 0, None
            self._learn(gray, cell_var, 1.0 / (self.n_warm + 1))
            self.n_warm += 1
            if ts - self.learn_since >= c["warmup_sec"]:
                self.ready = True
        else:
            self.learn_since = None  # chuỗi học bị cắt -> học lại từ đầu khi cảnh hợp lệ trở lại

        return self._debounce(None if valid else "baseline_invalid", ts, None, None)

    def _debounce(self, raw, ts, gray, cell_var):
        """Đếm thời gian chung cho mọi loại bất thường; đủ hold_sec thì báo loại xuất hiện nhiều nhất."""
        c = self.cfg
        abnormal = raw is not None
        if abnormal != self.cand_abn:
            self.cand_abn, self.cand_since = abnormal, ts
            self.reasons.clear()
        if abnormal and raw != HOLD:
            self.reasons[raw] += 1

        event = None
        hold = c["hold_sec"] if abnormal else c["clear_sec"]
        if abnormal != (self.state is not None) and ts - self.cand_since >= hold:
            prev = self.state
            self.state = self.reasons.most_common(1)[0][0] if abnormal else None
            self.state_since = ts
            event = dict(event="tamper_start" if abnormal else "tamper_end",
                         reason=self.state or prev, prev=prev, ts=ts, metrics=dict(self.metrics))

        if gray is None:
            return event
        if self.state is None and raw is None:
            self._learn(gray, cell_var, c["ref_alpha"])
        elif (self.state == "scene_changed" and c["relearn_sec"] > 0
              and ts - self.state_since >= c["relearn_sec"]):
            # không ai chỉnh lại góc -> chấp nhận cảnh mới làm ảnh gốc
            self.ref_gray = None
            self._learn(gray, cell_var, 1.0)
            self.state, self.cand_abn, self.cand_since = None, False, ts
            self.reasons.clear()
            event = dict(event="relearn", reason="scene_changed", prev="scene_changed",
                         ts=ts, metrics=dict(self.metrics))
        return event
