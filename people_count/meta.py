"""Hợp đồng perception -> logic (KIEN_TRUC mục 4, bản 0.1). Không phụ thuộc DeepStream.

pipeline.py đổi metadata của DeepStream sang các kiểu dưới đây; module logic chỉ đọc các kiểu này.
"""
from typing import NamedTuple, Optional, Tuple


class Obj(NamedTuple):
    cls: str                                  # "person", "dog", ...
    conf: float
    bbox: Tuple[float, float, float, float]   # x1, y1, x2, y2 theo TỈ LỆ 0..1 của khung hình
    track_id: Optional[int]                   # None khi profile tắt tracker; chỉ unique trong 1 camera


class FrameMeta(NamedTuple):
    cam_id: str
    ts: float              # giây. File: theo PTS của video; camera: giờ máy lúc nhận frame
    src_w: int             # kích thước khung gốc của camera (để kiểm tỉ lệ khung với ref_size của vùng)
    src_h: int
    objects: Tuple[Obj, ...]