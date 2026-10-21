"""Test ranh giới DeepStream (R1) và hợp đồng event (R2)."""
import json
import os
import subprocess
import sys

from events import SCHEMA_VERSION, envelope

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def test_r1_logic_khong_dinh_deepstream():
    """Tình huống thật: phải đổi phiên bản DeepStream (vd máy đích không hỗ trợ 7.0), hoặc test trên máy không có GPU.
    Giả lập: một tiến trình Python sạch import toàn bộ phần logic, rồi xem gi / pyds có bị nạp không.
    Đỏ thì hỏng gì: logic dính DeepStream -> đổi phiên bản phải sửa cả logic; test logic đòi GPU (KIEN_TRUC mục 3.4).
    """
    code = ("import sys; sys.path[:0] = [%r, %r]; "
            "import meta, zones, counter, crowd, config, events; "
            "bad = [m for m in ('gi', 'pyds') if m in sys.modules]; print(','.join(bad))") % (ROOT, os.path.dirname(ROOT))
    out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, check=True).stdout.strip()
    assert out == "", f"phần logic đã nạp: {out}"


def test_r2_event_dung_hop_dong():
    """Tình huống thật: backend đọc event của mọi chức năng theo một khuôn chung (KIEN_TRUC mục 8).
    Giả lập: bọc một báo cáo đếm thành event, chuyển JSON.
    Đỏ thì hỏng gì: thiếu / đổi tên trường, backend đọc sai hoặc bỏ qua event.
    """
    ev = {"event_type": "report", "function": "count", "cam_id": "cam_a", "zone_id": "z1", "ts": 30.0,
          "payload": {"frames": 61}}
    out = json.loads(json.dumps(envelope(ev, "id-1")))
    assert out == {"schema_version": SCHEMA_VERSION, "event_id": "id-1", "event_type": "report", "function": "count",
                   "cam_id": "cam_a", "zone_id": "z1", "ts": 30.0, "payload": {"frames": 61},
                   "evidence": {"snapshot": None}}