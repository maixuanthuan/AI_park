"""Hợp đồng event với backend (KIEN_TRUC mục 8, bản 0.1, chưa thống nhất với backend).

Module logic trả về phần nội dung; runner bọc thành envelope chung. event_id truyền vào để test được.
"""
SCHEMA_VERSION = "0.1"


def envelope(ev, event_id, snapshot=None):
    return {"schema_version": SCHEMA_VERSION, "event_id": event_id, "event_type": ev["event_type"],
            "function": ev["function"], "cam_id": ev["cam_id"], "zone_id": ev.get("zone_id"), "ts": ev["ts"],
            "payload": ev["payload"], "evidence": {"snapshot": snapshot}}