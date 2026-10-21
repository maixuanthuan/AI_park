"""Test overlay video debug / demo (O1-O5): module nói "vẽ gì", bật / tắt không đổi lõi. Không cần DeepStream."""
import io
import json
import os

import yaml

import run_engine
from counter import CountModule
from meta import FrameMeta, Line, Obj, Text

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CLASSES = ("person", "dog", "cat")
TRI = [[0.1, 0.2], [0.9, 0.2], [0.5, 0.9]]


def zone(zid="z1", cam_id="cam_a", polygon=TRI, report_sec=30):
    return {"zone_id": zid, "cam_id": cam_id, "function": "count", "polygon": polygon,
            "ref_size": [1920, 1080], "params": {"report_sec": report_sec}, "version": 1}


def frame(cid, ts, n_person=1, w=1920, h=1080):
    """n_person người, chân tại (0.5, 0.5): trong tam giác TRI."""
    return FrameMeta(cid, ts, w, h, tuple(Obj("person", 0.9, (0.48, 0.4, 0.52, 0.5), None) for _ in range(n_person)))


def cam(cid):
    return {"cam_id": cid, "name": cid, "url": __file__, "profile": "overview", "functions": ["count"]}


def engine(draw, events_out=None, zones=None):
    prof = yaml.safe_load(open(os.path.join(ROOT, "profiles/overview.yaml")))
    return run_engine.Engine([cam("cam_a")], zones or [zone()], prof, clock=lambda: 0.0, draw=draw,
                             events_out=events_out)


def texts(items):
    return [i.text for i in items if isinstance(i, Text)]


def test_o1_tat_ve_thi_loi_y_nguyen(caplog):
    """Tình huống thật: chạy vận hành không có --debug-out (mặc định), và chạy demo có --debug-out trên cùng camera.
    Giả lập: cùng 61 frame đưa vào Engine draw=False và draw=True.
    Đỏ thì hỏng gì: phần phụ (vẽ) làm đổi số báo cáo gửi backend, hoặc tắt vẽ mà vẫn tốn công dựng hình mỗi frame.
    """
    caplog.set_level("INFO", logger="engine")
    reports = {}
    for draw in (False, True):
        caplog.clear()
        eng = engine(draw)
        outs = [eng.on_frames([frame("cam_a", i / 2, n_person=1 + i % 3)]) for i in range(61)]
        if not draw:
            assert all(o == {} for o in outs)
        reports[draw] = [json.loads(r.getMessage())["payload"] for r in caplog.records
                         if '"event_type": "report"' in r.getMessage()]
    assert len(reports[False]) == 1 and reports[False] == reports[True]


def test_o2_ve_vien_vung_va_so_hien_tai():
    """Tình huống thật: khách xem video demo, cần thấy vùng đếm nằm đâu và hệ đang thấy bao nhiêu người.
    Giả lập: vùng tam giác, frame có 2 người trong vùng.
    Đỏ thì hỏng gì: video thiếu viền vùng / số đếm, khách không hiểu hệ đang làm gì.
    """
    mod = CountModule("cam_a", [zone()], CLASSES)
    mod.update(frame("cam_a", 0.0, n_person=2))
    items = mod.draw()
    lines = [i for i in items if isinstance(i, Line)]
    assert len(lines) == 3 and {(ln.x1, ln.y1) for ln in lines} == {(0.1, 0.2), (0.9, 0.2), (0.5, 0.9)}
    assert all(ln.kind == "zone" for ln in lines)
    assert texts(items) == ["z1 | hien tai: person 2, dog 0, cat 0"]
    assert all(i.x == 0.1 and i.y == 0.2 for i in items if isinstance(i, Text))  # góc trên trái khung bao vùng


def test_o3_so_tren_video_la_frame_dang_hien_khong_phai_trung_vi():
    """Tình huống thật: demo, người xem đếm khung trên hình rồi so với số in trên video (Thuận chốt 2026-10-21).
    Giả lập: 30.5 giây, số người đổi 1-2-3 mỗi frame, qua 1 lần báo cáo.
    Đỏ thì hỏng gì: video in số khác số khung đang thấy (vd trung vị cả cửa sổ) -> người xem tưởng hệ đếm sai.
    """
    mod = CountModule("cam_a", [zone()], CLASSES)
    events = []
    for i in range(62):
        events += mod.update(frame("cam_a", i / 2, n_person=1 + i % 3))
    assert len(events) == 1
    assert texts(mod.draw()) == ["z1 | hien tai: person 2, dog 0, cat 0"]  # frame cuối i=61 -> 1 + 61 % 3 = 2


def test_o4_file_event_giong_log_tung_dong():
    """Tình huống thật: làm bảng event cho demo từ file .jsonl cạnh video, đối chiếu với log vận hành.
    Giả lập: Engine ghi vào file giả (StringIO), chạy qua 1 cửa sổ rồi finish().
    Đỏ thì hỏng gì: bảng event demo thiếu / khác event thật, hoặc dòng không đọc được bằng JSON.
    """
    buf = io.StringIO()
    eng = engine(True, events_out=buf)
    for i in range(70):
        eng.on_frames([frame("cam_a", i / 2)])
    eng.finish()
    rows = [json.loads(line) for line in buf.getvalue().splitlines()]
    assert len(rows) == 2 and all(r["event_type"] == "report" and r["cam_id"] == "cam_a" for r in rows)
    assert len({r["event_id"] for r in rows}) == 2


def test_o5_camera_sai_ti_le_khung_hien_canh_bao_tren_video():
    """Tình huống thật: link điền nhầm luồng phụ 704x576, vùng vẽ trên 16:9 -> camera không được đếm.
    Giả lập: Engine draw=True, frame 704x576.
    Đỏ thì hỏng gì: video demo vẫn có hình nhưng không có số, người xem tưởng hệ đếm ra 0.
    """
    eng = engine(True)
    out = eng.on_frames([frame("cam_a", 0.0, w=704, h=576)])
    assert [(i.kind, "KHONG dem" in i.text) for i in out["cam_a"]] == [("alert", True)]