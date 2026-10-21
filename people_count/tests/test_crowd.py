"""Test module đám đông (D1-D9). Metadata giả, thời gian truyền tay, 2 frame/giây."""
import copy
import json
import os

import pytest
import yaml

import run_engine
from config import validate_all
from crowd import CrowdModule
from meta import FrameMeta, Obj, Text

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FPS = 2.0
FULL = [[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]]
LEFT = [[0.0, 0.0], [0.5, 0.0], [0.5, 1.0], [0.0, 1.0]]


def zone(zid="d1", polygon=FULL, min_people=5, hold=10, clear=10, cam_id="cam_a"):
    return {"zone_id": zid, "cam_id": cam_id, "function": "crowd", "polygon": polygon, "ref_size": [1920, 1080],
            "params": {"min_people": min_people, "hold_sec": hold, "clear_sec": clear}, "version": 1}


def people(n, x=0.25, cls="person"):
    """n đối tượng, chân tại (x, 0.5)."""
    return [Obj(cls, 0.9, (x - 0.02, 0.4, x + 0.02, 0.5), None) for _ in range(n)]


def run(mod, n_at, start, seconds):
    """Cho module xem từ giây `start` trong `seconds` giây; n_at(t) trả về danh sách đối tượng."""
    events = []
    for i in range(int(round(seconds * FPS))):
        t = start + i / FPS
        events += mod.update(FrameMeta("cam_a", t, 1920, 1080, tuple(n_at(t))))
    return events


def test_d1_dong_giu_du_lau_moi_bao():
    """Tình huống thật: một nhóm 6 người tụ tập ở quảng trường; nhóm 6 người đi ngang qua vài giây.
    Giả lập: ngưỡng 5 người, hold 10s. Lần 1: 6 người trong 4 giây rồi đi (dưới nửa cửa sổ); lần 2: 6 người ở lại 30 giây.
    Đỏ thì hỏng gì: người đi ngang cũng bị báo đám đông (báo nhầm), hoặc tụ tập thật không được báo.
    """
    mod = CrowdModule("cam_a", [zone()])
    assert run(mod, lambda t: people(6), 0, 4) + run(mod, lambda t: [], 4, 16) == []
    ev = run(mod, lambda t: people(6), 20, 30)
    assert [e["event_type"] for e in ev] == ["alert_start"]
    e = ev[0]
    assert e["function"] == "crowd" and e["cam_id"] == "cam_a" and e["zone_id"] == "d1"
    assert e["ts"] == pytest.approx(25.0)  # trung vị 10s gần nhất >= 5 lần đầu khi 6 người chiếm quá nửa cửa sổ (11/21 mẫu)
    assert e["payload"]["min_people"] == 5 and e["payload"]["window_sec"] == 10 and e["payload"]["max"] == 6


def test_d2_model_sot_nguoi_vai_frame_van_bao():
    """Tình huống thật: 7 người đứng yên, model sót 3-4 người ở 1/3 số frame (thấy số nhảy 8-10-7 trên video demo).
    Giả lập: số người theo frame lặp 7, 7, 3 với ngưỡng 5: cứ 3 frame có 1 frame dưới ngưỡng.
    Đỏ thì hỏng gì: xét từng frame thì mỗi frame sót người lại đếm từ đầu -> đám đông thật không bao giờ được báo.
    """
    mod = CrowdModule("cam_a", [zone()])
    ev = run(mod, lambda t: people((7, 7, 3)[int(t * FPS) % 3]), 0, 30)
    assert [e["event_type"] for e in ev] == ["alert_start"]
    assert ev[0]["payload"]["median"] == 7.0


def test_d3_het_dong_moi_bao_het_va_chi_mot_lan():
    """Tình huống thật: đám đông giải tán; giữa lúc đông có vài giây bị cây che khuất bớt người.
    Giả lập: đông 30s, tụt về 1 người 3s (ngắn hơn clear 10s), đông lại 20s, rồi vắng hẳn 30s.
    Đỏ thì hỏng gì: che khuất vài giây làm báo "hết" rồi lại "bắt đầu" (người quản lý nhận chuỗi cảnh báo rác),
    hoặc giải tán rồi mà cảnh báo không bao giờ đóng.
    """
    mod = CrowdModule("cam_a", [zone()])
    ev = run(mod, lambda t: people(8), 0, 30)
    ev += run(mod, lambda t: people(1), 30, 3)
    ev += run(mod, lambda t: people(8), 33, 20)
    ev += run(mod, lambda t: [], 53, 30)
    assert [e["event_type"] for e in ev] == ["alert_start", "alert_end"]
    assert ev[1]["payload"]["median"] < 5 and ev[1]["payload"]["window_sec"] == 10


def test_d4_chi_dem_nguoi_co_chan_trong_vung():
    """Tình huống thật: vùng đám đông là nửa trái quảng trường; bên phải đông người, có cả đàn chó chạy trong vùng.
    Giả lập: 10 người chân ở bên phải, 10 con chó và 2 người chân ở bên trái, ngưỡng 5.
    Đỏ thì hỏng gì: đám đông ngoài vùng hoặc chó bị tính, báo nhầm ở khu không cần theo dõi.
    """
    mod = CrowdModule("cam_a", [zone(polygon=LEFT)])
    ev = run(mod, lambda t: people(10, x=0.75) + people(10, x=0.25, cls="dog") + people(2, x=0.25), 0, 30)
    assert ev == []


def test_d5_mat_hinh_lau_khong_quyet_dinh_bang_du_lieu_cu():
    """Tình huống thật: camera mất kết nối 10 phút; trước đó có 4 giây đông người, sau khi có hình lại có 4 giây đông.
    Giả lập: 4s 8 người, mất 600s, rồi 4s 8 người (mỗi đoạn ngắn hơn hold 10s).
    Đỏ thì hỏng gì: ghép mẫu trước và sau 10 phút mất hình thành một cửa sổ -> báo đám đông dựa trên chuyện cũ.
    """
    mod = CrowdModule("cam_a", [zone()])
    assert run(mod, lambda t: people(8), 0, 4) + run(mod, lambda t: people(8), 604, 4) == []


def test_d6_nhieu_vung_doc_lap_nguong_rieng():
    """Tình huống thật: cổng chính ngưỡng 20 người, khu vui chơi ngưỡng 5 người trên cùng camera.
    Giả lập: 2 vùng phủ cùng chỗ, ngưỡng 20 và 5, có 8 người.
    Đỏ thì hỏng gì: mọi vùng dùng chung một ngưỡng, không làm được yêu cầu theo khu vực.
    """
    mod = CrowdModule("cam_a", [zone("cong", min_people=20), zone("vui_choi", min_people=5)])
    ev = run(mod, lambda t: people(8), 0, 30)
    assert [(e["zone_id"], e["event_type"]) for e in ev] == [("vui_choi", "alert_start")]


def test_d7_ve_so_nguoi_va_canh_bao_tren_video():
    """Tình huống thật: demo, người xem thấy số người / ngưỡng, và thấy rõ lúc hệ đang báo đám đông.
    Giả lập: trước và sau khi báo động, gọi draw().
    Đỏ thì hỏng gì: video không cho thấy lúc nào hệ báo, khách không đối chiếu được với hình.
    """
    mod = CrowdModule("cam_a", [zone()])
    run(mod, lambda t: people(6), 0, 2)
    before = [i for i in mod.draw() if isinstance(i, Text)]
    assert [(t.text, t.kind, t.row) for t in before] == [("d1 | dam dong: 6/5 nguoi", "info", 1)]
    run(mod, lambda t: people(6), 2, 20)
    after = mod.draw()
    assert [i.kind for i in after if not isinstance(i, Text)] == ["alert"] * 4  # viền vùng đỏ
    assert [(t.text, t.kind) for t in after if isinstance(t, Text)] == [("d1 | dam dong: 6/5 nguoi -> CANH BAO", "alert")]


def test_d8_engine_chay_ca_dem_va_dam_dong_tren_cung_camera():
    """Tình huống thật: một camera quảng trường bật cả đếm hiện diện và đám đông (thêm chức năng = thêm module + config).
    Giả lập: Engine với camera functions [count, crowd], 1 vùng mỗi chức năng, 8 người trong 31 giây.
    Đỏ thì hỏng gì: bật thêm chức năng thì mất chức năng cũ, hoặc event / hình vẽ của hai chức năng lẫn nhau.
    """
    prof = yaml.safe_load(open(os.path.join(ROOT, "profiles/overview.yaml")))
    cam = {"cam_id": "cam_a", "name": "a", "url": __file__, "profile": "overview", "functions": ["count", "crowd"]}
    count_zone = {**zone("c1"), "function": "count", "params": {"report_sec": 30}}
    rows = []

    class Out:
        def write(self, s):
            rows.extend(json.loads(x) for x in s.splitlines())

        def flush(self):
            pass

    eng = run_engine.Engine([cam], [count_zone, zone("d1")], prof, clock=lambda: 0.0, draw=True, events_out=Out())
    shapes = {}
    for i in range(62):
        shapes = eng.on_frames([FrameMeta("cam_a", i / 2, 1920, 1080, tuple(people(8)))])
    assert sorted((r["function"], r["event_type"]) for r in rows) == [("count", "report"), ("crowd", "alert_start")]
    assert [t.text.split(" |")[0] for t in shapes["cam_a"] if isinstance(t, Text)] == ["c1", "d1"]


def test_d9_config_dam_dong_sai_bi_bat():
    """Tình huống thật: backend gửi vùng đám đông thiếu ngưỡng, hoặc gõ số người dạng chữ; devops bật crowd mà chưa có vùng.
    Giả lập: bộ config hợp lệ có crowd, làm hỏng từng chỗ.
    Đỏ thì hỏng gì: lỗi lọt qua, module chết giữa lúc vận hành, hoặc bật crowd mà âm thầm không theo dõi gì.
    """
    prof = yaml.safe_load(open(os.path.join(ROOT, "profiles/overview.yaml")))
    cams = {"cameras": [{"cam_id": "cam_a", "name": "a", "url_env": "X", "profile": "overview",
                         "functions": ["count", "crowd"]}]}
    zones = {"zones": [{**zone("c1"), "function": "count", "params": {"report_sec": 30}}, zone("d1")]}

    def errs(z):
        return validate_all(cams, z, prof, ROOT, os.path.join(ROOT, "profiles"), check_env=False)

    assert errs(zones) == []
    bad = copy.deepcopy(zones)
    del bad["zones"][1]["params"]["hold_sec"]
    bad["zones"][1]["params"]["min_people"] = "6"
    e = errs(bad)
    assert any("zones[1].params.hold_sec: thiếu" in x for x in e) and any("zones[1].params.min_people = '6'" in x for x in e), e
    e = errs({"zones": zones["zones"][:1]})
    assert any("cam_a: bật crowd nhưng chưa có vùng crowd nào" in x for x in e), e