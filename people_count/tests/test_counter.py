"""Test module đếm hiện diện (C1-C9). Metadata giả, thời gian truyền tay, 2 frame/giây."""
import pytest

from counter import CountModule
from meta import FrameMeta, Obj

FPS = 2.0
CLASSES = ("person", "dog", "cat")
FULL = [[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]]
LEFT = [[0.0, 0.0], [0.5, 0.0], [0.5, 1.0], [0.0, 1.0]]
RIGHT = [[0.5, 0.0], [1.0, 0.0], [1.0, 1.0], [0.5, 1.0]]


def zone(zone_id, polygon=FULL, report_sec=30, cam_id="cam_a"):
    return {"zone_id": zone_id, "cam_id": cam_id, "function": "count", "polygon": polygon,
            "ref_size": [1920, 1080], "params": {"report_sec": report_sec}, "version": 1}


def obj(cls, x=0.5, foot=0.5, track_id=None):
    """Đối tượng có chân (giữa cạnh đáy khung) tại (x, foot)."""
    return Obj(cls, 0.9, (x - 0.02, foot - 0.1, x + 0.02, foot), track_id)


def run(mod, objects_at, start, seconds, cam_id="cam_a"):
    """Cho module xem từ giây `start` trong `seconds` giây; objects_at(t) trả về danh sách đối tượng."""
    events = []
    for i in range(int(round(seconds * FPS))):
        t = start + i / FPS
        events += mod.update(FrameMeta(cam_id, t, 1920, 1080, tuple(objects_at(t))))
    return events


def counts(ev, cls):
    return ev["payload"]["counts"][cls]


def test_c1_vung_rong_van_bao_cao_so_0():
    """Tình huống thật: sân chơi vắng cả buổi sáng.
    Giả lập: 60 giây không có đối tượng nào, cửa sổ 30 giây.
    Đỏ thì hỏng gì: backend không nhận được gì, không phân biệt được "vắng" với "hệ treo / mất camera".
    """
    mod = CountModule("cam_a", [zone("z1")], CLASSES)
    ev = run(mod, lambda t: [], 0, 61)
    assert len(ev) == 2
    for e in ev:
        assert e["event_type"] == "report" and e["function"] == "count"
        assert e["cam_id"] == "cam_a" and e["zone_id"] == "z1"
        assert all(counts(e, c) == {"median": 0, "max": 0} for c in CLASSES)


def test_c2_lop_moi_xuat_hien_cuoi_cua_so_khong_lam_lech_trung_vi():
    """Tình huống thật: cả 30 giây chỉ có 1 người, 2 giây cuối một con chó chạy ngang qua
    (lỗi của ds_count.py, chứng minh 2026-10-19: trung vị chó ra 1 vì các frame trước khi chó xuất hiện không được tính là 0).
    Giả lập: person ở mọi frame, dog chỉ ở 2 giây cuối cửa sổ.
    Đỏ thì hỏng gì: lớp hiếm bị báo cao hơn thật, báo cáo "thường có 1 con chó" trong khi chó chỉ chạy qua.
    """
    mod = CountModule("cam_a", [zone("z1")], CLASSES)
    ev = run(mod, lambda t: [obj("person")] + ([obj("dog", x=0.3)] if t >= 28 else []), 0, 30.5)
    assert len(ev) == 1
    assert counts(ev[0], "person") == {"median": 1, "max": 1}
    assert counts(ev[0], "dog") == {"median": 0, "max": 1}
    assert counts(ev[0], "cat") == {"median": 0, "max": 0}


def test_c3_chi_dem_doi_tuong_co_chan_trong_vung():
    """Tình huống thật: hai vùng trái / phải trên cùng camera, người đứng mỗi bên.
    Giả lập: 2 người chân ở bên trái, 1 người chân ở bên phải.
    Đỏ thì hỏng gì: số của vùng này lẫn sang vùng kia.
    """
    mod = CountModule("cam_a", [zone("trai", LEFT), zone("phai", RIGHT)], CLASSES)
    people = [obj("person", x=0.1), obj("person", x=0.3), obj("person", x=0.8)]
    ev = {e["zone_id"]: e for e in run(mod, lambda t: people, 0, 30.5)}
    assert counts(ev["trai"], "person")["median"] == 2
    assert counts(ev["phai"], "person")["median"] == 1


def test_c4_vung_chong_nhau_dem_cho_ca_hai():
    """Tình huống thật: vùng "cả sân" và vùng "khu cát" nằm trong nó, người đứng trong khu cát.
    Giả lập: hai vùng chồng nhau, một người ở phần chung.
    Đỏ thì hỏng gì: người ở phần chung chỉ được tính cho một vùng, vùng kia báo thiếu.
    """
    mod = CountModule("cam_a", [zone("ca_san", FULL), zone("khu_cat", LEFT)], CLASSES)
    ev = {e["zone_id"]: e for e in run(mod, lambda t: [obj("person", x=0.2)], 0, 30.5)}
    assert counts(ev["ca_san"], "person")["median"] == 1
    assert counts(ev["khu_cat"], "person")["median"] == 1


def test_c5_lop_khong_dem_thi_bo_qua():
    """Tình huống thật: model trả về lớp không thuộc bài đếm (xe đạp, ghế...) nếu bộ lọc lớp của model bị đổi.
    Giả lập: frame có bicycle và chair cùng 1 người.
    Đỏ thì hỏng gì: chương trình chết vì lớp lạ, hoặc báo cáo có lớp mà backend không biết.
    """
    mod = CountModule("cam_a", [zone("z1")], CLASSES)
    ev = run(mod, lambda t: [obj("person"), obj("bicycle"), obj("chair")], 0, 30.5)
    assert set(ev[0]["payload"]["counts"]) == set(CLASSES)
    assert counts(ev[0], "person")["median"] == 1


def test_c6_mat_hinh_lau_thi_cua_so_cu_dong_tai_frame_cuoi():
    """Tình huống thật: camera mất kết nối 10 phút giữa cửa sổ rồi có hình lại.
    Giả lập: 10 giây có 3 người, mất 600 giây, rồi có hình lại.
    Đỏ thì hỏng gì: báo cáo ghi một cửa sổ dài 10 phút như thể đã đếm suốt 10 phút đó.
    """
    mod = CountModule("cam_a", [zone("z1")], CLASSES)
    three = [obj("person", x=x) for x in (0.2, 0.5, 0.8)]
    assert run(mod, lambda t: three, 0, 10) == []
    ev = run(mod, lambda t: [], 610, 1)
    assert len(ev) == 1
    p = ev[0]["payload"]
    assert p["window_start"] == 0 and p["window_end"] == 9.5 and p["frames"] == 20
    assert counts(ev[0], "person")["median"] == 3


def test_c7_dung_nhip_report_sec_cua_tung_vung():
    """Tình huống thật: backend cần vùng A báo mỗi 30 giây, vùng B mỗi 60 giây (ngưỡng nghiệp vụ theo vùng).
    Giả lập: hai vùng khác report_sec, chạy 121 giây.
    Đỏ thì hỏng gì: mọi vùng dùng chung một nhịp, không làm được yêu cầu theo khu vực.
    """
    mod = CountModule("cam_a", [zone("a", report_sec=30), zone("b", report_sec=60)], CLASSES)
    ev = run(mod, lambda t: [obj("person")], 0, 121)
    assert [e["ts"] for e in ev if e["zone_id"] == "a"] == [30, 60.5, 91]
    assert [e["ts"] for e in ev if e["zone_id"] == "b"] == [60, 120.5]


def test_c8_dung_chuong_trinh_thi_bao_cao_phan_con_lai():
    """Tình huống thật: hết video, hoặc dừng chương trình giữa cửa sổ.
    Giả lập: chạy 40 giây với cửa sổ 30 giây, rồi gọi flush.
    Đỏ thì hỏng gì: 10 giây cuối mất số liệu; với video ngắn hơn report_sec thì không có số nào.
    """
    mod = CountModule("cam_a", [zone("z1")], CLASSES)
    assert len(run(mod, lambda t: [obj("person")], 0, 40)) == 1
    tail = mod.flush()
    assert len(tail) == 1 and tail[0]["payload"]["window_end"] == 39.5
    assert mod.flush() == []  # gọi lại không báo trùng


def test_c9_frame_cua_camera_khac_la_loi_code():
    """Tình huống thật: nhiều camera trong một pipeline; code chia frame theo camera bị nhầm chỉ số.
    Giả lập: đưa frame của cam_b vào module của cam_a.
    Đỏ thì hỏng gì: số của camera này âm thầm cộng vào camera khác, không ai phát hiện.
    """
    mod = CountModule("cam_a", [zone("z1")], CLASSES)
    with pytest.raises(ValueError, match="cam_b"):
        mod.update(FrameMeta("cam_b", 0.0, 1920, 1080, ()))


def test_c10_trung_vi_luon_la_so_thuc():
    """Tình huống thật: cửa sổ có số frame chẵn thì trung vị là trung bình 2 số giữa, lẻ thì là 1 số
    (thấy 2026-10-19 trên robo: cùng một trường lúc ra 7, lúc ra 7.0).
    Giả lập: một cửa sổ 61 frame (lẻ) và một cửa sổ 60 frame (chẵn).
    Đỏ thì hỏng gì: backend nhận một trường lúc kiểu số nguyên lúc kiểu số thực, code bên đó parse sai hoặc so sánh sai.
    """
    mod = CountModule("cam_a", [zone("z1")], CLASSES)
    odd = run(mod, lambda t: [obj("person")], 0, 30.5)              # frame 0..30: 61 frame
    even = run(mod, lambda t: [obj("person")], 30.5, 30) + mod.flush()  # frame 30.5..60: 60 frame
    for ev in odd + even:
        for c in CLASSES:
            assert type(counts(ev, c)["median"]) is float, ev
            assert type(counts(ev, c)["max"]) is int, ev
    assert {ev["payload"]["frames"] % 2 for ev in odd + even} == {0, 1}  # đã thử cả chẵn lẫn lẻ