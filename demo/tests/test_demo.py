"""Test công cụ demo (M1-M5): đọc đối số nguồn=chức năng, đổi kết quả model, module băng rôn. Thời gian truyền tay."""
from bang_ron import BannerModule
from meta import FrameMeta, Obj
from run_demo import parse_sources, to_objs

PERSON = Obj("person", 0.9, (0.40, 0.40, 0.60, 1.00), None)             # người đứng giữa khung
HELD = Obj("banner", 0.6, (0.35, 0.25, 0.65, 0.45), None)               # băng rôn giơ ngang ngực / đầu người
HANGING = Obj("banner", 0.9, (0.30, 0.00, 0.70, 0.08), None)            # băng rôn treo cao, xa trên đầu người


def run(mod, objs_at, start, seconds, fps=2.0):
    ev = []
    for i in range(int(round(seconds * fps))):
        t = start + i / fps
        ev += mod.update(FrameMeta("a", t, 1920, 1080, tuple(objs_at(t))))
    return ev


def test_m1_moi_nguon_gan_chuc_nang_rieng(tmp_path):
    """Tình huống thật: khách đưa video khác nhau cho từng chức năng, kèm một camera thật (link nằm trong .env).
    Giả lập: 1 file video gán count,crowd; 1 biến camera gán bang_ron.
    Đỏ thì hỏng gì: chức năng chạy nhầm nguồn, hoặc phải gõ link có mật khẩu ra dòng lệnh.
    """
    f = tmp_path / "quang_truong.mp4"
    f.write_bytes(b"x")
    out, err = parse_sources([f"{f}=count,crowd", "CAM_01_URL=bang_ron"], {"CAM_01_URL": "rtsp://u:p@1.2.3.4/x"})
    assert err == []
    assert out == [("quang_truong", str(f), False, ["count", "crowd"]), ("CAM_01_URL", "rtsp://u:p@1.2.3.4/x", True, ["bang_ron"])]


def test_m2_doi_so_sai_bao_du_mot_luot(tmp_path):
    """Tình huống thật: gõ nhầm tên chức năng, quên dấu =, gõ sai tên biến camera trước buổi demo.
    Giả lập: 3 đối số sai 3 kiểu.
    Đỏ thì hỏng gì: chạy mất vài phút rồi mới chết, hoặc sửa một lỗi mới thấy lỗi tiếp theo.
    """
    f = tmp_path / "a.mp4"
    f.write_bytes(b"x")
    _, err = parse_sources([f"{f}=cuont", str(f), "CAM_X=count"], {})
    assert len(err) == 3, err
    assert "'cuont'" in err[0] and "thiếu '=chức năng'" in err[1] and "'CAM_X': không phải file video" in err[2]


def test_m3_ket_qua_model_doi_sang_ti_le_va_bo_lop_khac():
    """Tình huống thật: model trả khung pixel, có lớp không dùng (xe buýt), khung lấn ra ngoài mép ảnh.
    Giả lập: 3 khung trên ảnh 1000x500: person, bus (id 5), dog lấn mép phải.
    Đỏ thì hỏng gì: logic nhận pixel thay vì tỉ lệ (vùng sai chỗ), hoặc đếm cả lớp không thuộc bài.
    """
    objs = to_objs([[100, 50, 300, 500], [0, 0, 10, 10], [900, 250, 1100, 500]], [0, 5, 16], [0.9, 0.8, 0.7],
                   {0: "person", 16: "dog"}, 1000, 500)
    assert [(o.cls, o.bbox) for o in objs] == [("person", (0.1, 0.1, 0.3, 1.0)), ("dog", (0.9, 0.5, 1.0, 1.0))]


def test_m4_bang_ron_cam_tay_bao_bang_ron_treo_khong_bao():
    """Tình huống thật: đoàn người giơ băng rôn; và đường phố Việt Nam có băng rôn khẩu hiệu treo cao suốt ngày.
    Giả lập: 20s người + băng rôn ngang ngực; 20s người + băng rôn treo sát mép trên khung, xa trên đầu.
    Đỏ thì hỏng gì: băng rôn treo bị báo biểu tình cả ngày, hoặc người cầm băng rôn không được báo.
    """
    held = BannerModule("a", 0.3, 10, 10)
    ev = run(held, lambda t: [PERSON, HELD], 0, 20)
    assert [e["event_type"] for e in ev] == ["alert_start"] and ev[0]["function"] == "bang_ron"
    hanging = BannerModule("a", 0.3, 10, 10)
    assert run(hanging, lambda t: [PERSON, HANGING], 0, 20) == []
    weak = BannerModule("a", 0.7, 10, 10)  # độ tin 0.6 < ngưỡng 0.7: không tính
    assert run(weak, lambda t: [PERSON, HELD], 0, 20) == []


def test_m5_nhan_chap_chon_van_bao_va_het_dung_luc():
    """Tình huống thật: model nhận băng rôn lúc có lúc mất (thử clip biểu tình 2026-10-21: 38% thời gian, liên tục
    dài nhất chỉ 4s); rồi đoàn người đi khỏi.
    Giả lập: 30s băng rôn thấy ở 2/3 số frame, rồi 30s không có băng rôn.
    Đỏ thì hỏng gì: luật "liên tục nghiêm ngặt" không bao giờ báo, hoặc báo rồi không bao giờ hết.
    """
    mod = BannerModule("a", 0.3, 10, 10)
    ev = run(mod, lambda t: [PERSON] + ([HELD] if int(t * 2) % 3 else []), 0, 30)
    ev += run(mod, lambda t: [PERSON], 30, 30)
    assert [e["event_type"] for e in ev] == ["alert_start", "alert_end"]
    assert ev[1]["ts"] - ev[0]["ts"] >= 10

# ---------- lan_chiem ----------
from lan_chiem import LanChiemModule  # noqa: E402

MOTO = ("motorcycle", "chair", "vendor_cart")


def xe(x, cls="motorcycle"):
    return Obj(cls, 0.8, (x, 0.6, x + 0.08, 0.75), None)


def lc_run(mod, objs_at, start, seconds, fps=1.0):
    ev = []
    for i in range(int(round(seconds * fps))):
        t = start + i / fps
        ev += mod.update(FrameMeta("a", t, 1920, 1080, tuple(objs_at(t))))
    return ev


def test_m6_xe_do_lau_thi_bao_mot_lan_kem_danh_sach_vat():
    """Tình huống thật: một xe máy và một bộ bàn ghế quán cóc đặt trên vỉa hè hơn 1 phút.
    Giả lập: dwell 60s, lấy mẫu mỗi 5s; xe máy + ghế đứng yên 150 giây.
    Đỏ thì hỏng gì: lấn chiếm thật không được báo, hoặc mỗi vật / mỗi mẫu một cảnh báo (người quản lý bị spam).
    """
    mod = LanChiemModule("a", MOTO, dwell_sec=60)
    ev = lc_run(mod, lambda t: [xe(0.2), xe(0.5, "chair")], 0, 150)
    assert [e["event_type"] for e in ev] == ["alert_start"]
    assert 60 <= ev[0]["ts"] <= 65 and ev[0]["function"] == "lan_chiem"
    assert sorted(v["cls"] for v in ev[0]["payload"]["vat"]) == ["chair", "motorcycle"]


def test_m7_xe_chay_ngang_qua_khong_bao():
    """Tình huống thật: dòng xe máy chạy qua đoạn vỉa hè / lòng đường trong khung hình liên tục.
    Giả lập: mỗi giây có 3 xe máy ở vị trí khác nhau (xe chạy), kéo dài 5 phút.
    Đỏ thì hỏng gì: xe chạy qua bị báo lấn chiếm cả ngày.
    """
    mod = LanChiemModule("a", MOTO, dwell_sec=60)
    assert lc_run(mod, lambda t: [xe((t * 0.07 + k * 0.3) % 0.9) for k in range(3)], 0, 300) == []
    assert lc_run(mod, lambda t: [], 300, 70) == []
    assert mod.cands == []  # vật không thấy lại quá dwell bị xoá: bộ nhớ không phình theo số xe chạy qua


def test_m8_bi_che_khuat_mot_phan_thoi_gian_van_bao():
    """Tình huống thật: xe đẩy hàng rong đứng yên, người đi qua che khuất / model sót 1/3 số lần.
    Giả lập: xe đẩy có mặt ở 2/3 số mẫu trong 150 giây.
    Đỏ thì hỏng gì: luật đòi thấy liên tục -> lấn chiếm thật ở chỗ đông người không bao giờ được báo.
    """
    mod = LanChiemModule("a", MOTO, dwell_sec=60)
    ev = lc_run(mod, lambda t: [xe(0.4, "vendor_cart")] if int(t / 5) % 3 else [], 0, 150)
    assert [e["event_type"] for e in ev] == ["alert_start"]


def test_m9_don_di_thi_het_va_khong_bao_lai_khi_doi_danh_sach():
    """Tình huống thật: quán cóc kê thêm ghế giữa chừng, rồi dọn hết đi.
    Giả lập: xe máy 120s; từ giây 90 thêm ghế; từ giây 120 không còn gì trong 120s.
    Đỏ thì hỏng gì: thêm 1 ghế lại bắn cảnh báo mới; dọn đi rồi mà cảnh báo không bao giờ đóng.
    """
    mod = LanChiemModule("a", MOTO, dwell_sec=60)
    ev = lc_run(mod, lambda t: [xe(0.2)] + ([xe(0.6, "chair")] if t >= 90 else []), 0, 120)
    ev += lc_run(mod, lambda t: [], 120, 120)
    assert [e["event_type"] for e in ev] == ["alert_start", "alert_end"]
    assert 175 <= ev[1]["ts"] <= 190  # vật dọn đi: hết khi không thấy lại lần nào trong dwell (lần thấy cuối 115s)


def test_m10_vat_dung_yen_nhan_sat_nua_so_lan_chi_bao_mot_lan():
    """Tình huống thật: ghế quán cóc đứng yên cả buổi nhưng model chỉ nhận ~nửa số lần (bị che, đổi nhãn chair / table).
    Giả lập: dwell 30s, mẫu 5s; ghế có mặt ở mẫu chẵn, mất ở mẫu lẻ, suốt 15 phút. Kèm vật thoáng qua 1/3 số mẫu.
    Đỏ thì hỏng gì: 1 cái ghế thành 174 tin báo / hết trong 15 phút (code cũ, 2026-10-22); người quản lý tắt cảnh báo.
    """
    mod = LanChiemModule("a", MOTO, dwell_sec=30)
    ev = lc_run(mod, lambda t: [xe(0.5, "chair")] if int(t / 5) % 2 == 0 else [], 0, 900)
    assert [e["event_type"] for e in ev] == ["alert_start"]
    # tách ngưỡng ra không được nới ngưỡng vào: vật chỉ có mặt 1/3 số mẫu vẫn không bao giờ được báo
    mod = LanChiemModule("a", MOTO, dwell_sec=30)
    assert lc_run(mod, lambda t: [xe(0.5, "chair")] if int(t / 5) % 3 == 0 else [], 0, 900) == []
