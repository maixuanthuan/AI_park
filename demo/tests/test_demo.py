"""Test công cụ demo (M1-M15): đọc đối số nguồn=chức năng, đổi kết quả model, module băng rôn. Thời gian truyền tay."""
from bang_ron import BannerModule
from meta import FrameMeta, Line, Obj, Text
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


# ---------- phong_ue ----------
from dlc_pose import to_animal_objs  # noqa: E402
from phong_ue import PhongUeModule, hip_ratio  # noqa: E402

W, H = 1280, 720  # clip đo 2026-10-22: khung KHÔNG vuông


def cho(ratio, body=100.0, conf=0.9, x=400.0, y=280.0):
    """Chó dài `body` px (cổ -> gốc đuôi nằm ngang), chân sau thấp hơn gốc đuôi ratio * body px. kp tỉ lệ 0..1."""
    px = {"neck_base": (x + body, y), "tail_base": (x, y), "back_left_paw": (x + 5, y + ratio * body),
          "back_right_paw": (x - 5, y + ratio * body), "nose": (x + body + 20, y - 10)}
    kp = {k: (a / W, b / H, conf) for k, (a, b) in px.items()}
    return Obj("animal", 0.95, (x / W, (y - 20) / H, (x + body + 30) / W, (y + body) / H), None, kp)


def test_m11_chieu_cao_hong_tinh_bang_pixel_va_bo_cho_khong_do_duoc():
    """Tình huống thật: camera 1280x720, chó đứng (hông 0.64 thân) / ngồi xổm (0.54); có lúc model không chắc điểm khớp,
    có lúc chó ở rất xa (thân 25 px).
    Giả lập: dựng keypoint bằng pixel rồi đổi sang tỉ lệ 0..1 như hợp đồng meta.
    Đỏ thì hỏng gì: tính trên toạ độ tỉ lệ thì trục x, y khác thước (1280 vs 720) -> ngưỡng 0.56 đo được vô nghĩa;
    điểm khớp mù / chó quá nhỏ mà vẫn tính -> báo nhầm.
    """
    assert abs(hip_ratio(cho(0.64), W, H, 0.3, 40) - 0.64) < 1e-6
    assert abs(hip_ratio(cho(0.54), W, H, 0.3, 40) - 0.54) < 1e-6
    assert hip_ratio(cho(0.54, conf=0.2), W, H, 0.3, 40) is None       # model không chắc điểm khớp
    assert hip_ratio(cho(0.54, body=25), W, H, 0.3, 40) is None        # chó quá nhỏ, keypoint nhảy
    assert hip_ratio(Obj("animal", 0.9, (0, 0, 1, 1), None), W, H, 0.3, 40) is None   # không có keypoint


def pu_run(mod, dog_at, start, seconds, fps=15.0):
    ev = []
    for i in range(int(round(seconds * fps))):
        t = start + i / fps
        o = dog_at(t)
        ev += mod.update(FrameMeta("a", t, W, H, (o,) if o else ()))
    return ev


def test_m12_ngoi_xom_du_lau_moi_bao_cui_ngui_thoang_qua_khong_bao():
    """Tình huống thật: chó đi dạo, cúi ngửi 1s (hông hạ), rồi ngồi xổm phóng uế 6s, rồi đi tiếp (clip đo 2026-10-22).
    Giả lập: 15 fps; hông 0.64 lúc đi, 0.54 lúc ngửi 1s và lúc phóng uế 6s; ngưỡng 0.56, cửa sổ 3s.
    Đỏ thì hỏng gì: cúi ngửi cũng báo (người quản lý bị gọi vô ích), hoặc phóng uế thật không báo / báo nhiều lần.
    """
    mod = PhongUeModule("a", 0.56, 3, 3)
    dang = lambda t: cho(0.54 if 10 <= t < 11 or 16 <= t < 22 else 0.64)  # noqa: E731
    ev = pu_run(mod, dang, 0, 30)
    assert [e["event_type"] for e in ev] == ["alert_start", "alert_end"]
    assert 16 < ev[0]["ts"] <= 19.5 and ev[0]["function"] == "phong_ue"   # báo trong ~3s sau khi ngồi
    assert ev[0]["payload"]["hong_than"] == 0.54
    assert 22 < ev[1]["ts"] <= 25.5                                          # hết trong ~3s sau khi đứng dậy


def test_m13_ket_qua_deeplabcut_doi_sang_obj_bo_hang_dem_va_khung_yeu():
    """Tình huống thật: DeepLabCut trả khung xywh pixel, đệm đủ max_individuals bằng -1, có khung yếu (bóng cây);
    detector không phân biệt loài -> mọi con vật là "animal" (Thuận chốt: chỉ cần biết động vật phóng uế).
    Giả lập: 3 hàng: chó thật (0.95), khung yếu (0.2), hàng đệm (-1); khung chó lấn ra mép phải ảnh.
    Đỏ thì hỏng gì: khung yếu / hàng đệm thành "chó" -> đo dáng trên rác; bbox sai định dạng -> vẽ lệch, hợp đồng meta vỡ.
    """
    import numpy as np
    names = ["neck_base", "tail_base"]
    boxes = np.array([[1200, 360, 160, 90], [10, 10, 50, 50], [-1, -1, -1, -1]], float)
    scores = np.array([0.95, 0.2, -1])
    poses = np.array([[[1250, 380, 0.9], [1210, 390, 0.8]], [[20, 20, 0.5], [30, 30, 0.5]], [[-1, -1, -1], [-1, -1, -1]]])
    objs = to_animal_objs(boxes, scores, poses, names, W, H, 0.5)
    assert len(objs) == 1 and objs[0].cls == "animal" and objs[0].conf == 0.95
    assert objs[0].bbox == (1200 / W, 0.5, 1.0, 450 / H)                     # xywh -> xyxy tỉ lệ, kẹp mép phải
    assert objs[0].kp["neck_base"] == (1250 / W, 380 / H, 0.9)



def test_m14_hinh_ve_the_hien_dung_luat_va_khop_phep_do():
    """Tình huống thật: người xem video demo phải hiểu VÌ SAO hệ gọi là ngồi xổm (Thuận 2026-10-22: bản đầu vẽ đường chéo
    gốc đuôi -> chân, không khớp phép đo, nhìn không hiểu).
    Giả lập: 1 con đứng (hông 0.70) và 1 con ngồi (0.50) cùng frame 1280x720, thân 100 px, ngưỡng 0.56.
    Đỏ thì hỏng gì: thanh hông vẽ sai chỗ / vạch ngưỡng sai độ sâu -> hình nói một đằng, luật làm một nẻo; khách hỏi không giải thích được.
    """
    mod = PhongUeModule("a", 0.56, 3, 3)
    mod.update(FrameMeta("a", 0.0, W, H, (cho(0.70, x=200), cho(0.50, x=800))))
    shapes = mod.draw()
    lines = [s for s in shapes if isinstance(s, Line)]
    texts = [s.text for s in shapes if isinstance(s, Text)]
    for x, r, kind in ((200, 0.70, "zone"), (800, 0.50, "alert")):
        d = (x / W, 280 / H)
        hip = [ln for ln in lines if (ln.x1, ln.y1, ln.x2) == (*d, d[0])]           # thanh ĐỨNG thả từ gốc đuôi
        assert len(hip) == 1 and abs(hip[0].y2 - (280 + r * 100) / H) < 1e-9 and hip[0].kind == kind
        mark = [ln for ln in lines if ln.kind == "info" and ln.y1 == ln.y2 and ln.x1 < d[0] < ln.x2]
        assert any(abs(ln.y1 - (280 + 0.56 * 100) / H) < 1e-9 for ln in mark)     # vạch ngưỡng đúng độ sâu 0.56 thân
    assert "hong 0.70 DUNG" in texts and "hong 0.50 < 0.56 NGOI XOM" in texts and {"C", "D", "P"} <= set(texts)
    assert any(t.startswith("phong ue: ngoi xom 100% / 3s") for t in texts)


def test_m15_canh_bao_kem_anh_bang_chung(tmp_path):
    """Tình huống thật: khách cần ẢNH làm bằng chứng mỗi lần báo (Thuận chốt 2026-10-22).
    Giả lập: video 6s tự dựng; perception giả (thay DeepLabCut) luôn thấy 1 con vật ngồi xổm; chạy run_source thật.
    Đỏ thì hỏng gì: event báo mà không có ảnh, hoặc đường dẫn ảnh trong event trỏ tới file không có.
    """
    import json
    import os
    import types

    import cv2
    import numpy as np
    from run_demo import run_source

    video = str(tmp_path / "v.avi")
    wr = cv2.VideoWriter(video, cv2.VideoWriter_fourcc(*"MJPG"), 15, (320, 180))
    for _ in range(90):
        wr.write(np.full((180, 320, 3), 90, np.uint8))
    wr.release()
    o = cho(0.50, body=400)  # khung 320x180 = 1/4 của 1280x720: thân còn 100 px (>= min_body_px 40)
    fake = lambda img, conf: (np.array([[100.0, 60.0, 80.0, 50.0]]), np.array([0.99]),  # noqa: E731
                              np.array([[[o.kp[n][0] * 320, o.kp[n][1] * 180, 0.9] for n in o.kp]]), list(o.kp))
    args = types.SimpleNamespace(report_sec=30, min_people=5, hold=10, banner_conf=0.3, dwell=60, squat_ratio=0.56,
                                 squat_hold=3, dog_conf=0.5, conf=0.25, fps=0, sec=0, show=False)
    with open(tmp_path / "events.jsonl", "w") as f:
        row = run_source("v", video, False, ["phong_ue"], {"dlc": fake}, args, str(tmp_path), f)
    evs = [json.loads(line) for line in open(tmp_path / "events.jsonl")]
    assert row["events"] == {"phong_ue:alert_start": 1} and len(evs) == 1
    snap = evs[0]["evidence"]["snapshot"]
    assert snap and os.path.isfile(snap) and cv2.imread(snap).shape == (180, 320, 3)
