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