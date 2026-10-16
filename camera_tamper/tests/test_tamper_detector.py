"""Test khoá hành vi của lõi TamperDetector (T1-T10, T13).

Mỗi test có 3 dòng: tình huống thật / phần giả lập / đỏ thì hỏng gì.
Test theo tính chất (che thì phải báo, đúng loại, đúng lúc), mốc thời gian đọc từ cấu hình.
"""
import numpy as np

from fake_camera import (covered, half_covered, kinds, noisy, partly_covered_shifted, scene,
                         shifted, watch, with_clock, with_person)
from tamper_detector import TamperDetector


def setup(cfg, **override):
    """Detector + cảnh bình thường; đã xem 20s cảnh bình thường (qua giai đoạn học ảnh gốc)."""
    c = cfg(**override)
    det = TamperDetector(c)
    rng = np.random.default_rng(1)
    s = scene()
    assert watch(det, lambda t: noisy(s, rng), 0, 20) == []
    assert det.status() == "ok"
    return c, det, rng, s


def test_t1_binh_thuong_ca_buoi_khong_bao(cfg):
    """Tình huống thật: camera bình thường cả buổi, ánh sáng ổn định, có nhiễu cảm biến.
    Giả lập: cảnh nhiều chi tiết + nhiễu ngẫu nhiên mỗi frame, 10 phút.
    Đỏ thì hỏng gì: báo nhầm liên tục, người quản lý bị làm phiền và mất tin vào hệ thống.
    """
    c, det, rng, s = setup(cfg)
    assert watch(det, lambda t: noisy(s, rng), 20, 600) == []
    assert det.status() == "ok"


def test_t2_che_kin_bao_dung_luc_roi_bao_het(cfg):
    """Tình huống thật: có người che kín ống kính 30 giây rồi bỏ ra.
    Giả lập: frame một màu đều (vật che sát ống kính thì nhòe hoàn toàn) từ giây 20 đến 50.
    Đỏ thì hỏng gì: bị phá mà không báo, hoặc báo sớm trước hold_sec (dễ báo nhầm), hoặc không báo hết.
    """
    c, det, rng, s = setup(cfg)
    ev = watch(det, lambda t: noisy(covered(), rng), 20, 30)
    assert kinds(ev) == [("tamper_start", "obstructed")]
    assert c["hold_sec"] <= ev[0]["ts"] - 20 <= c["hold_sec"] + 1

    ev = watch(det, lambda t: noisy(s, rng), 50, 30)
    assert kinds(ev) == [("tamper_end", "obstructed")]
    assert c["clear_sec"] <= ev[0]["ts"] - 50 <= c["clear_sec"] + 1


def test_t3_che_khong_kin_chi_so_dao_dong_van_bao(cfg):
    """Tình huống thật: che bằng tấm bìa nhưng tay run, lúc kín lúc hở lộ ra cảnh lệch
    (lỗi gặp ngày 2026-10-15: che mãi mà không báo, vì nhãn nhảy obstructed <-> moved làm reset bộ đếm).
    Giả lập: xen kẽ từng frame giữa che kín và che 1/4 + cảnh lệch, 30 giây.
    Đỏ thì hỏng gì: kẻ phá chỉ cần che không kín là qua mặt được hệ thống.
    """
    c, det, rng, s = setup(cfg)

    def frame(t):
        return noisy(covered() if int(t * 2) % 2 == 0 else partly_covered_shifted(s), rng)

    ev = watch(det, frame, 20, 30)
    assert [e["event"] for e in ev] == ["tamper_start"]
    assert ev[0]["ts"] - 20 <= c["hold_sec"] + 1


def test_t13_dang_bao_dong_nui_bia_ra_mot_nua_van_giu_bao_dong(cfg):
    """Tình huống thật: đang báo động vì bị che, kẻ phá nới bìa ra một nửa và giữ như vậy.
    Giả lập: che kín 10 giây (vào báo động), rồi che nửa trái 20 giây (dài hơn clear_sec), rồi bỏ hẳn.
    Đỏ thì hỏng gì: mất vùng đệm, báo động tắt bật liên tục theo tay người che, người quản lý nhận tin rác.
    """
    c, det, rng, s = setup(cfg)
    ev = watch(det, lambda t: noisy(covered(), rng), 20, 10)
    assert kinds(ev) == [("tamper_start", "obstructed")]

    assert watch(det, lambda t: noisy(half_covered(s), rng), 30, 20) == []  # vẫn giữ, không tắt bật

    ev = watch(det, lambda t: noisy(s, rng), 50, 20)
    assert kinds(ev) == [("tamper_end", "obstructed")]


def test_t4_che_truoc_roi_moi_bat_he_thong(cfg):
    """Tình huống thật: camera đã bị che từ trước khi bật hệ thống (khách hay thử tình huống này).
    Giả lập: 20 giây đầu là frame che kín, sau đó bỏ che, rồi che lại.
    Đỏ thì hỏng gì: hệ thống học nhầm cảnh bị che làm ảnh gốc, mù hoàn toàn mà không ai biết.
    """
    c = cfg()
    det = TamperDetector(c)
    rng = np.random.default_rng(1)
    s = scene()

    ev = watch(det, lambda t: noisy(covered(), rng), 0, 20)
    assert kinds(ev) == [("tamper_start", "baseline_invalid")]
    assert det.status() == "baseline_invalid"

    ev = watch(det, lambda t: noisy(s, rng), 20, 30)
    assert kinds(ev) == [("tamper_end", "baseline_invalid")]
    assert det.status() == "ok"

    # ảnh gốc phải là cảnh thật: che lại thì báo bị che
    ev = watch(det, lambda t: noisy(covered(), rng), 50, 10)
    assert kinds(ev) == [("tamper_start", "obstructed")]


def test_t5_xoay_camera_bao_dung_loai(cfg):
    """Tình huống thật: camera bị xoay sang hướng khác và để nguyên như vậy.
    Giả lập: cả cảnh trượt đi một đoạn lớn, giữ 60 giây.
    Đỏ thì hỏng gì: bị xoay mà không báo, hoặc báo sai loại (người quản lý đi tìm vật che không có).
    """
    c, det, rng, s = setup(cfg)
    ev = watch(det, lambda t: noisy(shifted(s), rng), 20, 60)
    assert kinds(ev) == [("tamper_start", "scene_changed")]
    assert det.status() == "scene_changed"


def test_t6_nguoi_di_qua_khong_bao(cfg):
    """Tình huống thật: người đi ngang qua trước camera, liên tục trong 2 phút.
    Giả lập: một mảng có chi tiết (quần áo) chiếm ~8% khung, đi từ trái sang phải lặp lại.
    Đỏ thì hỏng gì: mỗi lần có người đi qua là báo nhầm, công viên đông người thì báo liên tục.
    """
    c, det, rng, s = setup(cfg)
    assert watch(det, lambda t: noisy(with_person(s, t), rng), 20, 120) == []
    assert det.status() == "ok"


def test_t7_che_lau_khong_tu_het_bao_dong(cfg):
    """Tình huống thật: camera bị trùm túi và để nguyên 5 phút.
    Giả lập: frame che kín kéo dài 300 giây.
    Đỏ thì hỏng gì: ảnh gốc "quen" dần với cảnh bị che, báo động tự tắt trong khi camera vẫn bị che.
    """
    c, det, rng, s = setup(cfg)
    ev = watch(det, lambda t: noisy(covered(), rng), 20, 300)
    assert kinds(ev) == [("tamper_start", "obstructed")]
    assert det.status() == "obstructed"


def test_t8_hinh_dong_bang_dong_ho_van_chay(cfg):
    """Tình huống thật: đầu ghi / camera treo, hình đứng yên nhưng đồng hồ OSD vẫn chạy theo giây.
    Giả lập: một frame cố định + chữ giờ thay đổi ở dải 10% trên cùng, có khai báo osd_mask.
    Đỏ thì hỏng gì: hình đóng băng mà hệ thống tưởng bình thường, mất giám sát mà không ai biết.
    """
    c = cfg(osd_mask=[[0.0, 0.0, 1.0, 0.1]])
    det = TamperDetector(c)
    rng = np.random.default_rng(1)
    s = scene()
    assert watch(det, lambda t: with_clock(noisy(s, rng), t), 0, 20) == []

    still = noisy(s, rng)  # hình bị đóng băng: luôn đúng một frame
    ev = watch(det, lambda t: with_clock(still, t), 20, 30)
    assert kinds(ev) == [("tamper_start", "frozen")]
    assert ev[0]["ts"] - 20 <= c["frozen_sec"] + c["hold_sec"] + 1


def test_t9_mat_tin_hieu_roi_co_lai(cfg):
    """Tình huống thật: rút dây mạng camera 20 giây rồi cắm lại.
    Giả lập: gọi update_no_frame mỗi 5 giây (đúng nhịp thử kết nối lại của runner), rồi frame trở lại.
    Đỏ thì hỏng gì: mất hình mà không báo, hoặc có hình lại mà vẫn báo mất tín hiệu.
    """
    c, det, rng, s = setup(cfg)
    ev = []
    for t in (20, 25, 30, 35):
        e = det.update_no_frame(t)
        if e:
            ev.append(e)
    assert kinds(ev) == [("tamper_start", "no_signal")]

    ev = watch(det, lambda t: noisy(s, rng), 40, 20)
    assert kinds(ev) == [("tamper_end", "no_signal")]
    assert det.status() == "ok"


def test_t10_xoay_lau_khong_ai_chinh_thi_hoc_lai(cfg):
    """Tình huống thật: camera bị xoay, không ai lên chỉnh lại trong thời gian dài.
    Giả lập: cảnh trượt kéo dài quá relearn_sec (đặt 60s cho test nhanh), rồi tiếp tục 30 giây.
    Đỏ thì hỏng gì: báo động kẹt mãi mãi, và camera đó mất luôn khả năng phát hiện sự cố mới.
    """
    c, det, rng, s = setup(cfg, relearn_sec=60)
    moved = shifted(s)
    ev = watch(det, lambda t: noisy(moved, rng), 20, c["relearn_sec"] + c["hold_sec"] + 10)
    assert kinds(ev) == [("tamper_start", "scene_changed"), ("relearn", "scene_changed")]

    ev = watch(det, lambda t: noisy(moved, rng), 100, 30)
    assert ev == []
    assert det.status() == "ok"


def lose_signal(det, start, seconds):
    """Mất tín hiệu: runner gọi update_no_frame mỗi 5s khi thử kết nối lại."""
    events = []
    for t in range(start, start + seconds, 5):
        e = det.update_no_frame(t)
        if e:
            events.append(e)
    return events


def test_t27_camera_khoi_dong_lai_khong_bao_nham(cfg):
    """Tình huống thật: mất điện (camera PoE), camera khởi động lại, PTZ xoay / zoom / lấy nét rồi mới đứng yên
    (gặp thật 2026-10-16: báo nhầm "obstructed" 14s, 15s sau khi có hình lại).
    Giả lập: đúng thứ tự trong log thật: mất tín hiệu 30s, có hình lại bình thường 8s (hết no_signal),
    rồi 20s PTZ xoay / lấy nét (hình lệch, nhòe xen kẽ), rồi đứng yên ở cảnh cũ.
    Đỏ thì hỏng gì: mỗi lần mất điện là một loạt báo nhầm; nếu lỡ học ảnh gốc lúc đang chỉnh hình thì sau đó báo nhầm tiếp.
    """
    c, det, rng, s = setup(cfg)
    assert kinds(lose_signal(det, 20, 30)) == [("tamper_start", "no_signal")]

    settling = [shifted(s), covered(120)]
    ref_before = det.ref_gray.copy()
    ev = watch(det, lambda t: noisy(s, rng), 50, 8)
    ev += watch(det, lambda t: noisy(settling[int(t) % 2], rng), 58, 20)
    assert 78 < 50 + c["recover_grace_sec"]  # vẫn đang trong thời gian chờ
    assert np.array_equal(det.ref_gray, ref_before)  # không học ảnh gốc lúc camera đang xoay / chỉnh hình
    ev += watch(det, lambda t: noisy(s, rng), 78, 120)
    assert kinds(ev) == [("tamper_end", "no_signal")]
    assert det.status() == "ok"


def test_t28_rut_day_che_roi_cam_lai_van_bat_duoc(cfg):
    """Tình huống thật: kẻ phá rút dây camera, che ống kính, rồi cắm lại để qua mặt hệ thống.
    Giả lập: mất tín hiệu 30s, có hình lại nhưng đã bị che kín.
    Đỏ thì hỏng gì: thời gian chờ ổn định sau khi có hình lại bị lợi dụng để học / bỏ qua cảnh bị che.
    """
    c, det, rng, s = setup(cfg)
    lose_signal(det, 20, 30)
    ev = watch(det, lambda t: noisy(covered(), rng), 50, c["recover_grace_sec"] + c["hold_sec"] + 10)
    assert ("tamper_start", "obstructed") in kinds(ev)
    start = [e for e in ev if e["event"] == "tamper_start"][0]
    assert start["ts"] - 50 <= c["recover_grace_sec"] + c["hold_sec"] + 1
