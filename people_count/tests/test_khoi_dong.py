"""Test runner (E1-E4): khởi động với config sai, chia frame theo camera, thống kê định kỳ. Không cần DeepStream:
config sai phải thoát TRƯỚC khi import pipeline (dựng engine mất 1-3 phút, không được để tới đó mới báo lỗi)."""
import os
import subprocess
import sys

import yaml

import run_engine
from meta import FrameMeta, Obj

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FULL = [[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]]


def run_with(tmp_path, cams, zones, *extra):
    """Chạy run_engine.py với profile thật, chỉ tắt tracker: file config tracker chỉ có trong container DeepStream,
    test không được phụ thuộc máy chạy test."""
    prof = yaml.safe_load(open(os.path.join(ROOT, "profiles/overview.yaml")))
    prof["tracker"]["enable"] = False
    prof["pgie_config"] = os.path.join(ROOT, "profiles", prof["pgie_config"])
    for name, data in (("cameras.yaml", cams), ("zones.yaml", zones), ("overview.yaml", prof)):
        (tmp_path / name).write_text(yaml.safe_dump(data, allow_unicode=True))
    r = subprocess.run([sys.executable, os.path.join(ROOT, "run_engine.py"), "--cameras", str(tmp_path / "cameras.yaml"),
                        "--zones", str(tmp_path / "zones.yaml"), "--profile", str(tmp_path / "overview.yaml"), *extra],
                       capture_output=True, text=True, timeout=30)
    return r.returncode, r.stdout + r.stderr


def cam(cid, **kw):
    return {"cam_id": cid, "name": cid, "url": __file__, "profile": "overview", "functions": ["count"], **kw}


def zone(zid, cid, ref_size=(1920, 1080)):
    return {"zone_id": zid, "cam_id": cid, "function": "count", "polygon": FULL, "ref_size": list(ref_size),
            "params": {"report_sec": 30}, "version": 1}


def test_e1_config_sai_thi_khong_khoi_dong_va_bao_du_loi(tmp_path):
    """Tình huống thật: devops gõ nhầm tên chức năng, backend gửi vùng cho camera không tồn tại.
    Giả lập: chạy đúng run_engine.py với 2 lỗi ở 2 file.
    Đỏ thì hỏng gì: hệ dựng pipeline mất vài phút rồi mới chết, hoặc chạy mà âm thầm không đếm.
    """
    code, out = run_with(tmp_path, {"cameras": [cam("a", functions=["cout"])]}, {"zones": [zone("z1", "khong_co")]})
    assert code == 2, out
    assert "'cout' không thuộc profile overview" in out and "'khong_co': không có trong cameras.yaml" in out, out
    assert "dựng" not in out  # chưa tới bước dựng engine


def test_e2_khong_co_camera_nao_cua_profile(tmp_path):
    """Tình huống thật: cameras.yaml chỉ có camera biển số / khuôn mặt, chạy nhầm engine overview.
    Giả lập: một camera profile plate.
    Đỏ thì hỏng gì: engine chạy rỗng, log im lặng, tưởng là đang đếm.
    """
    code, out = run_with(tmp_path, {"cameras": [cam("p", profile="plate", functions=["plate"])]}, {"zones": []})
    assert code == 2 and "không có camera nào thuộc profile overview" in out, out


def frame(cid, ts, w=1920, h=1080, n=1):
    return FrameMeta(cid, ts, w, h, tuple(Obj("person", 0.9, (0.4, 0.4, 0.5, 0.8), None) for _ in range(n)))


def test_e3_camera_sai_ti_le_khung_khong_dem_camera_khac_van_dem(caplog):
    """Tình huống thật: một camera được điền link luồng phụ 704x576 (4:3) trong khi vùng vẽ trên luồng chính 16:9.
    Giả lập: 2 camera trong cùng batch, camera b gửi frame 704x576.
    Đỏ thì hỏng gì: camera b đếm theo vùng bị méo mà không ai biết; hoặc lỗi một camera làm dừng mọi camera.
    """
    prof = yaml.safe_load(open(os.path.join(ROOT, "profiles/overview.yaml")))
    cams = [cam("a"), cam("b")]
    eng = run_engine.Engine(cams, [zone("za", "a"), zone("zb", "b")], prof)
    caplog.set_level("INFO", logger="engine")
    for i in range(61):
        eng.on_frames([frame("a", i / 2), frame("b", i / 2, 704, 576)])
    reports = [r.getMessage() for r in caplog.records if '"event_type": "report"' in r.getMessage()]
    assert len(reports) == 1 and '"cam_id": "a"' in reports[0], reports
    errors = [r.getMessage() for r in caplog.records if r.levelname == "ERROR"]
    assert len(errors) == 1 and "b: luồng 704x576" in errors[0], errors  # báo 1 lần, không lặp mỗi frame


def engine_with_clock(now):
    """Engine 2 camera a, b với đồng hồ giả: now[0] là giờ hiện tại, test tự chỉnh."""
    prof = yaml.safe_load(open(os.path.join(ROOT, "profiles/overview.yaml")))
    return run_engine.Engine([cam("a"), cam("b")], [zone("za", "a"), zone("zb", "b")], prof, clock=lambda: now[0])


def stats_lines(caplog):
    return [r.getMessage() for r in caplog.records if "thống kê" in r.getMessage()]


def test_e4_thong_ke_dinh_ky_ke_ca_khi_bang_0(caplog):
    """Tình huống thật: đêm, mọi camera mất kết nối; người vận hành cần biết hệ còn sống hay đang treo.
    Giả lập: gọi stats() trước khi có frame nào, rồi một khoảng không camera nào gửi frame.
    Đỏ thì hỏng gì: log im lặng hàng giờ, không phân biệt "không có gì" với "chương trình treo".
    """
    now = [100.0]
    eng = engine_with_clock(now)
    caplog.set_level("INFO", logger="engine")
    eng.stats()
    now[0] = 200.0
    eng.on_frames([frame("a", 0.0)])
    now[0] = 210.0
    eng.stats()
    now[0] = 220.0
    eng.stats()
    lines = stats_lines(caplog)
    assert len(lines) == 3
    assert "2 camera | chưa có frame nào" in lines[0], lines[0]
    assert "2 camera không có frame" in lines[2] and "| 0 frame" in lines[2], lines[2]


def test_e5_thong_ke_khong_tinh_luc_khoi_dong_va_dung_fps_tung_camera(caplog):
    """Tình huống thật: lần chạy đầu dựng engine mất 4 phút 40 giây; một camera chạy chậm hơn camera khác
    (thấy 2026-10-19: dòng thống kê "315s ... 32.9 fps/camera" tính cả lúc dựng engine, số vô nghĩa).
    Giả lập: 100 giây "khởi động" không có frame, rồi trong 10 giây camera a gửi 10 frame, camera b gửi 5 frame.
    Đỏ thì hỏng gì: số fps dùng để đo capacity / phát hiện camera chậm bị sai, mọi quyết định dựa trên nó sai theo.
    """
    now = [100.0]
    eng = engine_with_clock(now)
    caplog.set_level("INFO", logger="engine")
    now[0] = 200.0  # frame đầu tiên tới sau 100 giây khởi động
    eng.on_frames([frame("a", i / 2) for i in range(10)] + [frame("b", i) for i in range(5)])
    now[0] = 210.0
    eng.stats()
    line = stats_lines(caplog)[0]
    assert "thống kê 10s" in line and "| 15 frame" in line, line
    assert "fps/camera thấp nhất 0.5 (b), cao nhất 1.0 (a)" in line, line


def test_e6_bao_cao_luc_dung_cung_duoc_dem(caplog):
    """Tình huống thật: dừng chương trình giữa cửa sổ, phần cửa sổ còn dở được gửi đi
    (thấy 2026-10-19: log ghi "báo cáo 25" trong khi đã gửi 27).
    Giả lập: camera a chạy 10 giây (chưa đủ cửa sổ 30 giây), rồi finish() và stats().
    Đỏ thì hỏng gì: số báo cáo trong log lệch số báo cáo backend nhận, đối soát không khớp.
    """
    now = [100.0]
    eng = engine_with_clock(now)
    caplog.set_level("INFO", logger="engine")
    eng.on_frames([frame("a", i / 2) for i in range(20)])
    eng.finish()
    eng.stats()
    reports = [r for r in caplog.records if '"event_type": "report"' in r.getMessage()]
    assert len(reports) == 1
    assert "báo cáo 1 |" in stats_lines(caplog)[0], stats_lines(caplog)[0]


def test_e7_video_debug_phai_la_mkv(tmp_path):
    """Tình huống thật: ghi video debug từ camera (không bao giờ tự hết) rồi dừng chương trình
    (gặp 2026-10-19: file mp4 không xem được bằng VLC vì mp4 chỉ ghi mục lục lúc kết thúc).
    Giả lập: config hợp lệ, --debug-out trỏ file .mp4.
    Đỏ thì hỏng gì: dev ghi cả buổi rồi mới biết file không xem được.
    """
    code, out = run_with(tmp_path, {"cameras": [cam("a")]}, {"zones": [zone("z1", "a")]}, "--debug-out", "out/x.mp4")
    assert code == 2 and "phải là file .mkv" in out, out