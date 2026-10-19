"""Engine profile overview: mọi camera overview trong 1 pipeline DeepStream (1 model dùng chung, batch = số camera),
module đếm hiện diện cho từng camera.

python3 run_engine.py [--cameras cameras.yaml] [--zones zones.yaml] [--profile profiles/overview.yaml]
                      [--log-level INFO] [--debug-out out/x.mkv]
Đường dẫn tương đối (tham số, url video) tính từ thư mục people_count, chạy từ thư mục nào cũng được.
Mã thoát: 0 dừng bình thường (hết video, Ctrl+C, docker stop), 1 lỗi code / lỗi pipeline, 2 config sai.
"""
import argparse
import json
import logging
import os
import sys
import threading
import time
import uuid

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(BASE_DIR))  # thư mục AI_park: để import common/

import yaml  # noqa: E402
from dotenv import load_dotenv  # noqa: E402

from config import validate_all  # noqa: E402
from counter import CountModule  # noqa: E402
from events import envelope  # noqa: E402
from zones import aspect_ok  # noqa: E402

PROFILE = "overview"
MODELS_DIR = os.path.join(BASE_DIR, "models")
log = logging.getLogger("engine")


def _path(p):
    return p if os.path.isabs(p) else os.path.join(BASE_DIR, p)


def _load(path):
    with open(path) as f:
        return yaml.safe_load(f)


def emit(ev):
    log.info(json.dumps(envelope(ev, str(uuid.uuid4())), ensure_ascii=False))
    # TODO: đẩy event sang backend khi chốt kênh (KIEN_TRUC mục 8)


class Engine:
    """Nhận FrameMeta từ pipeline, chia cho module của từng camera, đếm số cho dòng thống kê."""

    def __init__(self, cams, zones, prof, clock=time.time):
        self.prof = prof
        self.clock = clock  # truyền vào để test chỉnh được giờ
        self.zones = {c["cam_id"]: [z for z in zones if z["cam_id"] == c["cam_id"]] for c in cams}
        self.modules = {c["cam_id"]: CountModule(c["cam_id"], self.zones[c["cam_id"]], prof["count"]["classes"])
                        for c in cams if "count" in c["functions"]}
        self.checked, self.bad_aspect = set(), set()
        self.frames = dict.fromkeys(self.zones, 0)
        self.reports = 0
        # mốc đầu khoảng thống kê: bắt đầu từ FRAME ĐẦU TIÊN, không từ lúc khởi động (dựng engine mất tới 4 phút 40 giây,
        # tính vào thì fps vô nghĩa — thấy 2026-10-19)
        self.t_stats = None
        self.lock = threading.Lock()  # on_frames chạy trên luồng của DeepStream, stats chạy trên luồng chính

    def on_frames(self, frames):
        for fm in frames:
            cid = fm.cam_id
            with self.lock:
                if self.t_stats is None:
                    self.t_stats = self.clock()
                self.frames[cid] += 1
            if cid not in self.checked:  # frame đầu tiên của camera: kiểm tỉ lệ khung so với lúc vẽ vùng
                self.checked.add(cid)
                bad = [z["zone_id"] for z in self.zones[cid]
                       if not aspect_ok(z["ref_size"], fm.src_w, fm.src_h, self.prof["zones"]["aspect_tol"])]
                if bad:
                    self.bad_aspect.add(cid)
                    log.error("%s: luồng %dx%d khác tỉ lệ khung lúc vẽ vùng %s -> KHÔNG đếm camera này "
                              "(sửa link sang luồng đúng tỉ lệ, hoặc vẽ lại vùng)", cid, fm.src_w, fm.src_h, bad)
                else:
                    log.info("%s: có hình %dx%d, %d vùng", cid, fm.src_w, fm.src_h, len(self.zones[cid]))
            if cid in self.bad_aspect or cid not in self.modules:
                continue
            for ev in self.modules[cid].update(fm):
                with self.lock:
                    self.reports += 1
                emit(ev)

    def stats(self):
        """Một dòng, in cả khi mọi số bằng 0: thấy đều đặn = hệ còn sống."""
        now = self.clock()
        with self.lock:
            frames, reports, t0 = self.frames, self.reports, self.t_stats
            self.frames, self.reports = dict.fromkeys(frames, 0), 0
            if t0 is not None:
                self.t_stats = now
        n = len(frames)
        if t0 is None:
            log.info("thống kê: %d camera | chưa có frame nào (đang khởi động / dựng engine / chờ camera) | báo cáo %d",
                     n, reports)
            return
        dt = max(now - t0, 1e-9)
        total, idle = sum(frames.values()), [c for c, v in frames.items() if v == 0]
        lo, hi = min(frames, key=frames.get), max(frames, key=frames.get)
        log.info("thống kê %.0fs: %d camera | %d camera không có frame%s | %d frame | fps/camera thấp nhất %.1f (%s), "
                 "cao nhất %.1f (%s) | báo cáo %d | lỗi tỉ lệ khung %d", dt, n, len(idle), f" {idle}" if idle else "",
                 total, frames[lo] / dt, lo, frames[hi] / dt, hi, reports, len(self.bad_aspect))

    def finish(self):
        for mod in self.modules.values():
            for ev in mod.flush():
                with self.lock:
                    self.reports += 1
                emit(ev)


def main(argv=None):
    """Trả về mã thoát: 0 = dừng bình thường, 1 = lỗi code / pipeline, 2 = config sai."""
    ap = argparse.ArgumentParser()
    ap.add_argument("--cameras", default="cameras.yaml")
    ap.add_argument("--zones", default="zones.yaml")
    ap.add_argument("--profile", default=f"profiles/{PROFILE}.yaml")
    ap.add_argument("--log-level", default="INFO")
    ap.add_argument("--debug-out", help="ghi video có vẽ khung (chỉ để dev xem), vd out/x.mkv")
    args = ap.parse_args(argv)

    logging.basicConfig(level=args.log_level, format="%(asctime)s %(levelname)s %(message)s")
    load_dotenv(os.path.join(BASE_DIR, ".env"))
    paths = {k: _path(getattr(args, k)) for k in ("cameras", "zones", "profile")}
    try:
        cams_cfg, zones_cfg, prof = (_load(paths[k]) for k in ("cameras", "zones", "profile"))
    except (OSError, yaml.YAMLError) as e:
        log.error("không đọc được config: %s", e)
        return 2

    # config sai thì không khởi động: báo MỌI lỗi một lượt, trước khi dựng pipeline (dựng engine mất vài phút)
    profile_dir = os.path.dirname(paths["profile"])
    errors = validate_all(cams_cfg, zones_cfg, prof, BASE_DIR, profile_dir, check_env=True)
    cams = [c for c in (cams_cfg or {}).get("cameras", []) if isinstance(c, dict) and c.get("profile") == PROFILE]
    if not errors and not cams:
        errors = [f"{paths['cameras']}: không có camera nào thuộc profile {PROFILE}"]
    if errors:
        log.error("không khởi động, %d lỗi:\n  %s", len(errors), "\n  ".join(errors))
        return 2

    srcs = []
    for c in cams:
        if "url_env" in c:
            srcs.append(dict(cam_id=c["cam_id"], uri=os.getenv(c["url_env"]), live=True))
        else:
            srcs.append(dict(cam_id=c["cam_id"], uri="file://" + os.path.abspath(_path(c["url"])), live=False))
    debug_out = os.path.abspath(_path(args.debug_out)) if args.debug_out else None
    if debug_out and not debug_out.endswith(".mkv"):
        log.error("--debug-out %s: phải là file .mkv (mp4 bị cắt ngang thì không xem được)", args.debug_out)
        return 2
    if debug_out:
        os.makedirs(os.path.dirname(debug_out), exist_ok=True)

    # engine theo số camera (batch); DeepStream-Yolo ghi engine vào thư mục đang đứng -> đứng ở models/
    # để engine nằm lại trên host (thiếu bước này thì engine mất theo container --rm, mỗi lần dựng lại 1-3 phút)
    engine_file = os.path.join(MODELS_DIR, f"model_b{len(srcs)}_gpu0_fp16.engine")
    os.chdir(MODELS_DIR)
    log.info("profile %s: %d camera, engine %s%s", PROFILE, len(srcs), os.path.basename(engine_file),
             "" if os.path.isfile(engine_file) else " (chưa có, sẽ dựng ~1-3 phút)")

    from pipeline import Pipeline  # import muộn: kiểm config không cần DeepStream

    engine = Engine(cams, zones_cfg["zones"], prof)
    pipe = Pipeline(srcs, prof, os.path.join(profile_dir, prof["pgie_config"]), engine_file,
                    engine.on_frames, debug_out)
    fatal = pipe.run(prof["stats_sec"], engine.stats)
    engine.finish()
    engine.stats()
    return 1 if fatal else 0


if __name__ == "__main__":
    sys.exit(main())