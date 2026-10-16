"""Chạy TamperDetector cho nhiều camera theo config.yaml, mỗi camera 1 thread.

Đọc camera bằng ffmpeg (GPU NVDEC mặc định, hwaccel: false để dùng CPU).
Link camera nằm trong .env (url_env trong config trỏ tới tên biến). File video thì khai báo url trực tiếp.
python run_tamper.py [--config config.yaml] [--log-level DEBUG] [--show]
"""
import argparse
import json
import logging
import os
import sys
import threading
import time
from collections import Counter

os.environ.setdefault("OMP_NUM_THREADS", "1")  # phải đặt trước khi import cv2/numpy

import cv2
import numpy as np
import yaml
from dotenv import load_dotenv

import tamper_detector
from config_check import check_params
from frame_source import FFmpegSource
from tamper_detector import TamperDetector

# ảnh nhỏ (320px) chạy 1 luồng nhanh hơn; thread pool mặc định của OpenCV spin-wait ăn hết core
cv2.setNumThreads(1)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
SHOW_WIDTH = 960
log = logging.getLogger("tamper")

# Tham số đọc camera: tất cả bắt buộc trong config, ghi đè được theo từng camera ở cameras[].source.
SOURCE_PARAMS = {
    "hwaccel": ("bool", 0, 0, True),          # true = giải mã GPU (NVDEC), false = CPU (dự phòng)
    "out_width": ("int", 0, 7680, True),      # 0 = giữ độ phân giải gốc
    "read_timeout": ("num", 0, 300, False),   # quá bao nhiêu giây không có frame thì kill ffmpeg
    "sample_fps": ("num", 0, 30, False),      # số frame/giây đưa vào detector
    "reconnect_sec": ("num", 0, 300, False),  # mất kết nối thì chờ bao lâu rồi thử lại
}
TOP_KEYS = ("out_dir", "source", "detector", "cameras")
CAM_KEYS = ("id", "url", "url_env", "source", "detector")


def validate_config(cfg, check_env):
    """Kiểm toàn bộ config, trả về MỌI lỗi (rỗng = hợp lệ).

    check_env=True kiểm thêm môi trường của máy chạy: biến url_env có trong .env, file video có tồn tại.
    """
    if not isinstance(cfg, dict):
        return ["config phải là một nhóm mục (key: value)"]
    errors = [f"{k}: tên mục không tồn tại (gõ nhầm? các mục hợp lệ: {', '.join(TOP_KEYS)})"
              for k in cfg if k not in TOP_KEYS]
    errors += [f"{k}: thiếu" for k in TOP_KEYS if k not in cfg]
    if "out_dir" in cfg and not (isinstance(cfg["out_dir"], str) and cfg["out_dir"]):
        errors.append(f"out_dir = {cfg['out_dir']!r}: phải là đường dẫn thư mục")
    if "source" in cfg:
        errors += check_params(SOURCE_PARAMS, cfg["source"], "source")
    det_ok = "detector" in cfg and not tamper_detector.check_cfg(cfg["detector"])
    if "detector" in cfg:
        errors += tamper_detector.check_cfg(cfg["detector"])

    cams = cfg.get("cameras")
    if "cameras" in cfg and not (isinstance(cams, list) and cams):
        errors.append("cameras: phải có ít nhất 1 camera")
        cams = []
    seen = {}
    for i, cam in enumerate(cams or []):
        p = f"cameras[{i}]"
        if not isinstance(cam, dict):
            errors.append(f"{p}: phải là một nhóm mục (key: value)")
            continue
        errors += [f"{p}.{k}: tên mục không tồn tại (gõ nhầm? các mục hợp lệ: {', '.join(CAM_KEYS)})"
                   for k in cam if k not in CAM_KEYS]
        cid = cam.get("id")
        if not (isinstance(cid, str) and cid):
            errors.append(f"{p}.id: thiếu")
        elif cid in seen:
            errors.append(f"{p}.id = {cid!r}: trùng với cameras[{seen[cid]}]")
        else:
            seen[cid] = i
        if ("url" in cam) == ("url_env" in cam):
            errors.append(f"{p}: phải có đúng một trong hai: url (file video) hoặc url_env (tên biến trong .env)")
        elif check_env and "url_env" in cam and not os.getenv(str(cam["url_env"])):
            errors.append(f"{p}.url_env = {cam['url_env']!r}: chưa có trong .env")
        elif check_env and "url" in cam and "://" not in str(cam["url"]) and not os.path.isfile(str(cam["url"])):
            errors.append(f"{p}.url = {cam['url']!r}: không tìm thấy file (đường dẫn tính từ thư mục đang chạy)")
        # phần ghi đè: kiểm từng mục có mặt; quan hệ giữa các mục kiểm trên bản đã gộp, khi bản chung hợp lệ
        if "source" in cam:
            errors += check_params(SOURCE_PARAMS, cam["source"], f"{p}.source", partial=True)
        if "detector" in cam:
            own = tamper_detector.check_cfg(cam["detector"], f"{p}.detector", partial=True)
            errors += own
            if not own and det_ok and isinstance(cam["detector"], dict):
                errors += tamper_detector.check_cfg({**cfg["detector"], **cam["detector"]}, f"{p}.detector")
    return errors


def resolve_url(cam):
    if cam.get("url_env"):
        url = os.getenv(cam["url_env"])
        if not url:
            raise ValueError(f'{cam["id"]}: thiếu biến {cam["url_env"]} trong .env')
        return url
    return cam["url"]


def emit(event, frame, out_dir):
    log.warning(json.dumps(event, ensure_ascii=False))
    if frame is not None and out_dir and event["event"] != "tamper_end":
        os.makedirs(out_dir, exist_ok=True)
        name = f'{event["cam_id"]}_{int(event["ts"] * 1000)}_{event["reason"]}.jpg'
        cv2.imwrite(os.path.join(out_dir, name), frame)
    # TODO: đẩy event qua Kafka producer khi tích hợp


def draw(frame, det, cam_id):
    """Hình camera ở trên (không bị che), dải thông tin + ảnh gốc ở dưới.
    Lưới: đỏ = mất chi tiết, xanh = bình thường, xám = ô trơn không xét. Khung cam = vùng osd_mask bỏ qua."""
    h = int(round(frame.shape[0] * SHOW_WIDTH / frame.shape[1]))
    img = cv2.resize(frame, (SHOW_WIDTH, h), interpolation=cv2.INTER_AREA)
    g = det.cfg["grid"]

    if det.ready and det.cell_info is not None:
        textured, dropped = det.cell_info
        ch, cw = h / g, SHOW_WIDTH / g
        for r in range(g):
            for c in range(g):
                color = (0, 0, 255) if dropped[r, c] else ((0, 200, 0) if textured[r, c] else (128, 128, 128))
                cv2.rectangle(img, (int(c * cw) + 2, int(r * ch) + 2),
                              (int((c + 1) * cw) - 2, int((r + 1) * ch) - 2), color, 1)

    for x1, y1, x2, y2 in det.cfg["osd_mask"]:
        cv2.rectangle(img, (int(x1 * SHOW_WIDTH), int(y1 * h)),
                      (int(x2 * SHOW_WIDTH) - 1, int(y2 * h) - 1), (0, 165, 255), 2)

    tw = SHOW_WIDTH // 4
    th = int(round(h * tw / SHOW_WIDTH))
    bar = np.zeros((th, SHOW_WIDTH, 3), np.uint8)
    if det.ref_gray is not None:
        thumb = cv2.resize(det.ref_gray.astype(np.uint8), (tw, th))
        bar[:, SHOW_WIDTH - tw:] = cv2.cvtColor(thumb, cv2.COLOR_GRAY2BGR)
        cv2.putText(bar, "anh goc", (SHOW_WIDTH - tw + 6, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)

    st = det.status()
    color = (0, 200, 0) if st == "ok" else ((0, 200, 255) if st == "warmup" else (0, 0, 255))
    cv2.putText(bar, f"{cam_id}  {st.upper()}", (10, 32), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
    items = [f"{k}={v}" for k, v in det.metrics.items()]
    for i in range(0, len(items), 3):  # 3 chỉ số mỗi dòng cho vừa chỗ
        cv2.putText(bar, "  ".join(items[i:i + 3]), (10, 64 + 26 * (i // 3)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1)
    return np.vstack([img, bar])


def run_camera(cam, url, det_cfg, src_cfg, out_dir, stop, sources, latest, show):
    cam_id = cam["id"]
    sc = {**src_cfg, **cam.get("source", {})}  # phần ghi đè theo camera là tuỳ chọn; bản chung đã đủ mục
    reconnect_sec = sc["reconnect_sec"]
    src = FFmpegSource(url, fps=sc["sample_fps"], out_width=sc["out_width"],
                       hwaccel=sc["hwaccel"], read_timeout=sc["read_timeout"], name=cam_id)
    sources.append(src)
    det = TamperDetector({**det_cfg, **cam.get("detector", {})})
    n_alarms = Counter()  # đếm cảnh báo theo loại, in tổng kết khi kết thúc
    finished, t0 = False, time.time()

    while not stop.is_set():
        if not src.open():
            ev = det.update_no_frame(time.time())
            if ev:
                emit({**ev, "cam_id": cam_id}, None, out_dir)
            stop.wait(reconnect_sec)
            continue

        while not stop.is_set():
            ok, frame = src.read()
            if not ok:
                break
            ev = det.update(frame, src.ts)
            if ev:
                emit({**ev, "cam_id": cam_id}, frame, out_dir)
                if ev["event"] == "tamper_start":
                    n_alarms[ev["reason"]] += 1
            log.debug("%s %.1f state=%s %s", cam_id, src.ts, det.status(), det.metrics)
            if show:
                latest[cam_id] = draw(frame, det, cam_id)
        src.close()

        if stop.is_set():
            break
        if src.is_file:
            finished = True
            break
        log.info("%s: mất stream, thử lại sau %.0fs", cam_id, reconnect_sec)
        ev = det.update_no_frame(time.time())
        if ev:
            emit({**ev, "cam_id": cam_id}, None, out_dir)
        stop.wait(reconnect_sec)

    # tổng kết: luôn in, kể cả khi bấm q / Ctrl+C giữa chừng
    dur = src.ts if src.is_file else time.time() - t0
    log.info("%s: %s — đã chạy %.1f phút, %d cảnh báo %s", cam_id,
             "hết video" if finished else "dừng giữa chừng", dur / 60, sum(n_alarms.values()), dict(n_alarms))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=os.path.join(BASE_DIR, "config.yaml"))
    ap.add_argument("--log-level", default="INFO")
    ap.add_argument("--show", action="store_true", help="hiện cửa sổ live (chỉ dùng trên máy có màn hình)")
    args = ap.parse_args()

    logging.basicConfig(level=args.log_level, format="%(asctime)s %(levelname)s %(message)s")
    load_dotenv(os.path.join(BASE_DIR, ".env"))
    try:
        with open(args.config) as f:
            cfg = yaml.safe_load(f)
    except (OSError, yaml.YAMLError) as e:
        log.error("%s: không đọc được config: %s", args.config, e)
        sys.exit(2)

    # config sai thì không khởi động: báo MỌI lỗi một lượt, trước khi mở camera nào
    errors = validate_config(cfg, check_env=True)
    if errors:
        log.error("%s: không khởi động, %d lỗi:\n  %s", args.config, len(errors), "\n  ".join(errors))
        sys.exit(2)

    det_cfg = cfg["detector"]
    src_cfg = cfg["source"]
    out_dir = cfg["out_dir"]
    stop = threading.Event()
    sources, latest, shown = [], {}, {}
    threads = [threading.Thread(target=run_camera, daemon=True,
                                args=(cam, resolve_url(cam), det_cfg, src_cfg, out_dir, stop,
                                      sources, latest, args.show))
               for cam in cfg["cameras"]]
    for t in threads:
        t.start()
    try:
        while any(t.is_alive() for t in threads):
            if args.show:
                for cam_id, vis in list(latest.items()):
                    if shown.get(cam_id) is not vis:  # chỉ vẽ khi có frame mới
                        cv2.imshow(cam_id, vis)
                        shown[cam_id] = vis
                if cv2.waitKey(30) & 0xFF == ord("q"):
                    break
            else:
                time.sleep(0.5)
    except KeyboardInterrupt:
        pass
    finally:
        log.info("dừng...")
        stop.set()
        for s in sources:
            s.close()
        for t in threads:
            t.join(timeout=5)
        if args.show:
            cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
