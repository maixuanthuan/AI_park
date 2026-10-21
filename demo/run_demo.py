"""DEMO / PoC: đưa video hoặc camera vào, mỗi nguồn gán chức năng riêng, ra video đã vẽ + file event. KHÔNG phải sản phẩm.

  conda activate yoloe
  python demo/run_demo.py people_count/videos/park1.mp4=crowd duong.mp4=count CAM_01_URL=count,crowd clip.mp4=bang_ron

Mỗi đối số `nguồn=chức năng[,chức năng]`:
  nguồn     đường dẫn file video, hoặc TÊN biến camera trong people_count/.env (link có mật khẩu không gõ ra lệnh)
  chức năng count (đếm người / chó / mèo), crowd (đám đông), bang_ron (băng rôn đang được cầm)
Ra: demo/out/<giờ chạy>/<nguồn>.avi (xem bằng VLC) + events.jsonl (mọi event, 1 dòng JSON); cuối cùng in bảng tóm tắt.
Camera chạy tới --sec giây (0 = tới khi Ctrl+C). Ctrl+C: dừng gọn nguồn đang chạy, ghi xong file, bỏ các nguồn còn lại.

Perception bằng ultralytics (Python thuần, không DeepStream): YOLO26n (cùng model engine overview) cho count / crowd,
YOLOE (gõ chữ là nhận) cho bang_ron. LOGIC dùng lại nguyên các module của people_count (counter.py, crowd.py): đúng cái
lợi của việc tách perception khỏi logic (KIEN_TRUC 3.4). Lên nhiều camera thì logic đó chạy lại trên DeepStream.
Vùng: cả khung hình (PoC). Các nguồn chạy lần lượt (đủ cho demo vài nguồn, không phải cách chạy nhiều camera).
"""
import argparse
import json
import os
import sys
import time
import uuid

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # AI_park
sys.path[:0] = [os.path.join(ROOT, "people_count"), ROOT, os.path.dirname(os.path.abspath(__file__))]
os.environ.setdefault("OPENCV_FFMPEG_CAPTURE_OPTIONS", "rtsp_transport;tcp")  # camera: TCP như engine / camera_tamper

import cv2  # noqa: E402
from dotenv import load_dotenv  # noqa: E402

from bang_ron import BANNER, PERSON, BannerModule  # noqa: E402
from common.masking import mask  # noqa: E402
from counter import CountModule  # noqa: E402
from crowd import CrowdModule  # noqa: E402
from events import envelope  # noqa: E402
from meta import FrameMeta, Line, Obj, Text  # noqa: E402

FUNCS = ("count", "crowd", "bang_ron")
COCO = {0: "person", 15: "cat", 16: "dog"}   # chỉ giữ 3 lớp như engine overview
BANNER_PROMPTS = ["banner", "protest sign", "placard"]
FULL = [[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]]
COLORS = {"zone": (0, 215, 255), "info": (255, 255, 255), "alert": (60, 60, 255)}   # BGR
MODELS = os.path.join(os.path.dirname(os.path.abspath(__file__)), "models")


def parse_sources(args, env):
    """['a.mp4=count,crowd', 'CAM_01_URL=count'] -> ([(tên, uri, live, [chức năng])], [lỗi]). Báo MỌI lỗi một lượt."""
    out, errors = [], []
    for a in args:
        if "=" not in a:
            errors.append(f"'{a}': thiếu '=chức năng', vd {a}=count")
            continue
        src, fns = a.rsplit("=", 1)
        fns = [f for f in fns.split(",") if f]
        bad = [f for f in fns if f not in FUNCS]
        if not fns or bad:
            errors.append(f"'{a}': chức năng {bad or '(trống)'} không có; các chức năng: {', '.join(FUNCS)}")
        if os.path.isfile(src):
            out.append((os.path.splitext(os.path.basename(src))[0], src, False, fns))
        elif env.get(src):
            out.append((src, env[src], True, fns))
        else:
            errors.append(f"'{src}': không phải file video, cũng không phải biến camera trong people_count/.env")
    names = [n for n, *_ in out]
    errors += [f"'{n}': hai nguồn trùng tên (đổi tên file)" for n in sorted(set(names)) if names.count(n) > 1]
    return out, errors


def to_objs(xyxy, cls, conf, names, w, h):
    """Kết quả ultralytics -> Obj (bbox tỉ lệ 0..1, kẹp trong khung). names: id lớp -> tên của mình."""
    c01 = lambda v: min(max(v, 0.0), 1.0)  # noqa: E731
    return [Obj(names[int(c)], float(s), (c01(b[0] / w), c01(b[1] / h), c01(b[2] / w), c01(b[3] / h)), None)
            for b, c, s in zip(xyxy, cls, conf) if int(c) in names]


def draw(img, objs, shapes):
    h, w = img.shape[:2]
    for o in objs:
        x1, y1, x2, y2 = int(o.bbox[0] * w), int(o.bbox[1] * h), int(o.bbox[2] * w), int(o.bbox[3] * h)
        color = (0, 200, 0) if o.cls == BANNER else (0, 0, 230)
        cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
        cv2.putText(img, f"{o.cls} {o.conf:.2f}", (x1, max(y1 - 5, 12)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
    scale = max(h / 1080, 0.8)  # chữ đọc được cả ở video 720p
    for s in shapes:
        if isinstance(s, Line):
            cv2.line(img, (int(s.x1 * (w - 1)), int(s.y1 * (h - 1))), (int(s.x2 * (w - 1)), int(s.y2 * (h - 1))),
                     COLORS[s.kind], 3)
    for s in shapes:
        if isinstance(s, Text):
            org = (int(s.x * w) + 8, int(s.y * h) + int((34 + 44 * s.row) * scale))
            (tw, th), _ = cv2.getTextSize(s.text, cv2.FONT_HERSHEY_SIMPLEX, 1.0 * scale, 2)
            cv2.rectangle(img, (org[0] - 4, org[1] - th - 6), (org[0] + tw + 4, org[1] + 6), (0, 0, 0), -1)
            cv2.putText(img, s.text, org, cv2.FONT_HERSHEY_SIMPLEX, 1.0 * scale, COLORS[s.kind], 2)
    return img


def run_source(name, uri, live, fns, models, args, out_dir, events_out):
    """Chạy 1 nguồn. Trả về dòng tóm tắt (dung=True nếu bị Ctrl+C: ghi xong file rồi mới trả về)."""
    zones = [{"zone_id": f"{name}-{f}", "cam_id": name, "function": f, "polygon": FULL, "ref_size": [1, 1],
              "params": {"report_sec": args.report_sec} if f == "count" else
              {"min_people": args.min_people, "hold_sec": args.hold, "clear_sec": args.hold}, "version": 1}
             for f in fns if f in ("count", "crowd")]
    mods = {"count": lambda: CountModule(name, zones, ("person", "dog", "cat")),
            "crowd": lambda: CrowdModule(name, zones),
            "bang_ron": lambda: BannerModule(name, args.banner_conf, args.hold, args.hold,
                                             row=len([f for f in fns if f in ("count", "crowd")]))}
    det_mods = [mods[f]() for f in fns if f in ("count", "crowd")]
    ban_mods = [mods[f]() for f in fns if f == "bang_ron"]

    cap = cv2.VideoCapture(uri)
    if not cap.isOpened():
        print(f"  {name}: KHÔNG mở được nguồn ({mask(uri) if live else uri})")
        return dict(nguon=name, chuc_nang=",".join(fns), frame=0, events={}, file="-", dung=False)
    src_fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    every = max(1, round(src_fps / args.fps)) if args.fps > 0 else 1
    path = os.path.join(out_dir, f"{name}.avi")
    writer, n, i, t0, counts, stopped = None, 0, 0, time.time(), {}, False
    try:
        while True:
            ok, img = cap.read()
            if not ok or (live and args.sec and time.time() - t0 >= args.sec):
                break
            i += 1
            if (i - 1) % every:
                continue
            h, w = img.shape[:2]
            ts = time.time() if live else (i - 1) / src_fps
            shown, events = [], []
            if det_mods:
                r = models["det"].predict(img, conf=args.conf, classes=list(COCO), verbose=False)[0]
                objs = to_objs(r.boxes.xyxy.tolist(), r.boxes.cls.tolist(), r.boxes.conf.tolist(), COCO, w, h)
                shown += objs
                for m in det_mods:
                    events += m.update(FrameMeta(name, ts, w, h, tuple(objs)))
            if ban_mods:
                r = models["ban"].predict(img, conf=0.05, verbose=False)[0]
                ids = {k: (BANNER if k < len(BANNER_PROMPTS) else PERSON) for k in range(len(BANNER_PROMPTS) + 1)}
                objs = to_objs(r.boxes.xyxy.tolist(), r.boxes.cls.tolist(), r.boxes.conf.tolist(), ids, w, h)
                shown += [o for o in objs if o.cls == BANNER and o.conf >= args.banner_conf]
                for m in ban_mods:
                    events += m.update(FrameMeta(name, ts, w, h, tuple(objs)))
            for ev in events:
                emit(ev, events_out, counts)
            img = draw(img, shown, [s for m in det_mods + ban_mods for s in m.draw()])
            if writer is None:  # avi + MJPG: dừng giữa chừng vẫn xem được phần đã ghi
                writer = cv2.VideoWriter(path, cv2.VideoWriter_fourcc(*"MJPG"), src_fps / every, (w, h))
            writer.write(img)
            n += 1
            if args.show:
                cv2.imshow(name, img)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break
    except KeyboardInterrupt:
        stopped = True
    finally:  # kể cả Ctrl+C: báo cáo phần cửa sổ còn dở, đóng file
        for m in det_mods + ban_mods:
            for ev in m.flush():
                emit(ev, events_out, counts)
        cap.release()
        if writer is not None:
            writer.release()
        if args.show:
            cv2.destroyWindow(name)
    return dict(nguon=name, chuc_nang=",".join(fns), frame=n, events=counts, file=path if n else "-", dung=stopped)


def emit(ev, events_out, counts):
    line = json.dumps(envelope(ev, str(uuid.uuid4())), ensure_ascii=False)
    events_out.write(line + "\n")
    events_out.flush()
    key = f"{ev['function']}:{ev['event_type']}"
    counts[key] = counts.get(key, 0) + 1
    if ev["event_type"] != "report":  # cảnh báo in ngay để người xem thấy lúc chạy
        print(f"  {ev['cam_id']}: {ev['function']} {ev['event_type']} lúc {ev['ts']:.1f}s {ev['payload']}")


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("sources", nargs="+", help="nguồn=chức năng[,chức năng]")
    ap.add_argument("--min-people", type=int, default=5, help="crowd: từ bấy nhiêu người là đông (ngưỡng demo)")
    ap.add_argument("--hold", type=float, default=10, help="crowd / bang_ron: giữ bấy nhiêu giây mới báo / mới hết")
    ap.add_argument("--report-sec", type=float, default=30, help="count: mỗi bấy nhiêu giây 1 báo cáo")
    ap.add_argument("--conf", type=float, default=0.25, help="độ tin tối thiểu của người / chó / mèo")
    ap.add_argument("--banner-conf", type=float, default=0.3, help="độ tin tối thiểu của băng rôn (thử 2026-10-21: 0.3)")
    ap.add_argument("--fps", type=float, default=0, help="số frame / giây xử lý (0 = mọi frame, video mượt)")
    ap.add_argument("--sec", type=float, default=0, help="camera: chạy bấy nhiêu giây (0 = tới khi Ctrl+C)")
    ap.add_argument("--show", action="store_true", help="hiện cửa sổ xem trực tiếp (q: sang nguồn tiếp)")
    args = ap.parse_args(argv)

    load_dotenv(os.path.join(ROOT, "people_count", ".env"))
    sources, errors = parse_sources(args.sources, os.environ)
    if errors:
        print("LỖI, không chạy:\n  " + "\n  ".join(errors), file=sys.stderr)
        return 2

    from ultralytics import YOLO, YOLOE  # nạp muộn: sai đối số thì báo ngay, không chờ nạp torch
    os.makedirs(MODELS, exist_ok=True)
    models = {}
    if any(f in ("count", "crowd") for *_, fns in sources for f in fns):
        own = os.path.join(ROOT, "people_count", "models", "yolo26n.pt")
        models["det"] = YOLO(own if os.path.isfile(own) else os.path.join(MODELS, "yolo26n.pt"))
    if any("bang_ron" in fns for *_, fns in sources):
        models["ban"] = YOLOE(os.path.join(MODELS, "yoloe-11s-seg.pt"))
        names = BANNER_PROMPTS + [PERSON]
        models["ban"].set_classes(names, models["ban"].get_text_pe(names))

    out_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "out", time.strftime("%Y%m%d_%H%M%S"))
    os.makedirs(out_dir)
    rows = []
    with open(os.path.join(out_dir, "events.jsonl"), "w", encoding="utf-8") as events_out:
        for name, uri, live, fns in sources:
            print(f"== {name} ({'camera' if live else 'video'}): {', '.join(fns)}")
            rows.append(run_source(name, uri, live, fns, models, args, out_dir, events_out))
            if rows[-1]["dung"]:
                print("  Ctrl+C: đã ghi xong nguồn này, bỏ các nguồn còn lại")
                break

    print(f"\n== TÓM TẮT ({out_dir})")
    for r in rows:
        ev = ", ".join(f"{k} {v}" for k, v in sorted(r["events"].items())) or "không có event"
        print(f"{r['nguon']:<28} {r['chuc_nang']:<20} {r['frame']:>6} frame | {ev}\n{'':<28} vlc {r['file']}")
    print(f"event: {os.path.join(out_dir, 'events.jsonl')}")
    return 0


if __name__ == "__main__":
    sys.exit(main())