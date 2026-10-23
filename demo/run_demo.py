"""DEMO / PoC: đưa video hoặc camera vào, ra video đã vẽ + file event + ảnh bằng chứng. KHÔNG phải sản phẩm.

  conda activate dlc
  python demo/run_demo.py data/lan_chiem1.mp4 data/crowd+bang_ron1.mp4 --show     # VIDEO: chức năng theo TÊN FILE
  python demo/run_demo.py CAM_CONG=crowd,bang_ron CAM_BAI_CO=all --sec 300       # CAMERA: tên biến .env = chức năng | all
  python demo/run_demo.py --demo --show                                          # mọi video trong data/ + data/camera.txt

Video: tên = <chức năng>[+<chức năng>] + đuôi tuỳ ý bắt đầu bằng số / _ / - (lan_chiem1, phong_ue_2, crowd+bang_ron1).
Camera: TÊN biến trong people_count/.env (link có mật khẩu không gõ ra lệnh), =chức năng[,chức năng] hoặc =all (cả 5).
Chức năng: count (đếm người / chó / mèo), crowd (đám đông), bang_ron (băng rôn đang được cầm),
           lan_chiem (xe máy, ghế, xe đẩy... đứng yên quá --dwell giây; lớp + nguồn lấy từ dataset/classes.yaml),
           phong_ue (động vật ngồi xổm giữ quá --squat-hold giây; keypoint DeepLabCut, cần env có deeplabcut)
Ra: demo/out/<giờ chạy>/<nguồn>.avi (xem bằng VLC) + events.jsonl (mọi event, 1 dòng JSON) + bang_chung/*.jpg (ảnh lúc
mỗi alert_start, đường dẫn trong evidence.snapshot của event); cuối cùng in bảng tóm tắt.
Camera chạy tới --sec giây (0 = tới khi Ctrl+C). Ctrl+C: dừng gọn nguồn đang chạy, ghi xong file, bỏ các nguồn còn lại.

Perception bằng ultralytics (Python thuần, không DeepStream): YOLO26n (cùng model engine overview) cho count / crowd,
YOLOE (gõ chữ là nhận) cho bang_ron và các lớp lan_chiem COCO không có, DeepLabCut (keypoint chó, demo/dlc_pose.py)
cho phong_ue. LOGIC dùng lại nguyên các module của people_count (counter.py, crowd.py): đúng cái
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
if os.path.isdir("/usr/share/fonts/truetype"):  # cửa sổ --show (Qt của opencv pip) không tìm thấy font -> in "QFontDatabase" mỗi lần
    os.environ.setdefault("QT_QPA_FONTDIR", "/usr/share/fonts/truetype")

import cv2  # noqa: E402
import yaml  # noqa: E402
from dotenv import load_dotenv  # noqa: E402

from bang_ron import BANNER, PERSON, BannerModule  # noqa: E402
from common.masking import mask  # noqa: E402
from counter import CountModule  # noqa: E402
from crowd import CrowdModule  # noqa: E402
from events import envelope  # noqa: E402
from dlc_pose import to_animal_objs  # noqa: E402
from lan_chiem import LanChiemModule, iou  # noqa: E402
from meta import FrameMeta, Line, Obj, Text  # noqa: E402
from phong_ue import PhongUeModule  # noqa: E402

FUNCS = ("count", "crowd", "bang_ron", "lan_chiem", "phong_ue")
# Tên TÌNH HUỐNG (tiếng Việt không dấu) -> chức năng. Đặt tên video / ghi camera theo tình huống cho dễ hiểu (Thuận 2026-10-23);
# tên chức năng gốc (count, lan_chiem...) vẫn dùng được.
SITUATIONS = {"demnguoi": ["count"], "dongnguoi": ["count", "crowd"], "bieutinh": ["crowd", "bang_ron"],
              "bangron": ["bang_ron"], "lanchiem": ["lan_chiem"], "phongue": ["phong_ue"]}
NAMES = {**{f: [f] for f in FUNCS}, **SITUATIONS}
VI = {"count": "đếm", "crowd": "đám đông", "bang_ron": "băng rôn", "lan_chiem": "lấn chiếm", "phong_ue": "phóng uế"}
COCO = {0: "person", 15: "cat", 16: "dog"}   # count / crowd: 3 lớp như engine overview
# lan_chiem: các lớp vật lấn chiếm trong dataset/classes.yaml (nguồn coco / yoloe, độ tin của từng lớp lấy từ đó)
LAN_CHIEM = ("motorcycle", "bicycle", "car", "umbrella", "chair", "table", "vendor_cart", "shoulder_pole", "stall")
CLASSES_YAML = os.path.join(ROOT, "dataset", "classes.yaml")
BANNER_PROMPTS = ["banner", "protest sign", "placard"]
FULL = [[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]]
COLORS = {"zone": (0, 215, 255), "info": (255, 255, 255), "alert": (60, 60, 255)}   # BGR
MODELS = os.path.join(os.path.dirname(os.path.abspath(__file__)), "models")
MOD_FUNC = {"CountModule": "count", "CrowdModule": "crowd", "BannerModule": "bang_ron", "LanChiemModule": "lan_chiem",
            "PhongUeModule": "phong_ue"}
DEMO_DIR = os.path.join(ROOT, "data")  # clip thử của mọi bài (tên theo chức năng) + camera.txt; .gitignore chặn
VIDEO = (".mp4", ".mkv", ".avi", ".mov", ".webm")


def funcs_of(piece):
    """'bieutinh1' / 'lan_chiem' / 'phongue_2' -> [chức năng] hoặc None. Tên tình huống / chức năng + đuôi tuỳ ý bắt đầu bằng
    số / _ / - (countdown, crowded: không khớp)."""
    for name in sorted(NAMES, key=len, reverse=True):
        if piece == name or (piece.startswith(name) and piece[len(name)] in "0123456789_-"):
            return NAMES[name]
    return None


def funcs_from_name(path):
    """Tên video -> [chức năng] hoặc None. Tên = <tình huống | chức năng>[+...] + đuôi tuỳ ý (bieutinh1, lanchiem2,
    phongue_cho_den, crowd+bang_ron1). Video BẮT BUỘC đặt tên như vậy (Thuận 2026-10-23: tên mô tả kiểu "via_he" hiểu nhiều cách)."""
    fns = []
    for piece in os.path.splitext(os.path.basename(path))[0].lower().split("+"):
        got = funcs_of(piece)
        if got is None:
            return None
        fns += [f for f in got if f not in fns]
    return fns


def parse_sources(args, env):
    """['lan_chiem1.mp4', 'CAM_01_URL=crowd,bang_ron', 'CAM_02_URL=all'] -> ([(tên, uri, live, [chức năng])], [lỗi]).
    Video: chức năng theo TÊN FILE (funcs_from_name). Camera (tên biến trong people_count/.env): linh hoạt, ghi =chức năng
    hoặc =all (cả 5). Báo MỌI lỗi một lượt."""
    out, errors = [], []
    for a in args:
        src, _, spec = a.partition("=")
        if os.path.isfile(src):
            fns = funcs_from_name(src)
            if spec:
                errors.append(f"'{a}': video lấy chức năng theo TÊN FILE, bỏ phần '={spec}' (vd đặt tên lan_chiem1.mp4)")
            elif fns is None:
                errors.append(f"'{src}': tên video phải bắt đầu bằng tình huống ({', '.join(SITUATIONS)}) hoặc chức năng "
                              f"({', '.join(FUNCS)}), vd bieutinh1.mp4, lanchiem2.mp4")
            else:
                out.append((os.path.splitext(os.path.basename(src))[0], src, False, fns))
        elif env.get(src):
            got = [list(FUNCS) if x == "all" else funcs_of(x) for x in spec.lower().split(",") if x]
            bad = [x for x, g in zip([x for x in spec.lower().split(",") if x], got) if g is None]
            fns = [f for g in got if g for f in dict.fromkeys(g)]
            fns = list(dict.fromkeys(fns))
            if not fns or bad:
                errors.append(f"'{a}': camera cần tình huống / chức năng hợp lệ {bad or ''}: {src}=bieutinh, "
                              f"{src}=lanchiem,phongue hoặc {src}=all; tình huống: {', '.join(SITUATIONS)}")
            else:
                out.append((src, env[src], True, fns))
        else:
            errors.append(f"'{src}': không phải file video, cũng không phải biến camera trong people_count/.env")
    names = [n for n, *_ in out]
    errors += [f"'{n}': hai nguồn trùng tên (đổi tên file)" for n in sorted(set(names)) if names.count(n) > 1]
    return out, errors


def scan_demo(root):
    """data/ (thư mục con tuỳ ý) -> (['video', ..., 'TÊN_BIẾN=chức năng', ...], [lỗi]). Mọi video trong cây thư mục (chức năng
    theo tên file) + camera ghi trong <root>/camera.txt (mỗi dòng TÊN_BIẾN=chức năng | all; dòng trống / # bỏ qua).
    Kiểm tên / file / biến camera do parse_sources làm (một chỗ)."""
    if not os.path.isdir(root):
        return [], [f"chưa có thư mục {root}: tạo rồi bỏ video vào (tên theo chức năng, vd lan_chiem1.mp4)"]
    out = []
    for cur, dirs, files in os.walk(root):
        dirs[:] = sorted(d for d in dirs if not d.startswith("."))
        out += [os.path.join(cur, f) for f in sorted(files) if f.lower().endswith(VIDEO) and not f.startswith(".")]
    cam = os.path.join(root, "camera.txt")
    if os.path.isfile(cam):
        with open(cam) as fh:
            out += [x for x in (line.strip() for line in fh) if x and not x.startswith("#")]
    return out, [] if out else [f"{root}: chưa có video / camera nào (vd {root}/lan_chiem1.mp4, hoặc {root}/camera.txt)"]


def to_objs(xyxy, cls, conf, names, w, h):
    """Kết quả ultralytics -> Obj (bbox tỉ lệ 0..1, kẹp trong khung). names: id lớp -> tên của mình."""
    c01 = lambda v: min(max(v, 0.0), 1.0)  # noqa: E731
    return [Obj(names[int(c)], float(s), (c01(b[0] / w), c01(b[1] / h), c01(b[2] / w), c01(b[3] / h)), None)
            for b, c, s in zip(xyxy, cls, conf) if int(c) in names]


def nms(objs, thr=0.6):
    """Nhiều chữ cùng nghĩa của YOLOE (banner / placard) bắt cùng một vật -> trong cùng lớp giữ khung độ tin cao nhất."""
    keep = []
    for o in sorted(objs, key=lambda o: -o.conf):
        if all(k.cls != o.cls or iou(k.bbox, o.bbox) < thr for k in keep):
            keep.append(o)
    return keep


def load_lan_chiem(path):
    """dataset/classes.yaml -> ({id COCO: lớp}, [(chữ YOLOE, lớp)], {lớp YOLOE: độ tin}) cho các lớp LAN_CHIEM."""
    with open(path) as f:
        cls = {c["name"]: c for c in yaml.safe_load(f)["classes"]}
    missing = [n for n in LAN_CHIEM if n not in cls]
    if missing:
        raise SystemExit(f"LỖI: {path} thiếu lớp {missing}")
    coco = {cls[n]["coco"]: n for n in LAN_CHIEM if cls[n]["nguon"] == "coco"}
    prompts = [(p, n) for n in LAN_CHIEM if cls[n]["nguon"] == "yoloe" for p in cls[n]["prompts"]]
    return coco, prompts, {n: cls[n]["conf"] for n in LAN_CHIEM if cls[n]["nguon"] == "yoloe"}


def draw(img, objs, shapes):
    h, w = img.shape[:2]
    for o in objs:
        x1, y1, x2, y2 = int(o.bbox[0] * w), int(o.bbox[1] * h), int(o.bbox[2] * w), int(o.bbox[3] * h)
        color = (0, 200, 0) if o.cls == BANNER else (0, 0, 230) if o.cls in COCO.values() else (0, 140, 255)
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


def top_row(m, order):
    """Shapes của module m, dòng chữ trên cùng (Text ở góc 0,0) dời về dòng = thứ tự chức năng của m trong nguồn: count / crowd
    (people_count) đặt cứng dòng 0 / 1 -> crowd + bang_ron trên 1 nguồn từng đè nhau. Chữ khác (cạnh con vật...) giữ nguyên."""
    k = order.index(MOD_FUNC[type(m).__name__])
    return [s._replace(row=k) if isinstance(s, Text) and (s.x, s.y) == (0.0, 0.0) else s for s in m.draw()]


def run_source(name, uri, live, fns, models, args, out_dir, events_out):
    """Chạy 1 nguồn. Trả về dòng tóm tắt (dung=True nếu bị Ctrl+C: ghi xong file rồi mới trả về)."""
    zones = [{"zone_id": f"{name}-{f}", "cam_id": name, "function": f, "polygon": FULL, "ref_size": [1, 1],
              "params": {"report_sec": args.report_sec} if f == "count" else
              {"min_people": args.min_people, "hold_sec": args.hold, "clear_sec": args.hold}, "version": 1}
             for f in fns if f in ("count", "crowd")]
    order = [f for f in FUNCS if f in fns]  # dòng chữ trên video theo thứ tự cố định, không có khoảng trống
    mods = {"count": lambda: CountModule(name, zones, ("person", "dog", "cat")),
            "crowd": lambda: CrowdModule(name, zones),
            "bang_ron": lambda: BannerModule(name, args.banner_conf, args.hold, args.hold, row=order.index("bang_ron")),
            "lan_chiem": lambda: LanChiemModule(name, LAN_CHIEM, args.dwell, row=order.index("lan_chiem")),
            "phong_ue": lambda: PhongUeModule(name, args.squat_ratio, args.squat_hold, args.squat_hold,
                                              row=order.index("phong_ue"))}
    det_mods = [mods[f]() for f in fns if f in ("count", "crowd")]
    ban_mods = [mods[f]() for f in fns if f == "bang_ron"]
    lan_mods = [mods[f]() for f in fns if f == "lan_chiem"]
    pu_mods = [mods[f]() for f in fns if f == "phong_ue"]
    # model nào chạy cho nguồn này, lấy lớp nào (chỉ chạy model cần đến)
    coco_map = {**(COCO if det_mods else {}), **(models["lan_coco"] if lan_mods else {})}
    yoe_names = ({BANNER, PERSON} if ban_mods else set()) | (set(models["lan_conf"]) if lan_mods else set())

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
            det, yoe = [], []
            if coco_map:
                r = models["det"].predict(img, conf=args.conf, classes=list(coco_map), verbose=False)[0]
                det = to_objs(r.boxes.xyxy.tolist(), r.boxes.cls.tolist(), r.boxes.conf.tolist(), coco_map, w, h)
            if yoe_names:
                r = models["yoe"].predict(img, conf=0.05, verbose=False)[0]
                ids = {k: v for k, v in models["yoe_ids"].items() if v in yoe_names}
                yoe = nms(to_objs(r.boxes.xyxy.tolist(), r.boxes.cls.tolist(), r.boxes.conf.tolist(), ids, w, h))
            if det_mods:
                shown += [o for o in det if o.cls in COCO.values()]
                for m in det_mods:
                    events += m.update(FrameMeta(name, ts, w, h, tuple(det)))
            if ban_mods:
                shown += [o for o in yoe if o.cls == BANNER and o.conf >= args.banner_conf]
                for m in ban_mods:
                    events += m.update(FrameMeta(name, ts, w, h, tuple(o for o in yoe if o.cls in (BANNER, PERSON))))
            if lan_mods:
                lan = [o for o in det if o.cls in LAN_CHIEM] + \
                      [o for o in yoe if o.conf >= models["lan_conf"].get(o.cls, 2)]
                shown += lan
                for m in lan_mods:
                    events += m.update(FrameMeta(name, ts, w, h, tuple(lan)))
            if pu_mods:
                animals = to_animal_objs(*models["dlc"](img, args.dog_conf), w, h, args.dog_conf)
                shown += animals
                for m in pu_mods:
                    events += m.update(FrameMeta(name, ts, w, h, tuple(animals)))
            img = draw(img, shown, [t for m in det_mods + ban_mods + lan_mods + pu_mods for t in top_row(m, order)])
            for ev in events:  # sau khi vẽ: ảnh bằng chứng có cả khung + lý do báo
                emit(ev, events_out, counts, snapshot(img, out_dir, ev) if ev["event_type"] == "alert_start" else None)
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
        for m in det_mods + ban_mods + lan_mods + pu_mods:
            for ev in m.flush():
                emit(ev, events_out, counts)
        cap.release()
        if writer is not None:
            writer.release()
        if args.show:
            cv2.destroyWindow(name)
    return dict(nguon=name, chuc_nang=",".join(fns), frame=n, events=counts, file=path if n else "-", dung=stopped)


def snapshot(img, out_dir, ev):
    """Ảnh bằng chứng lúc alert_start (frame đã vẽ: thấy cả khung + lý do báo) -> đường dẫn cho evidence.snapshot."""
    path = os.path.join(out_dir, "bang_chung", f"{ev['cam_id']}_{ev['function']}_{ev['ts']:.1f}s.jpg")
    os.makedirs(os.path.dirname(path), exist_ok=True)
    cv2.imwrite(path, img)
    return path


def emit(ev, events_out, counts, snap=None):
    line = json.dumps(envelope(ev, str(uuid.uuid4()), snap), ensure_ascii=False)
    events_out.write(line + "\n")
    events_out.flush()
    key = f"{ev['function']}:{ev['event_type']}"
    counts[key] = counts.get(key, 0) + 1
    if ev["event_type"] != "report":  # cảnh báo in ngay để người xem thấy lúc chạy
        print(f"  {ev['cam_id']}: {ev['function']} {ev['event_type']} lúc {ev['ts']:.1f}s {ev['payload']}"
              + (f"\n    ảnh: {snap}" if snap else ""))


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("sources", nargs="*", help="(bỏ trống = MENU) video (chức năng theo tên) | TÊN_BIẾN_CAMERA=tình huống,... | =all")
    ap.add_argument("--demo", action="store_true", help="chạy HẾT mọi video trong data/ + data/camera.txt (không menu)")
    ap.add_argument("--demo-dir", default=DEMO_DIR, help=argparse.SUPPRESS)  # test truyền thư mục tạm
    ap.add_argument("--min-people", type=int, default=5, help="crowd: từ bấy nhiêu người là đông (ngưỡng demo)")
    ap.add_argument("--hold", type=float, default=10, help="crowd / bang_ron: giữ bấy nhiêu giây mới báo / mới hết")
    ap.add_argument("--report-sec", type=float, default=30, help="count: mỗi bấy nhiêu giây 1 báo cáo")
    ap.add_argument("--dwell", type=float, default=60, help="lan_chiem: đứng yên bấy nhiêu giây là lấn chiếm (demo)")
    ap.add_argument("--conf", type=float, default=0.25, help="độ tin tối thiểu của người / chó / mèo")
    ap.add_argument("--banner-conf", type=float, default=0.3, help="độ tin tối thiểu của băng rôn (thử 2026-10-21: 0.3)")
    ap.add_argument("--squat-ratio", type=float, default=0.56,
                    help="phong_ue: hông / thân dưới bấy nhiêu là ngồi xổm (đo 2026-10-22, 1 clip: 0.56 sạch, 0.58 đã nhầm)")
    ap.add_argument("--squat-hold", type=float, default=3, help="phong_ue: giữ bấy nhiêu giây mới báo / mới hết (đo: 3)")
    ap.add_argument("--dog-conf", type=float, default=0.5, help="phong_ue: độ tin tối thiểu khung con vật (CHƯA ĐO; clip 1: 0.99)")
    ap.add_argument("--fps", type=float, default=0, help="số frame / giây xử lý (0 = mọi frame, video mượt)")
    ap.add_argument("--sec", type=float, default=0, help="camera: chạy bấy nhiêu giây (0 = tới khi Ctrl+C)")
    ap.add_argument("--show", action="store_true", help="hiện cửa sổ xem trực tiếp (q: sang nguồn tiếp)")
    args = ap.parse_args(argv)

    load_dotenv(os.path.join(ROOT, "people_count", ".env"))
    menu = not args.sources and not args.demo
    raw, errors = scan_demo(args.demo_dir) if args.demo or menu else ([], [])
    if menu:  # menu: nguồn tên sai -> cảnh báo + bỏ khỏi menu, nguồn đúng vẫn demo được; có màn hình thì luôn --show
        # không màn hình (SSH vào server...): cửa sổ Qt sập cả chương trình (thấy 2026-10-23) -> chỉ ghi video
        args.show = bool(os.environ.get("DISPLAY") or os.environ.get("WAYLAND_DISPLAY"))
        if not args.show:
            print("  không có màn hình: không mở cửa sổ xem, video vẫn ghi ra demo/out/")
        ok = [x for x in raw if not parse_sources([x], os.environ)[1]]
        for x in raw:
            for e in parse_sources([x], os.environ)[1]:
                print(f"  bỏ qua: {e}")
        raw = ok
        if not raw and not errors:
            errors.append(f"{args.demo_dir}: không có video / camera nào chạy được")
    sources, more = parse_sources(args.sources + raw, os.environ)
    errors += more
    if errors:
        print("LỖI, không chạy:\n  " + "\n  ".join(errors), file=sys.stderr)
        return 2

    if menu:
        return run_menu(sources, args)
    print("== Sẽ chạy:\n" + "\n".join(f"  {n:<28} {', '.join(fns)}" for n, _, _, fns in sources))
    models = load_models({f for *_, fns in sources for f in fns})
    out_dir = new_out_dir()
    with open(os.path.join(out_dir, "events.jsonl"), "w", encoding="utf-8") as events_out:
        rows = run_many(sources, models, args, out_dir, events_out)
    summary(rows, out_dir)
    return 0


def new_out_dir():
    out_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "out", time.strftime("%Y%m%d_%H%M%S"))
    os.makedirs(out_dir, exist_ok=True)
    return out_dir


def run_many(sources, models, args, out_dir, events_out):
    rows = []
    for name, uri, live, fns in sources:
        print(f"== {name} ({'camera' if live else 'video'}): {', '.join(VI[f] for f in fns)}")
        rows.append(run_source(name, uri, live, fns, models, args, out_dir, events_out))
        if rows[-1]["dung"]:
            print("  Ctrl+C: đã ghi xong nguồn này, bỏ các nguồn còn lại")
            break
    return rows


def summary(rows, out_dir):
    print(f"\n== TÓM TẮT ({out_dir})")
    for r in rows:
        ev = ", ".join(f"{k} {v}" for k, v in sorted(r["events"].items())) or "không có event"
        print(f"{r['nguon']:<28} {r['chuc_nang']:<20} {r['frame']:>6} frame | {ev}\n{'':<28} vlc {r['file']}")
    print(f"event: {os.path.join(out_dir, 'events.jsonl')}")


def load_models(used):
    """Nạp đúng model các chức năng cần (nạp muộn: sai đối số thì báo ngay, không chờ nạp torch)."""
    os.makedirs(MODELS, exist_ok=True)
    models = {}
    if used - {"phong_ue"}:
        from ultralytics import YOLO, YOLOE
    if "phong_ue" in used:
        import dlc_pose
        models["dlc"] = dlc_pose.load()
    models["lan_coco"], lan_prompts, models["lan_conf"] = load_lan_chiem(CLASSES_YAML) if "lan_chiem" in used \
        else ({}, [], {})
    if used & {"count", "crowd"} or models["lan_coco"]:
        own = os.path.join(ROOT, "people_count", "models", "yolo26n.pt")
        models["det"] = YOLO(own if os.path.isfile(own) else os.path.join(MODELS, "yolo26n.pt"))
    # YOLOE: MỘT bộ chữ cho mọi nguồn (băng rôn + người nếu có bang_ron, lớp lan_chiem không có trong COCO)
    pairs = ([(p, BANNER) for p in BANNER_PROMPTS] + [(PERSON, PERSON)] if "bang_ron" in used else []) + lan_prompts
    if pairs:
        models["yoe"] = YOLOE(os.path.join(MODELS, "yoloe-11s-seg.pt"))
        words = [p for p, _ in pairs]
        models["yoe"].set_classes(words, models["yoe"].get_text_pe(words))
        models["yoe_ids"] = {k: n for k, (_, n) in enumerate(pairs)}
    return models


def parse_choice(text, n):
    """Câu gõ ở menu -> 'q' | [chỉ số 0..n-1] | None (gõ sai). '3' / '1 3' / '1,3' / 'a' (tất cả) / 'q' (thoát)."""
    t = text.strip().lower()
    if t in ("q", "quit", "thoat"):
        return "q"
    if t in ("a", "all", "tat ca"):
        return list(range(n))
    parts = t.replace(",", " ").split()
    if not parts or not all(p.isdigit() and 1 <= int(p) <= n for p in parts):
        return None
    return [int(p) - 1 for p in parts]


def run_menu(sources, args, ask=input):
    """MENU demo (Thuận 2026-10-23): khách muốn xem gì thì chọn số đó, chạy xong quay lại menu. Model nạp 1 lần lúc mở
    (đủ cho mọi nguồn trong menu) -> chuyển tình huống không chờ nạp lại. Ctrl+C lúc đang chạy: dừng nguồn đó, về menu."""
    print("Đang nạp model (1 lần)...")
    models = load_models({f for *_, fns in sources for f in fns})
    out_dir = new_out_dir()
    with open(os.path.join(out_dir, "events.jsonl"), "w", encoding="utf-8") as events_out:
        while True:
            print("\n== DEMO: chọn tình huống")
            for i, (name, _, live, fns) in enumerate(sources, 1):
                print(f"  {i:>2}. {name:<24} {'camera' if live else 'video '}  {', '.join(VI[f] for f in fns)}")
            print("  số = chạy (vd 2 hoặc 1 3) | a = chạy hết | q = thoát")
            try:
                pick = parse_choice(ask("Chọn: "), len(sources))
            except (EOFError, KeyboardInterrupt):
                pick = "q"
            if pick == "q":
                break
            if pick is None:
                print("  gõ số trong danh sách, a hoặc q")
                continue
            summary(run_many([sources[i] for i in pick], models, args, out_dir, events_out), out_dir)
    print(f"Kết quả cả buổi: {out_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())