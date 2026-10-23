"""Thử nhận diện + gán nhãn tự động, MỘT công cụ: chạy model trên clip theo danh sách lớp dataset/classes.yaml.

  conda activate yoloe
  python dataset/nhan_dien.py data/via_he [--fps 0] [--label-sec 2] [--out dataset/out/x]

Mỗi lớp lấy từ COCO (YOLO26n, model đếm hiện tại) và / hoặc YOLOE (gõ chữ). Ra trong <out>/:
  ve/<clip>.avi              video đã vẽ (nguồn chính của mỗi lớp, mỗi lớp một màu) -> nhìn tận mắt lớp nào nhận được
  thong_ke.csv + bảng in ra  theo lớp: % thời gian thấy, độ tin TB, số tối đa / frame; lớp có cả 2 nguồn thì so cả hai
  yolo/images, yolo/labels   frame lấy mẫu mỗi --label-sec giây + nhãn chuẩn YOLO (ĐỦ mọi lớp trong ảnh, theo nguồn chính)
  yolo/data.yaml             tên lớp theo đúng thứ tự classes.yaml -> mở trong CVAT ("Ultralytics YOLO Detection") để SỬA,
                             không vẽ từ đầu; cũng là đầu vào train sau này
Nhãn tự động CHƯA PHẢI đáp án: phải qua người sửa mới dùng để train / đánh giá.
Mọi lớp phải có nhãn trong mọi ảnh: ảnh biểu tình mà chỉ khoanh băng rôn, bỏ trống người -> model học "người là nền".
"""
import argparse
import csv
import os
import sys
import time

import yaml

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)  # AI_park
VIDEO = (".mp4", ".mkv", ".avi", ".mov", ".webm")
LOW_CONF = 0.05   # lấy hết khung >= 0.05, lọc theo conf từng lớp sau (thống kê và nhãn dùng chung một lần chạy model)
NMS_IOU = 0.6     # nhiều chữ cùng nghĩa của YOLOE (banner / placard) bắt cùng một vật -> giữ khung độ tin cao nhất


def load_classes(path):
    """-> (danh sách lớp, [lỗi]). Báo MỌI lỗi một lượt; sai thì không chạy (nhãn sai id là hỏng cả bộ dữ liệu)."""
    with open(path) as f:
        data = yaml.safe_load(f)
    cls = (data or {}).get("classes")
    if not isinstance(cls, list) or not cls:
        return [], [f"{path}: phải có mục 'classes' là danh sách lớp"]
    errors, seen = [], set()
    for i, c in enumerate(cls):
        p = f"classes[{i}]"
        if not isinstance(c, dict):
            errors.append(f"{p}: phải là một nhóm mục")
            continue
        errors += [f"{p}.{k}: tên mục không tồn tại (các mục: name, coco, prompts, nguon, conf)"
                   for k in c if k not in ("name", "coco", "prompts", "nguon", "conf")]
        name = c.get("name")
        if not (isinstance(name, str) and name):
            errors.append(f"{p}.name: thiếu")
        elif name in seen:
            errors.append(f"{p}.name = {name!r}: trùng")
        seen.add(name)
        src = c.get("nguon")
        if src not in ("coco", "yoloe"):
            errors.append(f"{p}.nguon = {src!r}: phải là coco hoặc yoloe")
        if src == "coco" and not isinstance(c.get("coco"), int):
            errors.append(f"{p}: nguon coco nhưng thiếu id 'coco'")
        if src == "yoloe" and not (isinstance(c.get("prompts"), list) and c["prompts"]):
            errors.append(f"{p}: nguon yoloe nhưng thiếu 'prompts'")
        if not (isinstance(c.get("conf"), (int, float)) and 0 < c["conf"] < 1):
            errors.append(f"{p}.conf = {c.get('conf')!r}: phải là số trong (0, 1)")
    return cls, errors


def iou(a, b):
    w, h = min(a[2], b[2]) - max(a[0], b[0]), min(a[3], b[3]) - max(a[1], b[1])
    if w <= 0 or h <= 0:
        return 0.0
    inter = w * h
    return inter / ((a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter)


def nms(dets, thr=NMS_IOU):
    """dets: [(lớp, độ tin, xyxy)]. Trong CÙNG lớp, khung trùng >= thr thì giữ khung độ tin cao nhất."""
    keep = []
    for d in sorted(dets, key=lambda d: -d[1]):
        if all(k[0] != d[0] or iou(k[2], d[2]) < thr for k in keep):
            keep.append(d)
    return keep


def yolo_line(cid, b, w, h):
    """Khung pixel -> 1 dòng nhãn YOLO: id cx cy w h (tỉ lệ 0..1, kẹp trong ảnh)."""
    x1, y1, x2, y2 = max(b[0], 0), max(b[1], 0), min(b[2], w), min(b[3], h)
    return f"{cid} {(x1 + x2) / 2 / w:.6f} {(y1 + y2) / 2 / h:.6f} {(x2 - x1) / w:.6f} {(y2 - y1) / h:.6f}"


def color(i):
    """Màu BGR cố định theo id lớp, dễ phân biệt."""
    pal = [(0, 0, 230), (0, 200, 255), (255, 120, 0), (0, 200, 0), (255, 0, 200), (200, 255, 0), (0, 128, 255),
           (180, 0, 255), (255, 255, 0), (0, 255, 128), (128, 128, 255), (255, 128, 128), (90, 200, 200)]
    return pal[i % len(pal)]


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("clips", nargs="+", help="file clip hoặc thư mục chứa clip")
    ap.add_argument("--classes", default=os.path.join(HERE, "classes.yaml"))
    ap.add_argument("--out", default=os.path.join(HERE, "out", time.strftime("%Y%m%d_%H%M%S")))
    # mặc định mọi frame: video để NHÌN TẬN MẮT phải mượt (bản đầu để 2 -> video giật, Thuận bất ngờ 2 lần, 2026-10-21).
    # Nhiều clip dài cần nhanh thì --fps 2. Ảnh nhãn không phụ thuộc --fps (theo --label-sec).
    ap.add_argument("--fps", type=float, default=0, help="số frame / giây chạy model (0 = mọi frame; 2 = nhanh, video giật)")
    ap.add_argument("--label-sec", type=float, default=2, help="mỗi bấy nhiêu giây lấy 1 frame làm ảnh + nhãn (0 = không)")
    args = ap.parse_args(argv)

    classes, errors = load_classes(args.classes)
    clips = []
    for c in args.clips:
        if os.path.isdir(c):
            clips += [os.path.join(c, f) for f in sorted(os.listdir(c)) if f.lower().endswith(VIDEO)]
        elif os.path.isfile(c):
            clips.append(c)
        else:
            errors.append(f"{c}: không có file / thư mục này")
    if not clips and not errors:
        errors.append("không có clip nào")
    if errors:
        print("LỖI, không chạy:\n  " + "\n  ".join(errors), file=sys.stderr)
        return 2

    import cv2
    from ultralytics import YOLO, YOLOE

    names = [c["name"] for c in classes]
    coco_map = {c["coco"]: c["name"] for c in classes if isinstance(c.get("coco"), int)}
    prompts, prompt_map = [], {}
    for c in classes:
        for p in c.get("prompts") or []:
            prompt_map[len(prompts)] = c["name"]
            prompts.append(p)
    models_dir = os.path.join(ROOT, "demo", "models")
    os.makedirs(models_dir, exist_ok=True)
    own = os.path.join(ROOT, "people_count", "models", "yolo26n.pt")
    coco = YOLO(own if os.path.isfile(own) else os.path.join(models_dir, "yolo26n.pt")) if coco_map else None
    yoloe = None
    if prompts:
        yoloe = YOLOE(os.path.join(models_dir, "yoloe-11s-seg.pt"))
        yoloe.set_classes(prompts, yoloe.get_text_pe(prompts))

    conf = {c["name"]: c["conf"] for c in classes}
    main_src = {c["name"]: c["nguon"] for c in classes}
    for d in ("ve", "yolo/images", "yolo/labels"):
        os.makedirs(os.path.join(args.out, d), exist_ok=True)
    rows, n_img = [], 0
    for clip in clips:
        base = os.path.splitext(os.path.basename(clip))[0]
        cap = cv2.VideoCapture(clip)
        src_fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
        every = max(1, round(src_fps / args.fps)) if args.fps > 0 else 1
        label_every = round(args.label_sec * src_fps) if args.label_sec > 0 else 0
        # thống kê: (lớp, nguồn) -> [số frame có thấy, tổng độ tin cao nhất / frame, số tối đa trong 1 frame]
        st = {(n, s): [0, 0.0, 0] for n in names for s in ("coco", "yoloe")}
        writer, i, n = None, 0, 0
        print(f"== {base}: {src_fps:.0f} fps, chạy model {src_fps / every:.1f} frame/giây")
        while True:
            ok, img = cap.read()
            if not ok:
                break
            i += 1
            if (i - 1) % every:
                continue
            h, w = img.shape[:2]
            dets = {"coco": [], "yoloe": []}
            if coco is not None:
                r = coco.predict(img, conf=LOW_CONF, classes=list(coco_map), verbose=False)[0]
                dets["coco"] = [(coco_map[int(k)], float(s), b) for k, s, b in
                                zip(r.boxes.cls.tolist(), r.boxes.conf.tolist(), r.boxes.xyxy.tolist())]
            if yoloe is not None:
                r = yoloe.predict(img, conf=LOW_CONF, verbose=False)[0]
                dets["yoloe"] = nms([(prompt_map[int(k)], float(s), b) for k, s, b in
                                     zip(r.boxes.cls.tolist(), r.boxes.conf.tolist(), r.boxes.xyxy.tolist())])
            for src, ds in dets.items():
                for name in names:
                    hits = [s for k, s, _ in ds if k == name and s >= conf[name]]
                    if hits:
                        a = st[(name, src)]
                        a[0], a[1], a[2] = a[0] + 1, a[1] + max(hits), max(a[2], len(hits))
            final = [d for src, ds in dets.items() for d in ds if main_src[d[0]] == src and d[1] >= conf[d[0]]]
            if label_every and (i - 1) % label_every < every:  # frame lấy mẫu gần nhất mỗi --label-sec giây
                stem = f"{base}_{i - 1:06d}"
                cv2.imwrite(os.path.join(args.out, "yolo/images", stem + ".jpg"), img)
                with open(os.path.join(args.out, "yolo/labels", stem + ".txt"), "w") as f:
                    f.write("".join(yolo_line(names.index(k), b, w, h) + "\n" for k, _, b in final))
                n_img += 1
            for k, s, b in final:
                col = color(names.index(k))
                cv2.rectangle(img, (int(b[0]), int(b[1])), (int(b[2]), int(b[3])), col, 2)
                cv2.putText(img, f"{k} {s:.2f}", (int(b[0]), max(int(b[1]) - 5, 12)), cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                            col, 2)
            if writer is None:  # avi + MJPG: dừng giữa chừng vẫn xem được
                writer = cv2.VideoWriter(os.path.join(args.out, "ve", base + ".avi"), cv2.VideoWriter_fourcc(*"MJPG"),
                                         src_fps / every, (w, h))
            writer.write(img)
            n += 1
        cap.release()
        if writer is not None:
            writer.release()
        for (name, src), (k, sc, mx) in st.items():
            has = isinstance(next(c for c in classes if c["name"] == name).get("coco"), int) if src == "coco" else \
                bool(next(c for c in classes if c["name"] == name).get("prompts"))
            if has:
                rows.append(dict(clip=base, lop=name, nguon=src, chinh="x" if main_src[name] == src else "",
                                 frame_xet=n, pt_thoi_gian=round(100 * k / max(n, 1)),
                                 do_tin_tb=round(sc / k, 2) if k else 0, toi_da_1_frame=mx))

    with open(os.path.join(args.out, "thong_ke.csv"), "w", newline="") as f:
        wr = csv.DictWriter(f, fieldnames=list(rows[0]))
        wr.writeheader()
        wr.writerows(rows)
    with open(os.path.join(args.out, "yolo", "data.yaml"), "w") as f:
        yaml.safe_dump({"path": os.path.abspath(os.path.join(args.out, "yolo")), "train": "images", "val": "images",
                        "names": dict(enumerate(names))}, f, allow_unicode=True, sort_keys=False)

    print(f"\n== THỐNG KÊ THEO LỚP ({len(clips)} clip; % thời gian thấy / độ tin TB / tối đa 1 frame; * = nguồn chính)")
    print(f"{'lớp':<15} {'COCO (YOLO26n)':>22} {'YOLOE':>22}")
    for name in names:
        cell = []
        for src in ("coco", "yoloe"):
            rs = [r for r in rows if r["lop"] == name and r["nguon"] == src]
            if not rs:
                cell.append("-")
                continue
            fr = sum(r["frame_xet"] for r in rs)
            pt = sum(r["pt_thoi_gian"] * r["frame_xet"] for r in rs) / max(fr, 1)
            hit = [r for r in rs if r["do_tin_tb"]]
            dt = sum(r["do_tin_tb"] for r in hit) / len(hit) if hit else 0
            cell.append(f"{'*' if rs[0]['chinh'] else ' '}{pt:3.0f}% / {dt:.2f} / {max(r['toi_da_1_frame'] for r in rs)}")
        print(f"{name:<15} {cell[0]:>22} {cell[1]:>22}")
    print(f"\nVideo đã vẽ: {os.path.join(args.out, 've')}  |  từng clip: {os.path.join(args.out, 'thong_ke.csv')}")
    print(f"Nhãn tự động: {n_img} ảnh trong {os.path.join(args.out, 'yolo')} (mở bằng CVAT để sửa)")
    return 0


if __name__ == "__main__":
    sys.exit(main())