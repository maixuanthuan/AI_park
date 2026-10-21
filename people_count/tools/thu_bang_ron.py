"""Thử KHẢ THI: model open-vocabulary (YOLOE, gõ chữ là nhận) có nhận ra băng rôn đủ ỔN ĐỊNH THEO THỜI GIAN cho luật
biểu tình không ("có băng rôn gần người, giữ liên tục hold giây"). Chạy OFFLINE trên clip / ảnh, ngoài DeepStream.
Trả lời câu hỏi trước khi xây, không phải code sản phẩm.

  python people_count/tools/thu_bang_ron.py --data ~/bang_ron --out people_count/out/bang_ron [--hold 3] [--fps 5]

Dữ liệu (clip .mp4 .mkv .avi .mov .webm, hoặc ảnh .jpg .png; tên đặt tuỳ ý):
  <data>/co/     CÓ người CẦM băng rôn / biển suốt clip (biểu tình, tuần hành, cổ động)  -> muốn: báo
  <data>/khong/  KHÔNG ai cầm băng rôn: đường phố có băng rôn treo cố định, quảng cáo, cờ, đám đông bình thường
                 -> muốn: không báo. Nhóm này quan trọng ngang nhóm "co": đo báo nhầm ở môi trường Việt Nam.
Ra: <out>/ve/ clip (.avi) / ảnh đã vẽ (xanh lá = băng rôn gần người, cam = không gần người, trắng = người),
    <out>/ket_qua.csv từng clip, và bảng tổng hợp theo ngưỡng độ tin in ra màn hình.

Đo theo CLIP (cả clip một nhãn co / khong), mỗi ngưỡng độ tin:
  - % thời gian có băng rôn gần người (thấp = model nhận chập chờn);
  - chuỗi liên tục dài nhất (giây) có băng rôn gần người. Chuỗi >= hold: clip "co" là bắt được, clip "khong" là báo nhầm.
Ảnh lẻ tính như clip 1 frame (không có chuỗi thời gian, chỉ xét có / không).

"Gần người" (thử luật logic đã bàn 2026-10-21, không phải luật chốt): khung băng rôn chồng ngang với một khung người,
và cạnh dưới băng rôn không cao hơn đỉnh đầu người quá nửa chiều cao người (cầm trên tay / giơ trên đầu).
Băng rôn treo cố định trên cột / cây thường cao hơn hẳn đầu người hoặc không có người bên dưới.
"""
import argparse
import csv
import os
import sys

import cv2
from ultralytics import YOLOE

BANNER = ["banner", "protest sign", "placard"]   # cùng nghĩa "băng rôn / biển cầm tay"; gộp lại khi xét
PERSON = "person"
GROUPS = ("co", "khong")
VIDEO = (".mp4", ".mkv", ".avi", ".mov", ".webm")
IMAGE = (".jpg", ".jpeg", ".png")
LOW_CONF = 0.05          # lấy hết khung >= 0.05 rồi quét ngưỡng sau, không chạy lại model
THRESHOLDS = (0.1, 0.2, 0.3, 0.4, 0.5)


def near_person(b, persons):
    """Băng rôn b có "gần người" không (xem đầu file). b, persons: (x1, y1, x2, y2) pixel."""
    for p in persons:
        h = p[3] - p[1]
        if min(b[2], p[2]) > max(b[0], p[0]) and b[3] >= p[1] - 0.5 * h and b[1] <= p[3]:
            return True
    return False


def longest_run(confs, thr, step):
    """Chuỗi liên tục dài nhất (giây) các frame có băng rôn gần người >= thr; step = giây giữa 2 frame được xét."""
    best = cur = 0
    for c in confs:
        cur = cur + 1 if c >= thr else 0
        best = max(best, cur)
    return best * step


def frames_of(path, fps):
    """(frame BGR, giây giữa 2 frame được xét). Clip: lấy ~fps frame/giây; ảnh: 1 frame."""
    if path.lower().endswith(IMAGE):
        img = cv2.imread(path)
        if img is not None:
            yield img, 0.0
        return
    cap = cv2.VideoCapture(path)
    src = cap.get(cv2.CAP_PROP_FPS) or 25.0
    every = max(1, round(src / fps))
    i = 0
    while True:
        ok, img = cap.read()
        if not ok:
            break
        if i % every == 0:
            yield img, every / src
        i += 1
    cap.release()


def detect(model, names, img):
    """-> (khung người, [(độ tin, khung băng rôn, gần người)])"""
    r = model.predict(img, conf=LOW_CONF, verbose=False)[0]
    boxes = [(names[int(c)], float(s), tuple(map(int, b))) for c, s, b in
             zip(r.boxes.cls.tolist(), r.boxes.conf.tolist(), r.boxes.xyxy.tolist())]
    persons = [b for n, s, b in boxes if n == PERSON and s >= 0.25]   # người: ngưỡng thường của detector
    return persons, [(s, b, near_person(b, persons)) for n, s, b in boxes if n in BANNER]


def draw(img, persons, banners):
    for b in persons:
        cv2.rectangle(img, b[:2], b[2:], (255, 255, 255), 1)
    for s, b, near in banners:
        color = (0, 200, 0) if near else (0, 140, 255)
        cv2.rectangle(img, b[:2], b[2:], color, 3)
        cv2.putText(img, f"{s:.2f}", (b[0], max(b[1] - 6, 14)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
    return img


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True, help="thư mục có 2 thư mục con: co/ và khong/")
    ap.add_argument("--out", required=True)
    ap.add_argument("--model", default="yoloe-11s-seg.pt", help="lần đầu tự tải (~27 MB)")
    ap.add_argument("--hold", type=float, default=3.0, help="giữ liên tục bấy nhiêu giây mới tính là báo")
    ap.add_argument("--fps", type=float, default=5.0, help="số frame / giây được xét trong mỗi clip")
    args = ap.parse_args(argv)

    files = [(g, os.path.join(args.data, g, f)) for g in GROUPS if os.path.isdir(os.path.join(args.data, g))
             for f in sorted(os.listdir(os.path.join(args.data, g))) if f.lower().endswith(VIDEO + IMAGE)]
    missing = [g for g in GROUPS if not any(x == g for x, _ in files)]
    if missing:
        print(f"LỖI: thiếu clip / ảnh trong {', '.join(os.path.join(args.data, g) for g in missing)}", file=sys.stderr)
        return 2

    model = YOLOE(args.model)
    names = BANNER + [PERSON]
    model.set_classes(names, model.get_text_pe(names))
    os.makedirs(os.path.join(args.out, "ve"), exist_ok=True)

    rows = []
    for group, path in files:
        base = f"{group}_{os.path.splitext(os.path.basename(path))[0]}"
        is_img = path.lower().endswith(IMAGE)
        confs, step, writer = [], 0.0, None
        for img, step in frames_of(path, args.fps):
            persons, banners = detect(model, names, img)
            confs.append(max((s for s, _, n in banners if n), default=0.0))
            img = draw(img, persons, banners)
            if is_img:
                cv2.imwrite(os.path.join(args.out, "ve", base + ".jpg"), img)
                continue
            if writer is None:  # avi + MJPG: dừng giữa chừng vẫn xem được phần đã ghi (VLC)
                writer = cv2.VideoWriter(os.path.join(args.out, "ve", base + ".avi"), cv2.VideoWriter_fourcc(*"MJPG"),
                                         1.0 / step, (img.shape[1], img.shape[0]))
            writer.write(img)
        if writer is not None:
            writer.release()
        if not confs:
            print(f"BỎ QUA {path}: không đọc được frame nào")
            continue
        row = dict(nhom=group, file=os.path.basename(path), so_frame_xet=len(confs), dai_s=round(len(confs) * step, 1))
        for t in THRESHOLDS:
            row[f"pt_thoi_gian@{t}"] = round(100 * sum(c >= t for c in confs) / len(confs))
            # ảnh lẻ: không có thời gian, "chuỗi" = có (>= hold) / không (0)
            row[f"chuoi_dai_nhat_s@{t}"] = (args.hold if confs[0] >= t else 0.0) if is_img else \
                round(longest_run(confs, t, step), 1)
        rows.append(row)
        print(f"{group}/{os.path.basename(path)}: {len(confs)} frame xét, chuỗi dài nhất @0.3 = {row['chuoi_dai_nhat_s@0.3']}s")

    with open(os.path.join(args.out, "ket_qua.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)

    n = {g: sum(r["nhom"] == g for r in rows) for g in GROUPS}
    print(f"\n{n['co']} clip 'co', {n['khong']} clip 'khong'. Báo = băng rôn gần người liên tục >= {args.hold:g}s")
    print(f"{'ngưỡng':>7} | {'bắt đúng co':>12} | {'báo nhầm khong':>15} | {'% thời gian TB co / khong':>26}")
    for t in THRESHOLDS:
        hit = sum(r["nhom"] == "co" and r[f"chuoi_dai_nhat_s@{t}"] >= args.hold for r in rows)
        fp = sum(r["nhom"] == "khong" and r[f"chuoi_dai_nhat_s@{t}"] >= args.hold for r in rows)
        pct = [sum(r[f"pt_thoi_gian@{t}"] for r in rows if r["nhom"] == g) / max(n[g], 1) for g in GROUPS]
        print(f"{t:>7} | {hit:>5}/{n['co']:<6} | {fp:>6}/{n['khong']:<8} | {pct[0]:>11.0f}% / {pct[1]:.0f}%")
    print(f"\nClip / ảnh đã vẽ: {os.path.join(args.out, 've')}  |  từng clip: {os.path.join(args.out, 'ket_qua.csv')}")
    return 0


if __name__ == "__main__":
    sys.exit(main())