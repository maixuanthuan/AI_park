"""Thử KHẢ THI: model open-vocabulary (YOLOE, gõ chữ là nhận) có nhận ra băng rôn đủ dùng cho luật biểu tình không.
Chạy OFFLINE trên ảnh, ngoài DeepStream. Trả lời câu hỏi trước khi xây, không phải code sản phẩm.

  python people_count/tools/thu_bang_ron.py --data ~/bang_ron --out people_count/out/bang_ron

Dữ liệu (ảnh .jpg / .png; clip thì trích ảnh trước, 1 ảnh / 2 giây):
  <data>/co/     CÓ băng rôn / biển do NGƯỜI CẦM (biểu tình, tuần hành)     -> muốn: báo
  <data>/khong/  KHÔNG có băng rôn cầm tay: đường phố có băng rôn treo cố định, biển quảng cáo, cờ, đám đông bình thường
                 -> muốn: không báo. Nhóm này quan trọng ngang nhóm "co": đo báo nhầm ở môi trường Việt Nam.
Ra: <out>/ve/ ảnh đã vẽ (xanh lá = băng rôn gần người, cam = băng rôn không gần người, trắng = người),
    <out>/ket_qua.csv từng ảnh, và bảng tổng hợp theo ngưỡng độ tin in ra màn hình.

"Gần người" (thử luật logic đã bàn 2026-10-21, không phải luật chốt): khung băng rôn chồng ngang với một khung người,
và cạnh dưới băng rôn không cao hơn đỉnh đầu người quá nửa chiều cao người (băng rôn cầm trên tay / giơ trên đầu).
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
LOW_CONF = 0.05          # lấy hết khung >= 0.05 rồi quét ngưỡng sau, không chạy lại model
THRESHOLDS = (0.1, 0.2, 0.3, 0.4, 0.5)


def near_person(b, persons):
    """Băng rôn b có "gần người" không (xem đầu file). b, persons: (x1, y1, x2, y2) pixel."""
    for p in persons:
        h = p[3] - p[1]
        if min(b[2], p[2]) > max(b[0], p[0]) and b[3] >= p[1] - 0.5 * h and b[1] <= p[3]:
            return True
    return False


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True, help="thư mục có 2 thư mục con: co/ và khong/")
    ap.add_argument("--out", required=True)
    ap.add_argument("--model", default="yoloe-11s-seg.pt", help="lần đầu tự tải (~27 MB)")
    args = ap.parse_args(argv)

    files = [(g, os.path.join(args.data, g, f)) for g in GROUPS if os.path.isdir(os.path.join(args.data, g))
             for f in sorted(os.listdir(os.path.join(args.data, g))) if f.lower().endswith((".jpg", ".jpeg", ".png"))]
    missing = [g for g in GROUPS if not any(x == g for x, _ in files)]
    if missing:
        print(f"LỖI: thiếu ảnh trong {', '.join(os.path.join(args.data, g) for g in missing)}", file=sys.stderr)
        return 2

    model = YOLOE(args.model)
    names = BANNER + [PERSON]
    model.set_classes(names, model.get_text_pe(names))
    os.makedirs(os.path.join(args.out, "ve"), exist_ok=True)

    rows = []
    for group, path in files:
        r = model.predict(path, conf=LOW_CONF, verbose=False)[0]
        boxes = [(names[int(c)], float(s), tuple(map(int, b))) for c, s, b in
                 zip(r.boxes.cls.tolist(), r.boxes.conf.tolist(), r.boxes.xyxy.tolist())]
        persons = [b for n, s, b in boxes if n == PERSON and s >= 0.25]   # người: ngưỡng thường của detector
        banners = [(s, b, near_person(b, persons)) for n, s, b in boxes if n in BANNER]
        img = r.orig_img.copy()
        for b in persons:
            cv2.rectangle(img, b[:2], b[2:], (255, 255, 255), 1)
        for s, b, near in banners:
            color = (0, 200, 0) if near else (0, 140, 255)
            cv2.rectangle(img, b[:2], b[2:], color, 3)
            cv2.putText(img, f"{s:.2f}", (b[0], max(b[1] - 6, 14)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
        cv2.imwrite(os.path.join(args.out, "ve", f"{group}_{os.path.basename(path)}"), img)
        rows.append(dict(nhom=group, file=os.path.basename(path), so_nguoi=len(persons),
                         max_bang_ron=round(max((s for s, _, _ in banners), default=0.0), 3),
                         max_bang_ron_gan_nguoi=round(max((s for s, _, n in banners if n), default=0.0), 3)))

    with open(os.path.join(args.out, "ket_qua.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)

    n = {g: sum(r["nhom"] == g for r in rows) for g in GROUPS}
    print(f"\n{n['co']} ảnh 'co', {n['khong']} ảnh 'khong'. Mỗi ô: bắt đúng 'co' / báo nhầm 'khong' (số ảnh)")
    print(f"{'ngưỡng':>7} | {'băng rôn bất kỳ':>17} | {'băng rôn gần người':>19}")
    for t in THRESHOLDS:
        cell = []
        for key in ("max_bang_ron", "max_bang_ron_gan_nguoi"):
            hit = sum(r["nhom"] == "co" and r[key] >= t for r in rows)
            fp = sum(r["nhom"] == "khong" and r[key] >= t for r in rows)
            cell.append(f"{hit}/{n['co']} | {fp}/{n['khong']}")
        print(f"{t:>7} | {cell[0]:>17} | {cell[1]:>19}")
    print(f"\nẢnh đã vẽ: {os.path.join(args.out, 've')}  |  từng ảnh: {os.path.join(args.out, 'ket_qua.csv')}")
    return 0


if __name__ == "__main__":
    sys.exit(main())