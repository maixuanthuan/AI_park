"""Camera giả cho test: dựng cảnh, vật che, người đi qua, đồng hồ OSD, và cho detector "xem" theo thời gian.

Frame 640x480 (detector tự thu về rộng 320). Thời gian truyền tay, 2 frame/giây như hệ thật,
nên test không phụ thuộc đồng hồ máy và chạy nhanh hơn thời gian thực.
"""
import cv2
import numpy as np

FPS = 2.0  # hệ thật lấy 2 frame/giây (sample_fps)
H, W = 480, 640


def scene(seed=0):
    """Cảnh công viên giả: trời trơn ở 1/4 trên, phía dưới là các khối đồ vật cố định có cạnh rõ.

    Khối 40px: đủ thưa để cạnh không phủ kín khung sau khi detector nới cạnh (dilate),
    nên phép so cạnh (edge_keep) phân biệt được cảnh cũ và cảnh bị xoay.
    """
    rng = np.random.default_rng(seed)
    small = rng.integers(0, 256, (H // 40, W // 40), np.uint8)
    img = cv2.resize(small, (W, H), interpolation=cv2.INTER_NEAREST)
    img[: H // 4] = 180
    return cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)


def noisy(img, rng):
    """Thêm nhiễu cảm biến: camera thật không bao giờ cho 2 frame giống hệt nhau."""
    return cv2.add(img, rng.integers(0, 6, img.shape, np.uint8))


def covered(value=200):
    """Vật che sát ống kính: nằm ngoài tầm lấy nét nên thành một mảng nhòe đều màu."""
    return np.full((H, W, 3), value, np.uint8)


def shifted(img, dy=97, dx=131):
    """Camera bị xoay: cả cảnh trượt đi một đoạn lớn, lệch khỏi lưới khối để cạnh không trùng lại."""
    return np.roll(img, (dy, dx), axis=(0, 1))


def partly_covered_shifted(img, frac=0.25):
    """Che bằng bìa không kín: 1/4 trái bị che, phần còn lại lộ ra cảnh lệch (tay cầm bìa bị lệch).

    Đo ngày 2026-10-15 trên cảnh giả: obstruct 0.49 (< 0.6), edge_keep 0.12 -> lúc mới vào bị xếp
    scene_changed, xen kẽ với che kín (obstructed) thì tái hiện đúng lỗi "nhãn nhảy qua lại".
    Che 1/2 cho obstruct 0.63 -> cả hai đều obstructed, không tái hiện được lỗi đó.
    """
    out = shifted(img)
    out[:, : int(W * frac)] = 200
    return out


def half_covered(img, frac=0.5):
    """Che dở dang: tay nới bìa ra, nửa trái vẫn bị che, nửa phải lộ cảnh đúng vị trí.

    Đo ngày 2026-10-15 trên cảnh giả: obstruct 0.43, edge_keep 0.52 -> nằm giữa vùng đệm
    (chưa đủ để vào báo động, nhưng cũng chưa đủ sạch để thoát), cách mỗi ngưỡng gần nhất 0.08.
    """
    out = img.copy()
    out[:, : int(W * frac)] = 200
    return out


def with_person(img, t, size=(220, 110), speed=60.0, seed=99):
    """Người đi ngang qua: một mảng có chi tiết (quần áo), chiếm ~8% khung, đi từ trái sang phải."""
    ph, pw = size
    rng = np.random.default_rng(seed)
    small = rng.integers(0, 256, (ph // 10, pw // 10), np.uint8)
    patch = cv2.resize(small, (pw, ph), interpolation=cv2.INTER_NEAREST)
    x = int(t * speed) % (W - pw)
    y = H - ph - 20
    out = img.copy()
    out[y:y + ph, x:x + pw] = patch[..., None]
    return out


def with_clock(img, t, band=0.1):
    """Đồng hồ OSD chạy theo giây, nằm gọn trong dải `band` trên cùng (như camera văn phòng)."""
    out = img.copy()
    text = f"2026-10-15 14:{int(t) // 60 % 60:02d}:{int(t) % 60:02d}"
    cv2.putText(out, text, (10, int(H * band) - 12), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2)
    return out


def watch(det, frame_at, start, seconds):
    """Cho detector xem từ giây `start` trong `seconds` giây; frame_at(t) trả về frame tại thời điểm t.

    Trả về danh sách event phát ra trong khoảng đó (mỗi event có trường 'ts').
    """
    events = []
    for i in range(int(round(seconds * FPS))):
        t = start + i / FPS
        ev = det.update(frame_at(t), t)
        if ev:
            events.append(ev)
    return events


def kinds(events):
    """Rút gọn event thành (loại, lý do) để so sánh cho dễ đọc."""
    return [(e["event"], e["reason"]) for e in events]
