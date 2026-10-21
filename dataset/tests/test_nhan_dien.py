"""Test danh sách lớp + gán nhãn tự động (N1-N4)."""
import os

import yaml

from nhan_dien import load_classes, nms, yolo_line

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def test_n1_danh_sach_lop_dang_giao_hop_le_va_giu_thu_tu():
    """Tình huống thật: classes.yaml là nguồn duy nhất của id lớp cho nhãn và model chung.
    Giả lập: đọc đúng file trong repo.
    Đỏ thì hỏng gì: công cụ từ chối chạy, hoặc thứ tự lớp đổi -> nhãn cũ mang nghĩa lớp khác (hỏng cả bộ dữ liệu).
    """
    cls, err = load_classes(os.path.join(HERE, "classes.yaml"))
    assert err == []
    assert [c["name"] for c in cls][:4] == ["person", "dog", "cat", "banner"]  # đã chốt, chỉ được thêm vào cuối


def test_n2_danh_sach_lop_sai_bao_du_mot_luot(tmp_path):
    """Tình huống thật: thêm lớp mới và gõ sai: trùng tên, quên prompts cho lớp YOLOE, conf dạng phần trăm.
    Giả lập: 3 lớp sai 3 kiểu.
    Đỏ thì hỏng gì: chạy cả giờ trên clip rồi mới thấy nhãn sai / thiếu lớp.
    """
    p = tmp_path / "c.yaml"
    p.write_text(yaml.safe_dump({"classes": [
        {"name": "person", "coco": 0, "nguon": "coco", "conf": 0.25},
        {"name": "person", "coco": 0, "nguon": "coco", "conf": 0.25},
        {"name": "cart", "nguon": "yoloe", "conf": 30}]}))
    _, err = load_classes(str(p))
    assert any("'person': trùng" in e for e in err), err
    assert any("nguon yoloe nhưng thiếu 'prompts'" in e for e in err), err
    assert any("conf = 30" in e for e in err), err


def test_n3_nhieu_chu_cung_nghia_cung_mot_vat_chi_giu_mot_khung():
    """Tình huống thật: YOLOE gõ "banner" và "placard" cùng bắt một tấm băng rôn; hai xe đẩy đứng cạnh nhau.
    Giả lập: 2 khung banner trùng nhau, 2 khung vendor_cart rời nhau, 1 khung banner trùng chỗ nhưng khác lớp với xe đẩy.
    Đỏ thì hỏng gì: một vật thành 2 nhãn (model học sai), hoặc 2 vật thật bị gộp mất 1.
    """
    out = nms([("banner", 0.5, (0, 0, 100, 50)), ("banner", 0.4, (2, 1, 101, 52)),
               ("vendor_cart", 0.6, (200, 0, 300, 100)), ("vendor_cart", 0.5, (310, 0, 410, 100)),
               ("vendor_cart", 0.3, (0, 0, 100, 50))])
    assert sorted((k, s) for k, s, _ in out) == [("banner", 0.5), ("vendor_cart", 0.3), ("vendor_cart", 0.5),
                                                 ("vendor_cart", 0.6)]


def test_n4_dong_nhan_yolo_dung_dinh_dang_va_kep_trong_anh():
    """Tình huống thật: CVAT / ultralytics đọc nhãn dạng "id cx cy w h" tỉ lệ 0..1; khung lấn ra mép ảnh.
    Giả lập: ảnh 1000x500, khung (100, 50, 300, 450) và khung lấn mép phải (900, 0, 1100, 100).
    Đỏ thì hỏng gì: CVAT không đọc được nhãn, hoặc khung lệch chỗ khi mở ra sửa.
    """
    assert yolo_line(4, (100, 50, 300, 450), 1000, 500) == "4 0.200000 0.500000 0.200000 0.800000"
    assert yolo_line(0, (900, 0, 1100, 100), 1000, 500) == "0 0.950000 0.100000 0.100000 0.200000"
