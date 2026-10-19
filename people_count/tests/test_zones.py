"""Test hình học vùng (Z1-Z5): điểm đại diện của đối tượng, trong / ngoài vùng, ranh giới, tỉ lệ khung."""
from zones import anchor, aspect_ok, contains

SQUARE = [[0.2, 0.2], [0.8, 0.2], [0.8, 0.8], [0.2, 0.8]]
FULL = [[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]]


def test_z1_trong_ngoai_vung():
    """Tình huống thật: người đứng giữa sân chơi và người đứng ngoài hàng rào.
    Giả lập: vùng vuông, một điểm ở giữa, một điểm ở ngoài.
    Đỏ thì hỏng gì: đếm cả người ngoài vùng, hoặc bỏ người trong vùng.
    """
    assert contains(SQUARE, 0.5, 0.5)
    assert not contains(SQUARE, 0.9, 0.5)
    assert not contains(SQUARE, 0.5, 0.1)


def test_z2_dung_tren_vach_van_tinh_la_trong_vung():
    """Tình huống thật: người đứng đúng trên vạch vùng, hoặc chân chạm mép dưới khung hình
    (thấy 2026-10-16 với nvdsanalytics: khung chạm y=1080 nằm ngoài vùng khai tới 1079 -> không đếm).
    Giả lập: điểm nằm trên cạnh, trên đỉnh, và trên mép dưới của vùng phủ cả khung.
    Đỏ thì hỏng gì: người đứng ở ranh giới lúc được đếm lúc không, số đếm nhảy theo từng frame.
    """
    assert contains(SQUARE, 0.5, 0.8)    # trên cạnh dưới
    assert contains(SQUARE, 0.2, 0.5)    # trên cạnh trái
    assert contains(SQUARE, 0.8, 0.8)    # đúng đỉnh
    assert contains(FULL, 0.5, 1.0)      # chân chạm mép dưới khung hình


def test_z3_vung_lom():
    """Tình huống thật: vùng đếm hình chữ L (bãi cỏ vòng quanh một hồ nước).
    Giả lập: polygon lõm, điểm nằm trong phần khuyết.
    Đỏ thì hỏng gì: đếm cả người đứng trong hồ / ngoài vùng khi vùng không phải hình lồi.
    """
    l_shape = [[0.0, 0.0], [1.0, 0.0], [1.0, 0.4], [0.4, 0.4], [0.4, 1.0], [0.0, 1.0]]
    assert contains(l_shape, 0.2, 0.8)
    assert contains(l_shape, 0.8, 0.2)
    assert not contains(l_shape, 0.8, 0.8)  # phần khuyết


def test_z4_diem_dai_dien_la_giua_canh_day():
    """Tình huống thật: người đứng ngoài vùng nhưng thân người chồng lên vùng trên hình (camera nhìn xiên).
    Giả lập: bbox có chân ở ngoài vùng, đầu ở trong vùng.
    Đỏ thì hỏng gì: đếm theo tâm khung thì người đứng ngoài hàng rào vẫn bị đếm vào sân chơi.
    """
    assert anchor((0.4, 0.3, 0.6, 0.9)) == (0.5, 0.9)
    box = (0.45, 0.5, 0.55, 0.85)          # đầu trong vùng, chân ở y=0.85 ngoài vùng
    assert not contains(SQUARE, *anchor(box))


def test_z5_luong_khac_ti_le_khung_bi_phat_hien():
    """Tình huống thật: vùng vẽ trên luồng chính 16:9, engine đọc luồng phụ 704x576 (4:3).
    Giả lập: so ref_size 1920x1080 với luồng 640x360 (cùng tỉ lệ) và 704x576 (khác tỉ lệ).
    Đỏ thì hỏng gì: toạ độ tỉ lệ bị méo, vùng đếm lệch chỗ mà không ai biết.
    """
    assert aspect_ok([1920, 1080], 640, 360, tol=0.02)
    assert aspect_ok([1920, 1080], 1280, 720, tol=0.02)
    assert not aspect_ok([1920, 1080], 704, 576, tol=0.02)