"""Vùng (KIEN_TRUC mục 6, bản 0.1): kiểm zones.yaml, hình học vùng. Không phụ thuộc DeepStream.

Toạ độ vùng và bbox đều theo tỉ lệ 0..1 -> đúng ở mọi độ phân giải (vẽ trên luồng chính, đọc luồng phụ).
"""
from common.config_check import check_params

ZONE_KEYS = ("zone_id", "cam_id", "function", "polygon", "ref_size", "params", "version")
# Ngưỡng NGHIỆP VỤ của từng chức năng, nằm trong params của vùng (KIEN_TRUC mục 7). Bắt buộc đủ mục.
FUNCTION_PARAMS = {
    "count": {
        "report_sec": ("num", 0, 86400, False),  # mỗi bao nhiêu giây gửi 1 báo cáo đếm của vùng
    },
}
NEEDS_ZONE = ("count",)  # chức năng không có vùng thì không làm được gì -> camera bật mà thiếu vùng là lỗi
_EPS = 1e-9


def anchor(bbox):
    """Điểm đại diện của đối tượng: giữa cạnh đáy khung (chân người / chân con vật), không phải tâm khung.
    Camera nhìn xiên: thân người đứng ngoài hàng rào có thể chồng lên vùng trên hình, chân thì không."""
    x1, _, x2, y2 = bbox
    return (x1 + x2) / 2, y2


def _on_segment(px, py, ax, ay, bx, by):
    if abs((bx - ax) * (py - ay) - (by - ay) * (px - ax)) > _EPS:
        return False
    return min(ax, bx) - _EPS <= px <= max(ax, bx) + _EPS and min(ay, by) - _EPS <= py <= max(ay, by) + _EPS


def contains(polygon, x, y):
    """Điểm (x, y) có trong polygon không. Nằm TRÊN cạnh / đỉnh tính là trong: người đứng trên vạch,
    hay chân chạm mép dưới khung (y = 1.0), không bị đếm lúc có lúc không (thấy 2026-10-16 với nvdsanalytics)."""
    inside = False
    n = len(polygon)
    for i in range(n):
        (ax, ay), (bx, by) = polygon[i], polygon[(i + 1) % n]
        if _on_segment(x, y, ax, ay, bx, by):
            return True
        if (ay > y) != (by > y) and x < ax + (y - ay) * (bx - ax) / (by - ay):
            inside = not inside
    return inside


def aspect_ok(ref_size, w, h, tol):
    """Tỉ lệ khung của luồng đang đọc có khớp khung lúc vẽ vùng không (lệch tương đối <= tol).
    Luồng phụ có thể khác tỉ lệ (vd chính 16:9, phụ 704x576 = 4:3) -> toạ độ tỉ lệ bị méo."""
    rw, rh = ref_size
    return abs((w / h) / (rw / rh) - 1) <= tol


def _is_num(v):
    return isinstance(v, (int, float)) and not isinstance(v, bool)


def check_zones(data, cameras):
    """Kiểm zones.yaml. cameras: {cam_id: [chức năng đã bật]}. Trả về MỌI lỗi (rỗng = hợp lệ)."""
    if not (isinstance(data, dict) and set(data) == {"zones"} and isinstance(data["zones"], list)):
        return ["zones.yaml: phải có đúng một mục 'zones' là danh sách vùng (có thể rỗng)"]
    errors, seen, has_zone = [], {}, set()
    for i, z in enumerate(data["zones"]):
        p = f"zones[{i}]"
        if not isinstance(z, dict):
            errors.append(f"{p}: phải là một nhóm mục (key: value)")
            continue
        errors += [f"{p}.{k}: tên mục không tồn tại (gõ nhầm? các mục hợp lệ: {', '.join(ZONE_KEYS)})"
                   for k in z if k not in ZONE_KEYS]
        errors += [f"{p}.{k}: thiếu" for k in ZONE_KEYS if k not in z]

        zid = z.get("zone_id")
        if "zone_id" in z and not (isinstance(zid, str) and zid):
            errors.append(f"{p}.zone_id = {zid!r}: phải là chuỗi không rỗng")
        elif zid in seen:
            errors.append(f"{p}.zone_id = {zid!r}: trùng với zones[{seen[zid]}]")
        elif zid:
            seen[zid] = i

        cid, fn = z.get("cam_id"), z.get("function")
        if "cam_id" in z and cid not in cameras:
            errors.append(f"{p}.cam_id = {cid!r}: không có trong cameras.yaml")
        elif "function" in z and cid in cameras:
            if fn not in cameras[cid]:
                errors.append(f"{p}.function = {fn!r}: camera {cid} không bật chức năng này")
            else:
                has_zone.add((cid, fn))

        if "polygon" in z:
            poly = z["polygon"]
            if not (isinstance(poly, list) and len(poly) >= 3 and all(
                    isinstance(pt, list) and len(pt) == 2 and all(_is_num(v) and 0 <= v <= 1 for v in pt)
                    for pt in poly)):
                errors.append(f"{p}.polygon: phải là danh sách >= 3 điểm [x, y] theo TỈ LỆ 0..1 (không phải pixel)")
        if "ref_size" in z:
            rs = z["ref_size"]
            if not (isinstance(rs, list) and len(rs) == 2 and all(isinstance(v, int) and not isinstance(v, bool)
                                                                   and v > 0 for v in rs)):
                errors.append(f"{p}.ref_size = {rs!r}: phải là [rộng, cao] số nguyên dương của khung lúc vẽ vùng")
        if "version" in z:
            v = z["version"]
            if not (isinstance(v, int) and not isinstance(v, bool) and v >= 1):
                errors.append(f"{p}.version = {v!r}: phải là số nguyên >= 1")
        if "params" in z and fn in FUNCTION_PARAMS:
            errors += check_params(FUNCTION_PARAMS[fn], z["params"], f"{p}.params")

    for cid, fns in cameras.items():
        errors += [f"{cid}: bật {fn} nhưng chưa có vùng {fn} nào trong zones.yaml"
                   for fn in fns if fn in NEEDS_ZONE and (cid, fn) not in has_zone]
    return errors