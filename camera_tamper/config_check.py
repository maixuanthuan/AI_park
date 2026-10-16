"""Kiểm một mục config theo bảng mô tả: đủ mục, không mục lạ, đúng kiểu, trong khoảng hợp lý.

Bảng mô tả: {tên: (kiểu, nhỏ nhất, lớn nhất, nhỏ nhất có được bằng không)}.
Kiểu: "num" (số), "int" (số nguyên), "bool" (true/false), "boxes" (danh sách [x1, y1, x2, y2] trong 0..1).
Khoảng chỉ để chặn lỗi gõ (vd 60 thay vì 0.6), không phải ngưỡng đã tune.
Trả về danh sách câu lỗi, rỗng = hợp lệ. Không dừng ở lỗi đầu tiên.
"""


def _is_num(v):
    return isinstance(v, (int, float)) and not isinstance(v, bool)


def check_params(spec, cfg, prefix, partial=False):
    """partial=True: chỉ kiểm các mục có mặt (dùng cho phần ghi đè theo từng camera)."""
    if not isinstance(cfg, dict):
        return [f"{prefix}: phải là một nhóm mục (key: value)"]
    dot = f"{prefix}." if prefix else ""
    errors = [f"{dot}{k}: tên mục không tồn tại (gõ nhầm? các mục hợp lệ: {', '.join(spec)})"
              for k in cfg if k not in spec]
    for k, (kind, lo, hi, lo_ok) in spec.items():
        if k not in cfg:
            if not partial:
                errors.append(f"{dot}{k}: thiếu")
            continue
        v, name = cfg[k], f"{dot}{k} = {cfg[k]!r}"
        if kind == "bool":
            if not isinstance(v, bool):
                errors.append(f"{name}: phải là true hoặc false")
        elif kind == "boxes":
            errors += _check_boxes(name, v)
        elif not _is_num(v):
            errors.append(f"{name}: phải là số")
        elif kind == "int" and v != int(v):
            errors.append(f"{name}: phải là số nguyên")
        elif not ((lo <= v if lo_ok else lo < v) and v <= hi):
            errors.append(f"{name}: phải trong khoảng {'[' if lo_ok else '('}{lo}, {hi}]")
    return errors


def _check_boxes(name, v):
    if not isinstance(v, list):
        return [f"{name}: phải là danh sách [[x1, y1, x2, y2], ...]"]
    for b in v:
        if not (isinstance(b, list) and len(b) == 4 and all(_is_num(x) for x in b)
                and 0 <= b[0] < b[2] <= 1 and 0 <= b[1] < b[3] <= 1):
            return [f"{name}: mỗi vùng phải là [x1, y1, x2, y2] với 0 <= x1 < x2 <= 1 và 0 <= y1 < y2 <= 1"]
    return []
