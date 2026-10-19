"""Kiểm 3 file config của engine (KIEN_TRUC mục 5). Không phụ thuộc DeepStream.

- cameras.yaml          devops: camera nào, profile nào, bật chức năng nào (không có con số kỹ thuật)
- zones.yaml            backend: vùng + ngưỡng nghiệp vụ
- profiles/<p>.yaml     dev: thông số kỹ thuật, có số đo
Mọi mục bắt buộc, không mặc định ngầm. Trả về MỌI lỗi một lượt (rỗng = hợp lệ).
"""
import os

from common.config_check import check_params
from zones import check_zones

# Module logic mỗi profile chạy được. Chỉ ghi module ĐÃ CÓ: khai chức năng chưa có module thì báo lỗi ngay.
PROFILES = {"overview": ("count",), "face": ("face",), "plate": ("plate",)}
CAM_KEYS = ("cam_id", "name", "url", "url_env", "profile", "functions")

# (kiểu, nhỏ nhất, lớn nhất, nhỏ nhất có được bằng không). Khoảng chỉ chặn lỗi gõ, giá trị thật nằm trong file.
PROFILE_KEYS = ("stats_sec", "pgie_config", "mux", "tracker", "rtsp", "count", "zones")
PROFILE_NUM = {"stats_sec": ("num", 0, 86400, False)}
MUX = {"width": ("int", 64, 7680, True), "height": ("int", 64, 4320, True),
       "batched_push_timeout_us": ("int", 0, 10_000_000, False)}
TRACKER = {"enable": ("bool", 0, 0, True), "width": ("int", 32, 3840, True), "height": ("int", 32, 2160, True)}
RTSP = {"reconnect_sec": ("int", 0, 300, False), "latency_ms": ("int", 0, 10000, True)}
ZONES = {"aspect_tol": ("num", 0, 0.5, False)}


def _str_ok(v):
    return isinstance(v, str) and v != ""


def check_cameras(data, base_dir, check_env):
    """check_env=True kiểm thêm máy đang chạy: biến url_env có trong .env, file video có tồn tại
    (đường dẫn tương đối tính từ thư mục people_count). Trả về (lỗi, {cam_id: [chức năng]})."""
    if not (isinstance(data, dict) and set(data) == {"cameras"} and isinstance(data["cameras"], list)):
        return ["cameras.yaml: phải có đúng một mục 'cameras' là danh sách camera"], {}
    errors, seen, funcs = [], {}, {}
    for i, cam in enumerate(data["cameras"]):
        p = f"cameras[{i}]"
        if not isinstance(cam, dict):
            errors.append(f"{p}: phải là một nhóm mục (key: value)")
            continue
        errors += [f"{p}.{k}: tên mục không tồn tại (gõ nhầm? các mục hợp lệ: {', '.join(CAM_KEYS)})"
                   for k in cam if k not in CAM_KEYS]
        errors += [f"{p}.{k}: thiếu" for k in ("cam_id", "name", "profile", "functions") if k not in cam]

        cid = cam.get("cam_id")
        if "cam_id" in cam and not _str_ok(cid):
            errors.append(f"{p}.cam_id = {cid!r}: phải là chuỗi không rỗng")
        elif cid in seen:
            errors.append(f"{p}.cam_id = {cid!r}: trùng với cameras[{seen[cid]}]")
        elif cid:
            seen[cid] = i
        if "name" in cam and not _str_ok(cam["name"]):
            errors.append(f"{p}.name = {cam['name']!r}: phải là chuỗi không rỗng")

        if ("url" in cam) == ("url_env" in cam):
            errors.append(f"{p}: phải có đúng một trong hai: url (file video) hoặc url_env (tên biến trong .env)")
        elif check_env and "url_env" in cam and not os.getenv(str(cam["url_env"])):
            errors.append(f"{p}.url_env = {cam['url_env']!r}: chưa có trong .env")
        elif check_env and "url" in cam and not os.path.isfile(os.path.join(base_dir, str(cam["url"]))):
            errors.append(f"{p}.url = {cam['url']!r}: không tìm thấy file (đường dẫn tính từ thư mục people_count)")

        prof = cam.get("profile")
        if "profile" in cam and prof not in PROFILES:
            errors.append(f"{p}.profile = {prof!r}: không tồn tại (các profile: {', '.join(PROFILES)})")
        fns = cam.get("functions")
        if "functions" in cam:
            if not (isinstance(fns, list) and fns):
                errors.append(f"{p}.functions: phải có ít nhất 1 chức năng, vd [count]")
            else:
                for j, fn in enumerate(fns):
                    if fn in fns[:j]:
                        errors.append(f"{p}.functions: {fn!r} khai 2 lần")
                    elif prof in PROFILES and fn not in PROFILES[prof]:
                        errors.append(f"{p}.functions: {fn!r} không thuộc profile {prof} "
                                      f"(profile này có: {', '.join(PROFILES[prof])})")
                if _str_ok(cid):
                    funcs[cid] = [f for f in fns if isinstance(f, str)]
    return errors, funcs


def _labels(pgie_path):
    """Đọc labels của model qua dòng labelfile-path trong config nvinfer (nvinfer tính đường dẫn từ thư mục file config)."""
    with open(pgie_path) as f:
        for line in f:
            if line.strip().startswith("labelfile-path="):
                path = os.path.join(os.path.dirname(pgie_path), line.split("=", 1)[1].strip())
                with open(path) as lf:
                    return [l.strip() for l in lf if l.strip()]
    raise ValueError("không có dòng labelfile-path")


def check_profile(data, profile_dir, check_env):
    """check_env=True kiểm thêm file chỉ có trong container DeepStream (config tracker)."""
    if not isinstance(data, dict):
        return ["profile: phải là một nhóm mục (key: value)"]
    errors = [f"{k}: tên mục không tồn tại (gõ nhầm? các mục hợp lệ: {', '.join(PROFILE_KEYS)})"
              for k in data if k not in PROFILE_KEYS]
    errors += [f"{k}: thiếu" for k in PROFILE_KEYS if k not in data]
    errors += check_params(PROFILE_NUM, {k: data[k] for k in PROFILE_NUM if k in data}, "", partial=True)
    for key, spec in (("mux", MUX), ("rtsp", RTSP), ("zones", ZONES)):
        if key in data:
            errors += check_params(spec, data[key], key)

    if "tracker" in data:
        t = data["tracker"]
        if isinstance(t, dict):
            errors += check_params(TRACKER, {k: v for k, v in t.items() if k != "config"}, "tracker")
            if "config" not in t:
                errors.append("tracker.config: thiếu")
            elif not _str_ok(t["config"]):
                errors.append(f"tracker.config = {t['config']!r}: phải là đường dẫn file config của tracker")
            elif check_env and t.get("enable") is True and not os.path.isfile(t["config"]):
                errors.append(f"tracker.config = {t['config']!r}: không tìm thấy file (chạy trong container DeepStream?)")
        else:
            errors.append("tracker: phải là một nhóm mục (key: value)")

    labels = None
    if "pgie_config" in data:
        path = os.path.join(profile_dir, str(data["pgie_config"]))
        try:
            labels = _labels(path)
        except (OSError, ValueError) as e:
            errors.append(f"pgie_config = {data['pgie_config']!r}: không đọc được config / labels của model: {e}")

    if "count" in data:
        c = data["count"]
        cls = c.get("classes") if isinstance(c, dict) else None
        if not (isinstance(c, dict) and set(c) == {"classes"}):
            errors.append("count: phải có đúng một mục 'classes'")
        elif not (isinstance(cls, list) and cls and all(_str_ok(x) for x in cls)):
            errors.append("count.classes: phải là danh sách tên lớp, vd [person, dog, cat]")
        elif labels is not None:
            errors += [f"count.classes: {x!r} không có trong labels của model" for x in cls if x not in labels]
    return errors


def validate_all(cams, zones, prof, base_dir, profile_dir, check_env):
    """Kiểm cả 3 file. base_dir = thư mục people_count, profile_dir = thư mục chứa file profile."""
    errors, funcs = check_cameras(cams, base_dir, check_env)
    errors += check_zones(zones, funcs)
    errors += check_profile(prof, profile_dir, check_env)
    return errors