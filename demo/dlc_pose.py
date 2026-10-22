"""Perception keypoint chó cho demo phong_ue: DeepLabCut SuperAnimal-Quadruped, chạy TỪNG FRAME (không qua file .h5).

File DUY NHẤT import deeplabcut (nạp muộn trong load()): đổi thư viện keypoint thì chỉ sửa file này; test không cần DeepLabCut.
Chuỗi model y như lệnh đã đo 2026-10-22 (video_inference_superanimal): detector fasterrcnn_resnet50_fpn_v2 tìm con vật ->
hrnet_w32 chấm 39 điểm trên từng khung. Bản gốc, KHÔNG video_adapt (thử 2026-10-22: độ tin tăng nhưng tách không hơn,
detector học lệch bắn vào màn chữ + người). Model tải về 1 lần (HuggingFace), sau đó chạy offline.
"""
SUPERANIMAL, POSE_MODEL, DETECTOR = "superanimal_quadruped", "hrnet_w32", "fasterrcnn_resnet50_fpn_v2"


def load(device="auto", max_individuals=10):
    """-> hàm infer(ảnh BGR của OpenCV, min_score) -> (khung xywh pixel, độ tin khung, keypoint (n, số điểm, 3), tên điểm)."""
    import numpy as np
    from deeplabcut.pose_estimation_pytorch.apis.utils import get_inference_runners
    from deeplabcut.pose_estimation_pytorch.config.pose import PoseConfig
    from deeplabcut.pose_estimation_pytorch.modelzoo.utils import get_super_animal_snapshot_path

    cfg = PoseConfig.build_for_superanimal_inference(SUPERANIMAL, model_name=POSE_MODEL, detector_name=DETECTOR,
                                                     max_individuals=max_individuals, device=device)
    names = list(cfg["metadata"]["bodyparts"])
    pose, det = get_inference_runners(
        model_config=cfg, snapshot_path=get_super_animal_snapshot_path(SUPERANIMAL, POSE_MODEL),
        max_individuals=max_individuals, num_bodyparts=len(names), num_unique_bodyparts=0,
        detector_path=get_super_animal_snapshot_path(SUPERANIMAL, DETECTOR))

    def infer(img_bgr, min_score):
        rgb = np.ascontiguousarray(img_bgr[..., ::-1])  # DeepLabCut đọc video ra RGB (VideoReader.read_frame)
        ctx = det.inference([rgb])[0]                    # {"bboxes": (n, 4) xywh, "bbox_scores": (n,)}
        # Lọc TRƯỚC khi chấm keypoint: khung yếu không cần chấm; khung rộng / cao < 2 px -> cắt ra ảnh rỗng, cv2.resize
        # trong DeepLabCut báo lỗi, sập cả demo (gặp 2026-10-22 khi kiểm bằng trọng số ngẫu nhiên).
        b, s = ctx["bboxes"], ctx["bbox_scores"]
        keep = (s >= min_score) & (b[:, 2] >= 2) & (b[:, 3] >= 2)
        ctx = {"bboxes": b[keep], "bbox_scores": s[keep]}
        if len(ctx["bboxes"]) == 0:
            return ctx["bboxes"], ctx["bbox_scores"], np.zeros((0, len(names), 3)), names
        poses = pose.inference([(rgb, ctx)])[0]["bodyparts"]
        return ctx["bboxes"], ctx["bbox_scores"], poses, names

    return infer


def to_dog_objs(boxes, scores, poses, names, w, h, min_score):
    """Kết quả DeepLabCut -> Obj("dog") có kp, toạ độ TỈ LỆ 0..1 kẹp trong khung (hợp đồng meta.py).
    Bỏ khung độ tin < min_score và hàng đệm (DeepLabCut đệm đủ max_individuals bằng -1)."""
    from meta import Obj
    c01 = lambda v: min(max(float(v), 0.0), 1.0)  # noqa: E731
    out = []
    for (x, y, bw, bh), s, kps in zip(boxes, scores, poses):
        if s < min_score or bw <= 0 or bh <= 0:
            continue
        kp = {n: (c01(k[0] / w), c01(k[1] / h), float(k[2])) for n, k in zip(names, kps)}
        out.append(Obj("dog", float(s), (c01(x / w), c01(y / h), c01((x + bw) / w), c01((y + bh) / h)), None, kp))
    return out
