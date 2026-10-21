# demo — PoC: video / camera vào, kết quả ra

**Không phải sản phẩm.** Mục đích: cho khách thấy kết quả khả thi, mỗi nguồn gán chức năng riêng.
Perception bằng ultralytics (Python thuần); logic dùng lại nguyên module của `people_count` (`counter.py`, `crowd.py`).
Khi khách đồng ý, chính logic đó chạy trên DeepStream (`people_count/run_engine.py`) để lên nhiều camera.

## Cài (1 lần)
Môi trường riêng `yoloe` (không nâng ultralytics trong `deepcv`): torch cu121 + `ultralytics` + CLIP, thêm `python-dotenv`.
Model tự tải vào `demo/models/` lần đầu (YOLO26n dùng `people_count/models/yolo26n.pt` nếu có).

## Chạy (từ thư mục `AI_park`)
```bash
conda activate yoloe
python demo/run_demo.py people_count/videos/park1.mp4=count,crowd ~/bang_ron/co/bieutinh.mp4=bang_ron
python demo/run_demo.py CAM_01_URL=count,crowd --sec 120 --show      # camera: tên biến trong people_count/.env
python demo/run_demo.py -h                                            # mọi tuỳ chọn
```
| Chức năng | Làm gì | Ngưỡng demo (tuỳ chọn) |
|---|---|---|
| `count` | đếm người / chó / mèo trong khung, báo cáo trung vị / max | `--report-sec 30` |
| `crowd` | số người >= ngưỡng, quá nửa thời gian `--hold` giây -> cảnh báo | `--min-people 5 --hold 10` |
| `bang_ron` | băng rôn đang được người cầm (băng rôn treo cao không tính), quá nửa thời gian `--hold` giây -> cảnh báo | `--banner-conf 0.3 --hold 10` |

Ra: `demo/out/<giờ chạy>/<nguồn>.avi` (VLC) + `events.jsonl`; cảnh báo in ngay lúc chạy; cuối cùng in bảng tóm tắt.
Ctrl+C: dừng gọn nguồn đang chạy (file vẫn xem được), bỏ các nguồn còn lại. `--fps 5`: chạy nhanh hơn, video giật hơn.

## Giới hạn (PoC)
- Vùng = cả khung hình. Các nguồn chạy lần lượt, không song song.
- `bang_ron`: thử khả thi 2026-10-21 mới có 1 clip biểu tình (quay điện thoại, cận) + 2 clip không: chưa kiểm góc camera cao, xa.
- Test: `python -m pytest -q demo/tests` (không cần model, GPU).