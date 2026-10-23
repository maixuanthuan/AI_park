# demo — PoC: video / camera vào, kết quả ra

**Không phải sản phẩm.** Mục đích: cho khách thấy kết quả khả thi, mỗi nguồn gán chức năng riêng.
Perception bằng ultralytics (Python thuần); logic dùng lại nguyên module của `people_count` (`counter.py`, `crowd.py`).
Khi khách đồng ý, chính logic đó chạy trên DeepStream (`people_count/run_engine.py`) để lên nhiều camera.

## Cài (1 lần)
Môi trường `dlc` chạy được MỌI chức năng (gồm `phong_ue`): Python 3.10 + torch cu121 + `deeplabcut` (`pip install --pre`)
+ `ultralytics` + CLIP + `python-dotenv` (Claude kiểm 2026-10-22: deeplabcut 3.0.2 + ultralytics 8.4.183 chạy chung, đủ test).
Env cũ `yoloe` vẫn chạy được mọi chức năng TRỪ `phong_ue`. Không nâng ultralytics trong `deepcv` (Detect2026 đang dùng).
Model tự tải lần đầu: YOLO / YOLOE vào `demo/models/` (YOLO26n dùng `people_count/models/yolo26n.pt` nếu có);
DeepLabCut (SuperAnimal-Quadruped) vào thư mục cài deeplabcut. Sau đó chạy offline.

## Chạy demo (từ thư mục `AI_park`, env `dlc`)
**1. Bỏ video vào `data/`, tên = tình huống** (+ số hoặc ghi chú phía sau):

| Tên video bắt đầu bằng | Chạy |
|---|---|
| `demnguoi` | đếm người / vật |
| `dongnguoi` | đếm + đám đông |
| `bieutinh` | đám đông + băng rôn |
| `bangron` | băng rôn |
| `lanchiem` | lấn chiếm |
| `phongue` | phóng uế |

Ví dụ: `data/bieutinh1.mp4`, `data/lanchiem2.mp4`, `data/phongue_cho_den.mp4`. Tên chức năng gốc (`lan_chiem1.mp4`) vẫn được.

**2. Mở menu, khách muốn xem gì chọn số đó** (chạy xong quay lại menu; model nạp 1 lần lúc mở):
```bash
cd ~/PycharmProjects/AI_park && conda activate dlc
python demo/run_demo.py
```
`2` = chạy mục 2, `1 3` = chạy mục 1 rồi 3, `a` = chạy hết, `q` = thoát. Đang chạy muốn dừng: bấm `q` trên cửa sổ video hoặc Ctrl+C
-> về menu. Video tên sai: in "bỏ qua" kèm lý do, không có trong menu.

**Camera thật** (linh hoạt): thêm `CAM_X=rtsp://...` vào `people_count/.env` (lần đầu `cp -n people_count/.env.example
people_count/.env`), rồi ghi vào `data/camera.txt` mỗi dòng `CAM_X=<tình huống>[,<tình huống>]` hoặc `CAM_X=all` -> camera hiện trong menu.

Lệnh khác: `python demo/run_demo.py data/bieutinh1.mp4 CAM_X=lanchiem --show` (chạy thẳng, không menu);
`python demo/run_demo.py --demo` (chạy hết `data/` một lượt). `data/` không vào git: video của khách không lên GitHub.

| Chức năng | Làm gì | Ngưỡng demo (tuỳ chọn) |
|---|---|---|
| `count` | đếm người / chó / mèo trong khung, báo cáo trung vị / max | `--report-sec 30` |
| `crowd` | số người >= ngưỡng, quá nửa thời gian `--hold` giây -> cảnh báo | `--min-people 5 --hold 10` |
| `bang_ron` | băng rôn đang được người cầm (băng rôn treo cao không tính), quá nửa thời gian `--hold` giây -> cảnh báo | `--banner-conf 0.3 --hold 10` |
| `lan_chiem` | xe máy / ô tô / ghế / bàn / ô dù / xe đẩy / gánh / sạp **đứng yên trùng chỗ** quá nửa thời gian `--dwell` giây -> cảnh báo kèm danh sách vật. Lớp + nguồn (COCO / YOLOE) lấy từ `dataset/classes.yaml`. **Cần camera đứng yên** | `--dwell 60` |
| `phong_ue` | **động vật** (không phân biệt loài) **ngồi xổm**: hông (chân sau - gốc đuôi) < `--squat-ratio` x thân (cổ - gốc đuôi), keypoint DeepLabCut; quá nửa thời gian `--squat-hold` giây -> cảnh báo. Video vẽ C / D / P, thanh hông + vạch ngưỡng. Không bắt chó đực nhấc chân. Ngưỡng mới đo trên 1 clip | `--squat-ratio 0.56 --squat-hold 3 --dog-conf 0.5` |

Ra: `demo/out/<giờ chạy>/<nguồn>.avi` (VLC) + `events.jsonl` + `bang_chung/*.jpg` (ảnh mỗi lần `alert_start`, đường dẫn trong
`evidence.snapshot`); cảnh báo in ngay lúc chạy; cuối cùng in bảng tóm tắt. `--show`: xem trực tiếp lúc chạy.
Ctrl+C: dừng gọn nguồn đang chạy (file vẫn xem được), bỏ các nguồn còn lại. `--fps 5`: chạy nhanh hơn, video giật hơn.

## Giới hạn (PoC)
- Vùng = cả khung hình. Các nguồn chạy lần lượt, không song song.
- `bang_ron`: thử khả thi 2026-10-21 mới có 1 clip biểu tình (quay điện thoại, cận) + 2 clip không: chưa kiểm góc camera cao, xa.
- `phong_ue`: ngưỡng 0.56 đo trên 1 clip (camera chuông cửa, chó ~100 px); 0.58 đã báo nhầm lúc chó cúi ngửi -> biên hẹp, phải kiểm nhiều clip.
  Detector DeepLabCut ~10 frame/s trên robo: video 15 fps chạy chậm hơn thời gian thực.
- Test: `python -m pytest -q demo/tests` (không cần model, GPU).