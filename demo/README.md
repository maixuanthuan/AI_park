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

## Chạy nhanh nhất: `--demo` (không sửa lệnh)
Mọi clip thử nằm trong **`data/<chức năng>/`** (thư mục con tuỳ ý, vd `co/` có phóng uế, `khong/` để xem có báo nhầm không).
Tên thư mục = chức năng chạy; `+` = nhiều chức năng. Camera thật: ghi TÊN biến `.env` vào `camera.txt` của thư mục:
```
data/
├── count+crowd/      # mọi video ở đây chạy count + crowd
│   └── camera.txt    # tuỳ chọn: mỗi dòng 1 TÊN biến trong people_count/.env, vd CAM_CONG_CHINH
├── bang_ron/co/, khong/
├── lan_chiem/
└── phong_ue/co/, khong/
```
```bash
cd ~/PycharmProjects/AI_park && conda activate dlc
python demo/run_demo.py --demo --show            # thêm --sec 300 nếu có camera (mặc định camera chạy tới Ctrl+C)
```
Thư mục trống: bỏ qua. Tên thư mục sai, biến camera chưa khai -> báo rõ, không chạy. Hai video trùng tên file -> đổi tên một cái.
Link camera (có mật khẩu) chỉ nằm trong `people_count/.env` (`cp -n people_count/.env.example people_count/.env` lần đầu).
`data/` không vào git: video của khách không bao giờ lên GitHub.

## Chạy (từ thư mục `AI_park`)
Clip thử để trong `data/<bài>/` (vd `data/phong_ue/co/`, `data/phong_ue/khong/`): trong project, `.gitignore` chặn, không vào git.
```bash
conda activate dlc
python demo/run_demo.py people_count/videos/park1.mp4=count,crowd data/bang_ron/co/bieutinh.mp4=bang_ron data/lan_chiem/via_he.mp4=lan_chiem
python demo/run_demo.py data/phong_ue/co/dogpoop.mp4=phong_ue
python demo/run_demo.py CAM_01_URL=count,crowd --sec 120 --show      # camera: tên biến trong people_count/.env
python demo/run_demo.py -h                                            # mọi tuỳ chọn
```
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