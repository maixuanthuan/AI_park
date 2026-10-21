# CONTEXT — PoC / demo (demo/, dataset/)

Cập nhật: 2026-10-21. Cách làm việc: `docs/CACH_LAM_VIEC.md`. Kiến trúc hệ thật (DeepStream, nhiều camera): `docs/KIEN_TRUC.md`.
Engine DeepStream + module đếm / đám đông: `people_count/CONTEXT.md`.

# PHẦN 1 — ĐỌC HẾT MỖI PHIÊN

## Hệ là gì (một câu)
Công cụ PoC: đưa video / camera vào, **mỗi nguồn gán chức năng riêng**, ra video đã vẽ + file event, để khách thấy
**kết quả khả thi**. Không phải sản phẩm.

## Hướng (Thuận chốt 2026-10-21)
- **PoC trước, tối ưu sau.** Khách chỉ quan tâm kết quả; DeepStream, nhiều camera, tối ưu tài nguyên làm SAU khi khách
  thấy giá trị. Tối ưu đã đo được cất lại (people_count/CONTEXT, mục "Tối ưu: đã đo, việc cất lại"), không bỏ.
- PoC chạy **ultralytics Python thuần**, vài camera / video. Kiến trúc vẫn giữ bản tổng quát.
- **Perception trước, logic sau**: chưa chắc nhận ra được vật thì chưa viết logic. Mỗi lớp mới: thử nhận diện trên clip trước.
- Hệ **không đoán ý định** từ hình: chỉ có số lượng / loại vật / thời gian -> logic suy ra tình huống.
- Một **model chung** cho profile overview là đích (làm dataset song song); PoC tạm dùng 2 model (YOLO26n + YOLOE).
- CONTEXT không gửi kèm mỗi gói; Claude báo khi thông tin bắt đầu lệch / chồng chéo, Thuận đồng ý mới cập nhật.

## Đang ở đâu (2026-10-21 cuối ngày)
- `demo/run_demo.py`: 4 chức năng, mỗi nguồn `nguồn=chức năng[,chức năng]` (file video hoặc tên biến camera trong `people_count/.env`):
  | Chức năng | Model | Luật | Trạng thái |
  |---|---|---|---|
  | `count` | YOLO26n (COCO) | đếm hiện diện trong vùng, báo cáo trung vị / max (`people_count/counter.py`) | chạy, demo park1 OK |
  | `crowd` | YOLO26n | người >= `--min-people`, quá nửa thời gian `--hold` s -> báo; hết chỉ xét dữ liệu sau lúc báo (`people_count/crowd.py`) | chạy, demo park1 OK |
  | `bang_ron` | YOLOE gõ chữ | băng rôn GẦN NGƯỜI, quá nửa thời gian `--hold` s (`demo/bang_ron.py`) | chạy, 1 clip biểu tình báo, phố băng rôn treo không báo |
  | `lan_chiem` | COCO + YOLOE theo `dataset/classes.yaml` | logic B: vật cùng lớp trùng chỗ (IoU 0.5), quá nửa số mẫu (5 s/mẫu) trong `--dwell` s; báo theo khung kèm danh sách vật (`demo/lan_chiem.py`) | **code + test xong, CHƯA chạy trên clip thật** |
- `dataset/`: `classes.yaml` (13 lớp, danh sách DUY NHẤT cho model chung, thứ tự = id nhãn) + `nhan_dien.py` (thử nhận diện từng lớp
  COCO vs YOLOE trên clip + gán nhãn tự động chuẩn YOLO mỗi 2 s để sửa trong CVAT).
- Test (không cần GPU / model): `demo/tests` 9, `dataset/tests` 4, `people_count/tests` 64, `camera_tamper/tests` 57.
- Git: nhánh `poc-overlay` (tách từ `bench-tools`, chưa gộp main); đã push tới commit demo/ + dataset/ (v10). **Gói v11
  (`lan_chiem`, `--fps 0`) đã áp + test xanh, kiểm lại đã commit chưa** (`git status --short`).

## Phiên tới làm gì
1. Chạy `lan_chiem` trên clip vỉa hè (lệnh ở PHẦN 2). Clip phải **camera đứng yên**; quay tay thì không báo là đúng giới hạn.
   Xem: xe / ghế đỗ lâu có khung đỏ, xe chạy qua chỉ khung cam. Commit v11 nếu chưa.
2. Thuận chọn tiếp: **phóng uế vật nuôi** (bài cuối nhóm overview) hoặc **quy trình gán nhãn CVAT** cho model chung.
3. Mở PR `poc-overlay -> main` (gồm cả commit `bench-tools`), Thuận tự duyệt + gộp.

## Quyết định và lý do
| Quyết định | Lý do |
|---|---|
| PoC bằng ultralytics, không DeepStream | khách chỉ cần thấy kết quả; logic tách khỏi perception nên dùng lại nguyên khi lên DeepStream |
| Demo theo `nguồn=chức năng` | thực tế mỗi camera chỉ làm chức năng của khu vực nó, không phải 1 nguồn làm tất cả |
| "Quá nửa thời gian trong cửa sổ" cho crowd / bang_ron / lan_chiem | model nhận chập chờn (số người nhảy 8-10-7; băng rôn có 38% thời gian nhưng liên tục dài nhất 4 s); luật "liên tục" không bao giờ báo |
| Hết cảnh báo chỉ xét dữ liệu SAU lúc báo | demo park1: cảnh báo 82.3 -> 86.0 bật tắt 3.7 s vì cửa sổ "hết" còn chứa số thấp trước lúc báo (test D10) |
| Băng rôn phải "gần người" | đường phố VN đầy băng rôn khẩu hiệu treo cố định; clip phố Đà Nẵng: 0% báo ở ngưỡng 0.3 |
| Lấn chiếm không dùng tracker (logic B) | dwell tính bằng phút, mã tracker đổi khi bị che -> không bao giờ đủ thời gian |
| Lấn chiếm báo theo khung kèm danh sách vật, đổi danh sách không báo lại | 1 quán cóc không thành mười mấy tin cho người quản lý |
| Một file `dataset/classes.yaml` cho demo, gán nhãn, train | không lệch danh sách lớp giữa các nơi; thứ tự = id nhãn, chỉ thêm vào cuối |
| Gán nhãn ĐỦ mọi lớp trong mỗi ảnh | ảnh chỉ khoanh băng rôn, bỏ trống người -> model học "người là nền", đếm người kém đi ngầm |
| Môi trường conda riêng `yoloe` | không nâng ultralytics trong `deepcv` (Detect2026 đang dùng) |
| Overlay video chỉ hiện số HIỆN DIỆN (không trung vị) | người xem đếm khung trên hình để đối chiếu; trung vị nằm trong file event |

## Việc mở
- **Băng rôn**: mới 1 clip "co" (điện thoại, cận); chưa thử góc camera cao / xa. Trong clip vỉa hè YOLOE bắn "banner" 14%
  thời gian, nhiều khả năng **biển hiệu cửa hàng** -> khi làm dataset phải có ảnh biển hiệu làm mẫu "không phải băng rôn".
- **Biểu tình** = đám đông + băng rôn cùng lúc: chưa có luật kết hợp (hiện 2 cảnh báo riêng).
- Vùng PoC = cả khung; vùng thật (vỉa hè, tránh ghế đá / bồn cây) chưa có cách khai trong demo.
- Lấn chiếm: tỉ lệ có mặt sát 50% có thể bật tắt; hai vật đỗ chồng chỗ tính là một; giờ được phép bán hàng chưa có.
- `tamper` chưa nối vào demo (chỉ vài dòng khi cần). Chữ trên video không dấu (OpenCV không vẽ được dấu).
- Model chung: tiêu chí trước khi dùng = băng rôn tốt hơn YOLOE **và** người không kém YOLO26n gốc; tách bộ đánh giá THEO CLIP.

---

# PHẦN 2 — TRA KHI CẦN

## Lệnh
```bash
cd ~/PycharmProjects/AI_park && conda activate yoloe
python demo/run_demo.py people_count/videos/park1.mp4=count,crowd ~/bang_ron/co/bieutinh.mp4=bang_ron ~/via_he/via_he.mp4=lan_chiem --dwell 30
python demo/run_demo.py CAM_01_URL=count,crowd --sec 120 --show        # camera: tên biến trong people_count/.env
python dataset/nhan_dien.py ~/via_he --out dataset/out/via_he           # thử nhận diện + nhãn tự động
python people_count/tools/thu_bang_ron.py --data ~/bang_ron --out people_count/out/bang_ron   # thử khả thi băng rôn
conda activate deepcv && python -m pytest -q demo/tests && python -m pytest -q dataset/tests && python -m pytest -q people_count/tests
```
- Ra: `demo/out/<giờ>/<nguồn>.avi` + `events.jsonl`; `dataset/out/<tên>/ve/*.avi`, `thong_ke.csv`, `yolo/` (images, labels, data.yaml).
- Xem video: `vlc <file>` (không mở cả thư mục). Video mượt = xét mọi frame (`--fps 0`, mặc định); `--fps 2` nhanh nhưng giật.
- Đóng gói gửi đi: `tools/pack.sh` (theo git, chặn `.env` và giá trị bí mật). Không nén tay.

## Môi trường `yoloe` (robo)
`conda create -n yoloe python=3.11`; `pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121`;
`pip install ultralytics "git+https://github.com/ultralytics/CLIP.git" python-dotenv`. Model tự tải vào `demo/models/`
(`yoloe-11s-seg.pt` ~27 MB); YOLO26n dùng `people_count/models/yolo26n.pt`. Claude thử trên ultralytics 8.4.183 (CPU).

## Số đo (2026-10-21, robo trừ khi ghi khác)
| Số đo | Giá trị | Nguồn |
|---|---|---|
| Băng rôn, 1 clip "co" (biểu tình, điện thoại, ~60 s) + 2 clip "khong" | ngưỡng 0.3: bắt 1/1, báo nhầm 0/2; % thời gian co 38% / khong 2%; liên tục dài nhất co 4.0 s | `thu_bang_ron.py --fps 30` |
| Băng rôn, phố Đà Nẵng nhiều băng rôn treo (`park1_2min`) | 0% ở ngưỡng 0.2 và 0.3 | như trên |
| Demo `bieutinh=bang_ron` (hold 10) | 2 cặp start/end (10.0-25.9 s, 37.0-47.0 s) | `run_demo.py` |
| Demo `park1=count,crowd` (min 5, hold 10) | 7 cặp, đóng cuối 355.0 s, 23 báo cáo đếm | `run_demo.py` |
| Engine DeepStream `park1` crowd (min 6, hold/clear 10) | 7 cặp, mọi cặp >= 10 s sau sửa D10, đóng cuối 354.8 s | `run_engine.py --debug-out` |
| Vỉa hè `via_he.mp4` (1 clip, 25 fps, mọi frame), % thời gian thấy COCO / YOLOE | xe máy 42/44, ô tô 23/25, ghế 15/15, ô dù 9/14, bàn 4/7, xe đạp 3/10; chỉ YOLOE: xe đẩy 9, sạp 8, gánh 2 (độ tin 0.37); banner 14 (nghi biển hiệu) | `nhan_dien.py --fps 0` |
- Bảng trên chỉ là "model có bắn", CHƯA có đáp án (đúng / sai) -> chưa phải độ chính xác.

## Cấu trúc
| File | Vai trò |
|---|---|
| `demo/run_demo.py` | đọc `nguồn=chức năng`, chạy model cần thiết cho từng nguồn, đưa vào module logic, vẽ bằng `draw()` của module, ghi avi + jsonl, bảng tóm tắt |
| `demo/bang_ron.py`, `demo/lan_chiem.py` | module logic PoC (không IO, thời gian truyền vào), cùng khuôn `update / flush / draw` với `people_count/counter.py`, `crowd.py` |
| `dataset/classes.yaml` | 13 lớp: person, dog, cat, banner, motorcycle, bicycle, car, umbrella, chair, table, vendor_cart, shoulder_pole, stall |
| `dataset/nhan_dien.py` | thử nhận diện (bảng COCO vs YOLOE) + nhãn tự động YOLO cho CVAT |
| `people_count/tools/thu_bang_ron.py` | thử khả thi băng rôn theo clip (thư mục `co/`, `khong/`) |