# CONTEXT — PoC / demo (demo/, dataset/)

Cập nhật: 2026-10-23. Cách làm việc: `docs/CACH_LAM_VIEC.md`. Kiến trúc hệ thật (DeepStream, nhiều camera): `docs/KIEN_TRUC.md`.
Engine DeepStream + module đếm / đám đông: `people_count/CONTEXT.md`. Phát hiện camera bị che: `camera_tamper/CONTEXT.md`.

# PHẦN 1 — ĐỌC HẾT MỖI PHIÊN

## Hệ là gì (một câu)
Công cụ PoC: đưa video / camera vào, **mỗi nguồn gán chức năng riêng**, ra video đã vẽ + file event + ảnh bằng chứng, để khách
thấy **kết quả khả thi**. Không phải sản phẩm.

## Hướng (Thuận chốt)
- **PoC trước, tối ưu sau.** Khách chỉ quan tâm kết quả; DeepStream, nhiều camera, tối ưu tài nguyên làm SAU khi khách thấy giá trị.
- **Perception trước, logic sau**: chưa chắc nhận ra được vật thì chưa viết logic. Mỗi lớp mới: thử nhận diện trên clip trước.
- Hệ **không đoán ý định** từ hình: chỉ có số lượng / loại vật / dáng / thời gian -> logic suy ra tình huống.
- **Không chỉnh model / ngưỡng theo clip không đại diện** (quay tay, YouTube): clip chỉ để kiểm hệ CHẠY và cơ chế ĐÚNG; độ chính xác
  đo trên camera thật của khách. Logic sai thì sửa ngay (bằng test dữ liệu giả), không chờ camera.
- Clip "đúng chuẩn tình huống" rất khó tìm: demo bằng clip có đủ yếu tố, **nói trước giới hạn của từng clip** với khách
  (2026-10-23). Sau demo mới tìm / dựng clip chuẩn hơn.
- **Mọi thứ nằm TRONG project**: clip thử ở `AI_park/data/<bài>/` (`.gitignore` chặn `/data/`), không tạo thư mục ở home.
- Một **model chung** cho profile overview là đích (dataset làm song song); PoC tạm dùng nhiều model.
- CONTEXT không gửi kèm mỗi gói; Claude báo khi thông tin lệch / chồng chéo, Thuận đồng ý mới cập nhật.

## Đang ở đâu (2026-10-23)
`demo/run_demo.py`. **Không đối số = MENU**: liệt kê video trong `data/` + camera trong `data/camera.txt`, chọn số nào chạy số đó,
xong quay lại menu, model nạp 1 lần. **Video: tên = TÌNH HUỐNG** (`bieutinh1`, `dongnguoi1`, `lanchiem1`, `phongue1`, `demnguoi`,
`bangron`; tên chức năng gốc vẫn được); **camera: linh hoạt** (`CAM_X=bieutinh,phongue`, `CAM_X=all`). Thuận chốt 2026-10-23:

| Chức năng | Perception | Luật | Trạng thái demo |
|---|---|---|---|
| `count` | YOLO26n (COCO) | đếm hiện diện, báo cáo trung vị / max (`people_count/counter.py`) | chạy, park1 OK |
| `crowd` | YOLO26n | người >= `--min-people` quá nửa `--hold` s; hết chỉ xét dữ liệu sau lúc báo (`people_count/crowd.py`) | chạy, park1 OK |
| `bang_ron` | YOLOE gõ chữ | băng rôn GẦN NGƯỜI quá nửa `--hold` s (`demo/bang_ron.py`) | chạy, mới 1 clip biểu tình |
| `lan_chiem` | COCO + YOLOE theo `dataset/classes.yaml` | vật cùng lớp trùng chỗ (IoU 0.5) quá nửa mẫu (5 s/mẫu) trong `--dwell` s -> báo; **hết khi không thấy lại lần nào trong dwell** (`demo/lan_chiem.py`) | logic xong; clip có là quay tay (chỉ báo đoạn máy đứng yên, đúng giới hạn) |
| `phong_ue` | DeepLabCut SuperAnimal-Quadruped (detector + 39 keypoint), từng frame (`demo/dlc_pose.py`) | động vật **ngồi xổm**: hông / thân < `--squat-ratio` 0.56 quá nửa `--squat-hold` 3 s (`demo/phong_ue.py`) | chạy đúng khi con vật đủ to, nhìn ngang (clip 2: báo 18.9-22.5 s, đáp án 16-22 s); con vật nhỏ không bắt được |

- Mọi `alert_start` kèm **ảnh bằng chứng** `demo/out/<giờ>/bang_chung/*.jpg` (frame đã vẽ), đường dẫn ở `evidence.snapshot`.
- `data/` CHỈ chứa video (tên theo tình huống) + `camera.txt`, không thư mục con co/khong (Thuận: demo chỉ cần tình huống hệ
  chắc chắn nhận ra). `--demo`: chạy hết một lượt. Nhiều chức năng trên 1 nguồn chạy chung không xung đột (test M19, M20).
- `--show`: xem trực tiếp lúc chạy (`q` sang nguồn tiếp). Overlay `phong_ue`: C (cổ), D (gốc đuôi), P (chân sau), thanh hông
  đứng + vạch ngưỡng (chân qua vạch = đứng / chưa tới = ngồi xổm), dòng trên cùng % ngồi xổm trong cửa sổ.
- `dataset/`: `classes.yaml` (13 lớp, danh sách DUY NHẤT cho model chung) + `nhan_dien.py` (thử nhận diện + nhãn tự động YOLO cho CVAT).
- Test (không cần GPU / model): `demo/tests` 21, `dataset/tests` 4, `people_count/tests` 64, `camera_tamper/tests` 57.
- Env **`dlc`** chạy được MỌI chức năng (thay `yoloe`; `yoloe` vẫn chạy được trừ `phong_ue`).
- Git: nhánh `poc-overlay` (chưa gộp main). Đã push: lan_chiem tách ngưỡng (`21cb4ae`), phong_ue v13. **v14b (nhãn animal, ảnh bằng
  chứng, overlay mới, `/data/`) kiểm đã commit chưa** (`git status --short`).

## Phiên tới làm gì
1. `data/` phẳng, tên theo tình huống (v19). Commit nếu chưa; mở PR `poc-overlay -> main`, Thuận tự duyệt + gộp.
2. Thuận chọn: luật **biểu tình** (đông người + băng rôn cùng lúc) / **lan_chiem trên camera văn phòng** (kê ghế, dựng xe vài phút:
   camera cố định thật, gần như miễn phí) / quy trình sửa nhãn **CVAT** cho model chung.

## Quyết định và lý do
| Quyết định | Lý do |
|---|---|
| PoC bằng Python thuần (ultralytics, DeepLabCut), không DeepStream | khách chỉ cần thấy kết quả; logic tách khỏi perception nên dùng lại nguyên khi lên DeepStream |
| Demo theo `nguồn=chức năng` | thực tế mỗi camera chỉ làm chức năng của khu vực nó |
| Video tên = TÌNH HUỐNG tiếng Việt không dấu (bắt buộc); camera linh hoạt (`=tình huống` / `=all`); MENU chọn tình huống (Thuận 2026-10-23) | tên mô tả ("via_he") hiểu nhiều cách; khách muốn xem gì demo nấy, không chạy hết rồi ngồi đợi; model nạp 1 lần; camera ghép tuỳ ý chức năng (production quy định chặt hơn) |
| Một nguồn sai thì không chạy gì, báo đủ lỗi một lượt | gõ nhầm đường dẫn trước buổi demo phải biết ngay, không chạy xong mới thấy thiếu nguồn |
| "Quá nửa thời gian trong cửa sổ" cho crowd / bang_ron / lan_chiem / phong_ue | model nhận chập chờn (người 8-10-7; băng rôn 38% thời gian, liên tục dài nhất 4 s); luật "liên tục" không bao giờ báo |
| Hết cảnh báo chỉ xét dữ liệu SAU lúc báo (crowd, bang_ron, phong_ue) | demo park1: báo rồi tắt sau 3.7 s vì cửa sổ "hết" còn số thấp trước lúc báo (test D10) |
| lan_chiem: vào = quá nửa mẫu, **ra = không thấy lại lần nào trong dwell** | 1 ghế nhận ~50% số lần -> 174 event / 15 phút (giả lập 2026-10-22); chỉ áp D10 còn 43; tách ngưỡng -> 1 (test M10). Giá: dọn đi thật thì dwell s sau mới hết |
| Băng rôn phải "gần người" | đường phố VN đầy băng rôn treo cố định; clip phố Đà Nẵng: 0% báo ở ngưỡng 0.3 |
| Lấn chiếm không dùng tracker | dwell tính bằng phút, mã tracker đổi khi bị che |
| Lấn chiếm báo theo khung kèm danh sách vật, đổi danh sách không báo lại | 1 quán cóc không thành mười mấy tin |
| phong_ue: yêu cầu = **động vật** (không phân biệt loài, tiểu / đại tiện) + ảnh bằng chứng; BA tư vấn khách | khách cần biết có phóng uế + bằng chứng, không cần loài; "phóng uế" vốn gồm cả tiểu -> chó cái ngồi tiểu không còn là báo nhầm |
| phong_ue: keypoint + định nghĩa TỰ VIẾT (hông / thân), không train | vận hành chỉ suy luận; hình học dựa trên cơ chế (hông hạ khi ngồi xổm); test bằng dữ liệu giả |
| Chiều cao hông tính bằng PIXEL, chia chiều dài thân (cổ -> gốc đuôi) | toạ độ tỉ lệ 0..1 khác thước x / y khi khung không vuông; chia thân để to / nhỏ, gần / xa so được |
| Giữ detector của DeepLabCut (Faster R-CNN), chưa thay YOLO | yêu cầu "mọi động vật" -> không cần biết loài; YOLO26n cũng không thấy con vật nhỏ (đo 2026-10-22); thay là đổi crop của model pose -> phải đo lại. Lý do còn lại (tốc độ) để sau demo |
| Không `video_adapt` của DeepLabCut | tự huấn luyện: độ tin gốc đuôi 0.7 -> 0.9 nhưng tách không đổi (0.80 -> 0.79), detector học lệch bắn vào màn chữ + người (không có ảnh âm) |
| `demo/dlc_pose.py` là file DUY NHẤT import deeplabcut | đổi thư viện keypoint chỉ sửa 1 file; test không cần DeepLabCut |
| `Obj` thêm trường `kp` (mặc định None) | hợp đồng perception -> logic chỉ THÊM, mọi chỗ tạo Obj 4 trường giữ nguyên (64 test people_count khoá) |
| Ảnh bằng chứng cho MỌI alert_start, chụp frame đã vẽ | `evidence.snapshot` có sẵn trong envelope cho mọi chức năng; ảnh đã vẽ thấy luôn lý do báo |
| Overlay chỉ dùng 3 màu sẵn có (trắng / vàng / đỏ) | không đổi hợp đồng vẽ của pipeline.py (DeepStream) |
| Một file `dataset/classes.yaml` cho demo, gán nhãn, train | không lệch danh sách lớp; thứ tự = id nhãn, chỉ thêm vào cuối |
| Gán nhãn ĐỦ mọi lớp trong mỗi ảnh | ảnh chỉ khoanh băng rôn, bỏ trống người -> model học "người là nền" |
| Overlay count chỉ hiện số HIỆN DIỆN | người xem đếm khung trên hình để đối chiếu; trung vị nằm trong file event |

## Đã thử và bỏ
- `video_adapt` của DeepLabCut (lý do ở bảng trên).
- Chạy DeepLabCut ra file `.h5` rồi mới chạy luật (2 bước): demo phải một lệnh, liền mạch -> gọi DeepLabCut từng frame.
- Vẽ đường chéo gốc đuôi -> chân sau: không khớp phép đo (đo theo chiều ĐỨNG), nhìn không hiểu -> thanh đứng + vạch ngưỡng.
- Thư mục dữ liệu ở home (`~/phong_ue`, `~/bang_ron`, `~/via_he`): bị xoá nhầm, mất clip -> `AI_park/data/`.

## Việc mở
- **phong_ue, con vật nhỏ**: clip 1 (~75x35 px, mắt cá, góc xa) và clip 3 (~50x55 px) không bắt được. Cơ chế: model co ảnh về 640 px
  -> con vật còn ~35 px. Hướng: cắt vùng quan tâm (ROI) trước model / chia ô (tiling) / **yêu cầu lắp đặt** (con vật trong vùng đủ to,
  viết theo px/m ở điểm xa nhất để đội lắp tự kiểm bằng thước 1 m). Cỡ tối thiểu chưa đo (Thuận: để khi có camera thật).
- phong_ue, con vật **quay mặt vào camera**: thân co ngắn trên ảnh -> hông / thân sai; luật 2D chỉ đúng khi nhìn ngang / chéo.
- phong_ue: chó đực nhấc chân đi tiểu không bắt được (hông không hạ); **mèo** ngồi nghỉ hông vốn thấp -> có thể báo nhầm, chưa có clip;
  ngưỡng 0.56 / 3 s mới đo trên 1 clip; `--dog-conf` 0.5 chưa đo (clip 2: khung 0.99).
- **Biểu tình** = đám đông + băng rôn cùng lúc: chưa có luật kết hợp (hiện 2 cảnh báo riêng).
- Băng rôn mới 1 clip "co" (điện thoại, cận); trong clip vỉa hè YOLOE bắn "banner" 14% (nghi biển hiệu) -> dataset cần mẫu biển hiệu.
- Vùng PoC = cả khung; vùng thật (vỉa hè, khu cỏ) chưa khai được trong demo.
- Lấn chiếm: hai vật đỗ chồng chỗ tính là một; giờ được phép bán hàng chưa có.
- `tamper` chưa nối vào demo. Chữ trên video không dấu (OpenCV không vẽ được dấu).
- Model chung: tiêu chí trước khi dùng = băng rôn tốt hơn YOLOE **và** người không kém YOLO26n gốc; tách bộ đánh giá THEO CLIP.

---

# PHẦN 2 — TRA KHI CẦN

## Lệnh
```bash
cd ~/PycharmProjects/AI_park && conda activate dlc
python demo/run_demo.py                                # MENU: chọn tình huống nào chạy tình huống đó
python demo/run_demo.py data/bieutinh1.mp4 data/lanchiem1.mp4 --dwell 30 --show   # chạy thẳng: video theo tên tình huống
python demo/run_demo.py CAM_01_URL=dongnguoi --sec 120 --show          # camera: tên biến .env = tình huống | all
python dataset/nhan_dien.py data/lanchiem1.mp4 --out dataset/out/lanchiem       # thử nhận diện + nhãn tự động
python -m pytest -q demo/tests && python -m pytest -q dataset/tests && python -m pytest -q people_count/tests
```
- Ra: `demo/out/<giờ>/<nguồn>.avi` + `events.jsonl` + `bang_chung/*.jpg`; `dataset/out/<tên>/ve/*.avi`, `thong_ke.csv`, `yolo/`.
- Xem video: `vlc <file>`. Video mượt = xét mọi frame (`--fps 0`, mặc định); `--fps 2` nhanh nhưng giật.
- Đóng gói gửi đi: `tools/pack.sh` (theo git, chặn `.env` và giá trị bí mật). Không nén tay.

## Môi trường `dlc` (robo)
```bash
conda create -y -n dlc python=3.10 && conda activate dlc
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
pip install --pre "deeplabcut"                    # 3.0.2; --pre kéo vài thư viện bản rc, chưa thấy lỗi
pip install ultralytics "git+https://github.com/ultralytics/CLIP.git" python-dotenv pytest
```
- Claude kiểm 2026-10-22 (Python 3.10, CPU): deeplabcut 3.0.2 + ultralytics 8.4.183 chạy chung, đủ test, `cv2` có GUI (QT5).
- Model tự tải lần đầu: YOLO / YOLOE vào `demo/models/`; DeepLabCut vào thư mục cài deeplabcut (HuggingFace). Sau đó offline.
- Không nâng ultralytics trong `deepcv` (Detect2026 đang dùng). GPU robo 6 GB: train trong DeepLabCut (vd `video_adapt`) cần
  `video_adapt_batch_size=1` (batch 8 hết VRAM; batch 1 dùng ~3.6 GB).

## Số đo (robo trừ khi ghi khác; chưa có đáp án = chưa phải độ chính xác)
| Số đo | Giá trị | Nguồn |
|---|---|---|
| Băng rôn, 1 clip "co" + 2 clip "khong" | ngưỡng 0.3: bắt 1/1, nhầm 0/2; % thời gian co 38% / khong 2%; liên tục dài nhất 4.0 s | `thu_bang_ron.py`, 2026-10-21 |
| Băng rôn, phố Đà Nẵng nhiều băng rôn treo | 0% ở ngưỡng 0.2 và 0.3 | như trên |
| Demo `park1=count,crowd` (min 5, hold 10) | 7 cặp, đóng cuối 355.0 s, 23 báo cáo đếm | `run_demo.py`, 2026-10-21 |
| Vỉa hè `via_he.mp4` (quay tay), % thời gian thấy COCO / YOLOE | xe máy 42/44, ô tô 23/25, ghế 15/15, ô dù 9/14, bàn 4/7, xe đạp 3/10; chỉ YOLOE: xe đẩy 9, sạp 8, gánh 2 | `nhan_dien.py`, 2026-10-21 |
| lan_chiem trên `via_he` (dwell 30, code cũ) | 3 cặp: ghế 35-55 s, bàn 80-85 s, ô tô 195-210 s | `run_demo.py`, 2026-10-22 |
| lan_chiem giả lập: ghế nhận 50% / 15 phút | code cũ 174 event, chỉ D10 43, tách ngưỡng **1** | test M10, 2026-10-22 |
| phong_ue clip 2 (`dogpoop2`, 1280x720 15 fps, chuông cửa, chó đen ~180x100 px), hông / thân | đứng 0.64 / phóng uế 0.54 (trung vị, cùng cỡ thân ~100 px); tách theo frame 0.80; góc gối sau 0.55 (không dùng) | `.h5` DeepLabCut, đáp án 16-22 s, 2026-10-22 |
| phong_ue clip 2, luật đầy đủ (0.56, 3 s) | báo 18.9 s, hết 22.5 s; không báo nhầm | `run_demo.py` robo + phát lại `.h5`, khớp từng số |
| phong_ue clip 2, ngưỡng 0.58 (mô phỏng tay, cửa sổ chỉ tính frame đo được) | báo nhầm 10.5 s lúc chó cúi ngửi -> biên an toàn HẸP | 2026-10-22 |
| DeepLabCut tốc độ, 1280x720 | detector ~10.8 frame/s (nút cổ chai), pose ~67 frame/s | log robo, 2026-10-22 |
| Con vật nhỏ: YOLO26n thấy chó / mèo (3 frame/s) | clip 1 (~75x35 px*): 0/157 ở 640, **30/157 ở 1280**; clip 3 (~50x55 px*): 13/52 ở 640, 32/52 ở 1280; clip 2: 18/120 ở 640 (DeepLabCut gần như mọi frame) | Claude, CPU, 2026-10-22. *ước lượng bằng mắt, không phải model đo |

## Cấu trúc
| File | Vai trò |
|---|---|
| `demo/run_demo.py` | đọc `nguồn=chức năng`, chỉ nạp model / thư viện cần, đưa vào module logic, vẽ bằng `draw()`, ghi avi + jsonl + ảnh bằng chứng, bảng tóm tắt |
| `demo/dlc_pose.py` | DeepLabCut từng frame (lọc khung yếu / suy biến TRƯỚC khi chấm keypoint) + `to_animal_objs` -> `Obj("animal", kp=...)` |
| `demo/bang_ron.py`, `lan_chiem.py`, `phong_ue.py` | module logic PoC (không IO, thời gian truyền vào), khuôn `update / flush / draw` như `people_count/counter.py`, `crowd.py` |
| `people_count/meta.py` | hợp đồng perception -> logic: `FrameMeta`, `Obj(cls, conf, bbox, track_id, kp=None)`, `Line`, `Text` |
| `dataset/classes.yaml` | 13 lớp: person, dog, cat, banner, motorcycle, bicycle, car, umbrella, chair, table, vendor_cart, shoulder_pole, stall |
| `dataset/nhan_dien.py` | thử nhận diện (bảng COCO vs YOLOE) + nhãn tự động YOLO cho CVAT |
| `demo/tools/thu_bang_ron.py` | thử khả thi băng rôn theo clip (thư mục `co/`, `khong/`); chuyển từ people_count/tools 2026-10-23 |
| `data/` | video demo (tên = tình huống) + `camera.txt`; trong project, không vào git; menu / `--demo` đọc thẳng đây |
