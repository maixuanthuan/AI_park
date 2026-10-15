# CONTEXT — Bài toán: Phát hiện camera bị che / phá (camera_tamper)

> Tài liệu bối cảnh để mở phiên làm việc mới. Đọc hết trước khi đề xuất gì.
> Cập nhật lần cuối: 2026-10-15.

## 1. Bối cảnh dự án

- Dự án **AI cho camera công viên** (tên thư mục dự kiến: `AI_park`). Gồm 7 bài toán:
  1. Nhận diện & bóc tách biển số xe — **đã xong** (ngoài phạm vi tài liệu này)
  2. Nhận diện khuôn mặt — **đã xong**
  3. **Phát hiện camera bị che phủ — tài liệu này**
  4. Nhận diện vật nuôi phóng uế
  5. Đo đếm người, vật nuôi trong khuôn viên
  6. Nhận diện đám đông, biểu tình / gây mất trật tự
  7. Phát hiện lấn chiếm vỉa hè
- Giai đoạn hiện tại: **chuẩn bị trước cho buổi đánh giá**, chưa có camera công viên. Test bằng camera văn phòng + video công viên tải từ YouTube.
- Hệ thống AI chỉ **phát hiện & phát event**. Người quản lý xem **link camera gốc** (đầu ghi / VMS); backend nhận event để thông báo / hiển thị. `--show` chỉ để dev debug, không bao giờ lên dashboard.

## 2. Cách làm việc đã thống nhất

- Giao tiếp tiếng Việt, thẳng, thoải mái. **Thảo luận, chốt rồi mới code** — không lao vào code.
- Giải thích bằng ngôn ngữ đời thường, có ví dụ; **mỗi lần 1 việc**, nói rõ làm để làm gì trước khi đưa lệnh.
- Giải pháp **tối giản, tiết kiệm tài nguyên tối đa**, miễn đúng chức năng.
- Claude **không chạy / deploy** trên máy của Thuan. Giao code dạng **zip + lệnh** để Thuan tự chạy:
  ```bash
  cd ~/PycharmProjects/camera_tamper
  unzip -o ~/Downloads/camera_tamper.zip && rm ~/Downloads/camera_tamper.zip
  ```
  (zip chứa file ở gốc, không có thư mục con; không chứa `.env`.)
- Thông tin nhạy cảm (link camera, user/pass) để trong `.env`, không vào code / git. Log tự che `user:pass`.
- Chạy ngon trên máy local (robo) trước, sau đó mới lên server. Cuối cùng đóng **Docker**, chạy 1 lệnh.
- Phân vai: **devops (sysadmin)** chỉ cung cấp link camera + xem log; **dev** chịu trách nhiệm logic. Sai link → lỗi devops; đúng link sai logic → lỗi dev.

## 3. Môi trường

- Máy dev **robo**: Ubuntu, GPU **RTX A3000 Laptop** (driver 535), conda env `deepcv`.
  - ffmpeg hệ thống `/bin/ffmpeg` **4.4.2** (Ubuntu 22.04), có `-hwaccel cuda`, `h264_cuvid`, `hevc_cuvid`.
  - Cảnh báo `libncursesw.so.6: no version information available` khi gọi ffmpeg trong env deepcv: **vô hại**, bỏ qua.
- Server: **V100**, Ubuntu 22.04, Docker Compose. Lưu ý Docker cần `NVIDIA_DRIVER_CAPABILITIES=compute,utility,video` (thiếu `video` → không có NVDEC).
- Camera test văn phòng (`cam_01`): **HEVC 1280x960 @30fps**, có **OSD đồng hồ** ở dải trên cùng.

## 4. Cấu trúc project `~/PycharmProjects/camera_tamper`

| File | Vai trò |
|---|---|
| `tamper_detector.py` | Lõi detector, không IO: `update(frame, ts)` → event \| None; `update_no_frame(ts)`; `status()` |
| `frame_source.py` | `FFmpegSource`: đọc RTSP / file bằng ffmpeg subprocess, GPU (NVDEC) hoặc CPU; ffprobe tự dò codec + kích thước; watchdog; drain stderr; che user:pass trong log |
| `run_tamper.py` | Runner đa camera (1 thread / camera), `--show`, `--log-level`, tổng kết khi kết thúc |
| `config.yaml` | Văn phòng / demo: `hold_sec: 3`, cam_01 có `osd_mask` |
| `config.park.yaml` | Công viên / test video: `hold_sec: 10`, camera là file trong `videos/` |
| `.env` / `.env.example` | `CAM_01_URL=rtsp://...` |
| `CONTEXT.md` | Tài liệu này |

Chạy:
```bash
python run_tamper.py --log-level DEBUG --show 2>&1 | tee test_$(date +%H%M).log      # camera thật
python run_tamper.py --config config.park.yaml 2>&1 | tee park_$(date +%H%M).log       # video công viên
```

## 5. Thuật toán (không dùng model, chỉ opencv + numpy, CPU)

**Ý tưởng:** như người bảo vệ — nhớ cảnh bình thường (ảnh gốc), rồi liên tục so sánh.

- **Tiền xử lý**: ffmpeg giảm còn **2 fps** ngay trên GPU → Python resize về rộng 320px, xám, GaussianBlur 3x3.
- **Lưới 8x8**, mỗi ô tính **độ chi tiết** = variance của Laplacian.
  - Ô có chi tiết trong ảnh gốc (`>= cell_tex_min`) mới được theo dõi; ô trơn (tường, trời, cỏ) bị bỏ qua (xám trên `--show`).
  - Ô "mất chi tiết" nếu var < ref × `cell_drop_ratio`. **`obstruct`** = tỉ lệ ô mất chi tiết / ô được theo dõi.
  - Vì sao hiệu quả với **mọi vật che** (bất kể màu): vật sát ống kính luôn **nhòe** (ngoài tầm lấy nét) → mất chi tiết.
- **`edge_keep`**: tỉ lệ cạnh Canny của ảnh gốc còn khớp với frame hiện tại (dilate 5x5). Bắt **cảnh bị thay đổi** (xoay camera, vật rõ nét đặt trước camera).
  - Ảnh gốc là trung bình theo thời gian → cạnh lá cây đung đưa bị nhòe, cạnh vật cố định (ghế, cột, khung tập) được giữ → tự ưu tiên "mỏ neo" cố định.
- **`mean`**: quá tối (`< dark_thr`) / quá chói (`> bright_thr`) → obstructed.
- **frozen**: hash md5 frame (đã bỏ vùng OSD) giống hệt liên tục `frozen_sec`.
- **no_signal**: không mở được / mất stream.

**Các loại sự cố (reason):** `obstructed`, `scene_changed` (trước đây tên `moved`, đã đổi cho đúng nghĩa), `frozen`, `no_signal`, `baseline_invalid`.

**Chống báo nhầm:**
- **Đếm thời gian chung cho mọi loại bất thường** (debounce), đủ `hold_sec` mới báo; báo loại xuất hiện nhiều nhất. Lý do: khi che, obstruct dao động quanh ngưỡng, loại nhảy obstructed ↔ moved làm reset bộ đếm → trước đây không bao giờ báo.
- **Hysteresis**: vào báo động khi obstruct ≥ 0.6 hoặc edge_keep < 0.35; chỉ thoát khi obstruct < 0.35 **và** edge_keep > 0.6.
- Hết báo động sau `clear_sec` (5s) bình thường liên tục.

**Ảnh gốc:**
- **Warmup** 10s lúc khởi động (chỉ học, không báo). Log / `--show` hiện `state=warmup`.
- **Lớp 1 (chỉ chạy lúc warmup)**: frame phải có ≥ `baseline_min_cells` (3) ô có chi tiết và ≥ `min_ref_edges` (300) cạnh mới được học. Không đạt → chờ, báo `baseline_invalid`, khi cảnh hợp lệ thì tự học và báo hết. Giải quyết tình huống **"che camera trước rồi mới bật hệ thống"** (khách hay test). Lúc vận hành bình thường **không chạy** lớp này (tránh báo nhầm với cảnh trơn: đêm tối, hồ nước, sương). `baseline_min_cells: 0` để tắt cho camera vốn nhìn cảnh trơn.
- Sau warmup, ảnh gốc **cập nhật rất chậm (EMA `ref_alpha` 0.02)**, **chỉ khi bình thường** → thích nghi ánh sáng, không học nhầm cảnh bị che.
- `scene_changed` kéo dài `relearn_sec` (600s) → chấp nhận cảnh mới, phát event `relearn`.

**`osd_mask`** (theo từng camera, tỉ lệ 0..1 `[x1,y1,x2,y2]`): bỏ vùng chữ đồng hồ khỏi hash / độ chi tiết / cạnh.
**BẮT BUỘC với camera có OSD**: giả lập cho thấy không có mask thì chữ đồng hồ trên nền vật che bị coi là "chi tiết" → học nhầm cảnh bị che, không báo gì (cả baseline_invalid lẫn frozen).

**Event (JSON, in ra log WARNING, ảnh JPG lưu `out_dir`):**
```json
{"event": "tamper_start|tamper_end|relearn", "reason": "obstructed", "prev": null, "ts": 310.5,
 "metrics": {"mean": 175.0, "obstruct": 1.0, "edge_keep": 0.0, "n_tex": 62, "n_tex_abs": 0}, "cam_id": "park1_tamper"}
```

## 6. Hiệu năng — bài học áp dụng cho mọi bài toán sau

- **`cv2.setNumThreads(1)` + `OMP_NUM_THREADS=1`**: thread pool OpenCV mặc định spin-wait ăn ~1000% CPU (10 core) dù việc thật rất nhỏ. Sau sửa: python ~4% / camera.
- **ffmpeg filter: `fps` đặt TRƯỚC `hwdownload`** → chỉ chép 2 frame/s từ GPU về CPU.
- Giải mã GPU (NVDEC) là chip riêng, không ăn CUDA core của YOLO. 1 camera ≈ 1% NVDEC, ffmpeg ~3% CPU.
- `--show` chỉ vẽ khi có frame mới.
- Một camera < 10% của 1 core. Video 1080p 2 tiếng xử lý xong trong ~5.5 phút (~22x realtime).
- Đọc file video: detector dùng **thời gian của video** (frame_idx / fps), tốc độ chạy không ảnh hưởng kết quả.

## 7. Kết quả test

- **Camera văn phòng** (che bằng bìa trắng, người đi qua): ô chuyển đỏ đúng; sau khi sửa debounce + hysteresis, replay log cho kết quả báo đúng lúc che, không báo khi người đi qua.
- **Video công viên `park1`** (2 giờ, 1080p, không có phá hoại): **0 cảnh báo**.
- **`park1_tamper`** (15 phút đầu park1 + 3 sự cố giả bằng ffmpeg drawbox / boxblur): **bắt đủ 3/3**, đúng thời điểm (+10s), đúng loại `obstructed`:
  - che kín (5:00–5:30) → báo 5:10; che 75% (7:30–8:00) → báo 7:40; làm mờ toàn bộ (10:00–10:30) → báo 10:10.
- Cơ chế kết nối lại hoạt động (file chưa tạo xong → no_signal → tự đọc lại khi file sẵn sàng).

## 8. Các quyết định đã chốt (và lý do)

- Đọc camera **chỉ bằng ffmpeg** (bỏ cv2.VideoCapture), giữ cờ `hwaccel: false` làm dự phòng CPU.
- Snapshot ở **độ phân giải gốc** (`out_width: 0`).
- **Không làm `show_fps`** (hiển thị mượt) — `--show` chỉ để debug, 2fps là đủ.
- **Không "ép học" ảnh gốc** sau timeout: học từ cảnh trơn thì vẫn mù → chờ + báo `baseline_invalid` sạch hơn.
- **Lưới 8x8** (cân bằng giữa vùng mù và độ ổn định; 32x32 quá nhỏ, nhiễu).
- `hold_sec`: **3s** cho văn phòng / demo, **10s** cho công viên (camera gắn cao, loại chim đậu / lá rụng).
- Không ghi log ra file trong code: local dùng `tee`, Docker dùng `docker compose logs`.

## 9. Giới hạn đã biết

- Che dưới 60% khung hình → không báo (chủ đích, chỉnh `obstruct_frac`).
- Vật che xuất hiện **dần dần qua nhiều ngày** (cây mọc, bụi bám) → ảnh gốc học theo, không báo (chủ đích, thuộc bảo trì).
- Chưa test: **chạng vạng / chuyển IR**, ban đêm, gió to, mưa, mạng nhện; chưa test rút dây mạng camera thật.
- Mất tín hiệu lâu (vd chiều → tối) → khi có hình lại, ánh sáng khác ảnh gốc, có thể báo nhầm.

## 10. Việc còn lại (backlog, chưa làm)

1. Tổng kết in thêm **biên an toàn**: edge_keep thấp nhất / obstruct cao nhất lúc bình thường.
2. Bỏ dòng log thừa `ffmpeg dừng (code=0):` khi hết video bình thường.
3. **Ảnh "trước"** làm bằng chứng: giữ 1 frame bình thường cuối cùng (không buffer nhiều frame — tốn RAM).
4. Mất tín hiệu quá X phút → học lại ảnh gốc từ đầu.
5. Đóng gói cho devops: `cameras.yaml` (cam_id **UUID** + url, devops quản lý, gitignore), `--check` in bảng OK/LỖI (401 sai mật khẩu, 404 sai đường dẫn, timeout), **Lớp 2: ảnh gốc chuẩn chốt lúc lắp đặt** (lưu file, so khung cạnh khi khởi động), Docker 1 lệnh.
6. Hàm `load_cameras()` để sau này lấy danh sách camera từ **API backend**.
7. Chốt **"hợp đồng" format event** + kênh gửi (Kafka hay HTTP) với team backend.
8. Kiểm tra mở rộng: VRAM mỗi tiến trình ffmpeg (mỗi cái 1 CUDA context ~100–300MB), giới hạn kết nối RTSP của camera / đầu ghi.
9. Về lâu dài: các bài toán dùng chung **1 lớp đọc camera** (giải mã 1 lần, chia frame) thay vì mỗi bài kéo stream riêng.
