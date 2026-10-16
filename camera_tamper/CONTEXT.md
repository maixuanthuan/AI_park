# CONTEXT — camera_tamper (phát hiện camera bị che / phá)

Cập nhật: 2026-10-16. Cách làm việc chung: `docs/CACH_LAM_VIEC.md` (Project "AI in town park").

# PHẦN 1 — ĐỌC HẾT MỖI PHIÊN

## Hệ là gì (một câu)
Theo dõi từng camera, phát **event** khi camera bị che / làm mờ / xoay lệch / đóng băng hình / mất tín hiệu.

- Bài toán 3/7 của dự án AI cho camera công viên (repo `AI_park`, GitHub public `maixuanthuan/AI_park`).
- Chỉ dùng opencv + numpy (CPU) cho logic; đọc camera bằng ffmpeg giải mã trên GPU (NVDEC). Không có model.
- **Event** (`tamper_start` / `tamper_end`, JSON) là cho người quản lý, sau này đẩy sang backend.
  **Log vận hành** là cho dev / devops / người tiếp nhận. Hai luồng tách biệt; README mục "Đọc log".
- Người quản lý xem hình bằng link camera gốc (đầu ghi / VMS). `--show` chỉ để dev debug.

## Đang ở đâu
- **Chạy được và đã kiểm** trên camera văn phòng (thật) và video công viên YouTube.
- 57 test (`python -m pytest -q tests`, ~20s, không cần camera / GPU / ffmpeg). Mọi test có 3 dòng mô tả.
- Mỗi chỗ kiểm quan trọng đã được thử "cài lỗi vào thì test có đỏ không" (2026-10-15/16): đều đỏ.
- Config đủ mục, không mặc định ngầm; sai thì không khởi động, in mọi lỗi một lượt (mã thoát 2).
- Bài toán coi như **xong giai đoạn chuẩn bị**; việc còn lại bên dưới không chặn sang bài khác.

## Phiên tới làm gì
Chuyển sang bài toán tiếp theo (đếm người / vật nuôi). Việc còn lại của bài này: xem cuối file.

## Nguyên tắc riêng của bài này
- **Không bao giờ học lại ảnh gốc khi vừa có hình lại / đang bất thường**: kẻ phá có thể rút dây, che, cắm lại.
- Gán nhãn sự cố luôn theo **ngưỡng vào**; vùng đệm (ngưỡng thoát) chỉ để **giữ** trạng thái bất thường.
- Mọi con số trong config có comment: đo gì, ngày nào, cỡ mẫu. Con số chưa đo thì ghi rõ là chưa đo.

---

# PHẦN 2 — TRA KHI CẦN

## Cấu trúc
| File | Vai trò |
|---|---|
| `tamper_detector.py` | Lõi, không IO: `update(frame, ts)` / `update_no_frame(ts)` -> event \| None. Bảng `PARAMS` + `check_cfg` |
| `frame_source.py` | `FFmpegSource`: ffprobe dò codec, ffmpeg (GPU/CPU) qua pipe, watchdog, `last_error`; **không tự in log** |
| `run_tamper.py` | Runner: kiểm config, 1 thread / camera, log vận hành, thống kê định kỳ, `--show`. Mã thoát 0/1/2 |
| `config_check.py` | Kiểm 1 mục config theo bảng: đủ, không mục lạ, đúng kiểu, trong khoảng |
| `config.yaml` / `config.park.yaml` | Văn phòng (`hold_sec` 3) / công viên + test video (`hold_sec` 10) |
| `tests/` | `fake_camera.py` (camera giả), test detector / config / khởi động / log |

## Thuật toán
- ffmpeg giảm còn `sample_fps` (2) **ngay trên GPU** (filter `fps` đứng trước `hwdownload`), Python thu về rộng 320, xám.
- **Lưới 8x8**, mỗi ô độ chi tiết = variance Laplacian. Ô trơn trong ảnh gốc (`< cell_tex_min`) không xét.
  `obstruct` = tỉ lệ ô mất chi tiết (`< ref * cell_drop_ratio`). Vật che sát ống kính luôn nhòe -> mất chi tiết, bất kể màu.
- `edge_keep` = tỉ lệ cạnh Canny của ảnh gốc còn khớp -> `scene_changed` (xoay, vật rõ nét chắn trước).
- `mean` quá tối / quá chói -> `obstructed`. Hash frame (bỏ vùng `osd_mask`) giống hệt `frozen_sec` -> `frozen`.
- **Bộ đếm thời gian chung** cho mọi loại bất thường: đủ `hold_sec` thì báo loại nhiều phiếu nhất; hết sau `clear_sec`.
- **Ảnh gốc**: học `warmup_sec` lúc đầu, chỉ từ frame đủ chi tiết (`baseline_min_cells`, `min_ref_edges`), không đạt
  thì báo `baseline_invalid` và chờ (bắt "che trước rồi mới bật"). Sau đó cập nhật chậm (`ref_alpha`) **chỉ khi bình thường**.
- `scene_changed` kéo dài `relearn_sec` -> chấp nhận cảnh mới (event `relearn`).
- Có hình lại sau mất tín hiệu -> `recover_grace_sec` không báo sự cố mới, không học ảnh gốc (camera khởi động lại).
- `osd_mask` **bắt buộc** với camera có chữ đồng hồ: không mask thì chữ trên nền vật che bị coi là "chi tiết" -> học nhầm.

## Số đo (ngày, nguồn, cỡ mẫu)
| Số đo | Giá trị | Nguồn |
|---|---|---|
| Báo nhầm, camera văn phòng thật | 0 / 45 phút | 2026-10-16 09:34–10:19, giờ làm việc, có người qua lại |
| Báo nhầm, video công viên `park1` | 0 / 120 phút | YouTube 1080p, cây + người, chạy lại sau mỗi lần sửa |
| Bắt phá hoại giả `park1_tamper` | 3 / 3, đúng thời điểm + `hold_sec` | 15 phút đầu park1 + che kín / che 75% / làm mờ (ffmpeg) |
| Độ trễ báo `no_signal` khi rút dây | ~26s | 1 lần, camera văn phòng (gồm `read_timeout` 10s + ffprobe 10s) |
| Camera PTZ ổn định sau khi khởi động lại | ~25s | 1 lần (rút dây PoE) -> `recover_grace_sec` 60. **Mẫu nhỏ** |
| CPU | python ~4%, ffmpeg ~3% / camera | 1 camera văn phòng; NVDEC ~1%. Video 1080p chạy ~22x thời gian thực |

## Quyết định và lý do
- Đọc camera **chỉ bằng ffmpeg**, `hwaccel: false` để dự phòng CPU. Lý do: nhiều camera thì giải mã CPU nghẽn trước.
- `cv2.setNumThreads(1)` + `OMP_NUM_THREADS=1`: thread pool OpenCV spin-wait ăn ~1000% CPU với ảnh nhỏ (đo 2026-10-15).
- Snapshot độ phân giải gốc (`out_width: 0`). `--show` 2 fps là đủ, không làm hiển thị mượt (tốn tài nguyên).
- Lưới 8x8 (4x4 bỏ sót vùng chi tiết lẫn trong mảng trơn; 32x32 ô quá nhỏ, nhiễu).
- `hold_sec` 3 cho demo, 10 cho công viên (camera gắn cao, loại chim đậu / lá rụng). **Chưa đo trên camera công viên.**
- Lớp kiểm ảnh gốc chỉ chạy lúc học, không chạy khi vận hành (cảnh trơn tự nhiên như đêm, hồ nước sẽ báo nhầm).
- Lỗi code trong thread camera -> dừng toàn bộ, mã 1 (lỗi code thường đánh mọi camera; để Docker khởi động lại).
- Thống kê định kỳ `stats_sec` 600, in cả khi bằng 0 (phân biệt "không có gì" với "treo").
- Mất kết nối: mỗi lý do in 1 lần / sự cố (lý do đổi qua lại khi camera khởi động lại: đo 4 dòng / 1 lần rút dây).

## Đã thử và bỏ
- Học lại ảnh gốc khi có hình lại: bỏ, mở đường "rút dây, che, cắm lại".
- Ép học ảnh gốc sau timeout khi cảnh trơn: bỏ, học từ cảnh trơn thì vẫn mù.
- Test tranh thời gian bằng "in nhiều rồi chạy nhiều lần": không bao giờ lộ lỗi; thay bằng dựng chắc chắn tình huống.

## Giới hạn đã biết
- Che dưới ~60% khung hình không báo (chủ đích, chỉnh `obstruct_frac`).
- Che dần qua nhiều ngày (cây mọc, bụi bám) không báo: ảnh gốc học theo (chủ đích, thuộc bảo trì).
- Chưa thử: chạng vạng / chuyển IR, ban đêm, gió to, mưa, mạng nhện, nhiều camera cùng lúc trên server.
- Sau `recover_grace_sec`, sự cố có từ lúc mất tín hiệu mới được báo (chậm thêm tối đa 60s).

## Môi trường
- Dev **robo**: RTX A3000 Laptop, driver 535, conda `deepcv`, ffmpeg hệ thống 4.4.2 (có cuvid).
  Cảnh báo `libtinfo / libncursesw: no version information` khi ở deepcv: vô hại, do `LD_LIBRARY_PATH` của conda.
  Git push trong deepcv: đã đặt `core.sshCommand = env -u LD_LIBRARY_PATH ssh`.
- Server: V100, Ubuntu 22.04, Docker Compose; container cần `NVIDIA_DRIVER_CAPABILITIES=compute,utility,video`.
- Camera văn phòng `cam_01`: PTZ, HEVC 1280x960 @30, PoE, có OSD đồng hồ (dải 7% trên cùng).

## Việc còn lại (theo thứ tự)
1. Đo lại `recover_grace_sec` (3 lần rút dây, log DEBUG; và trên camera công viên khi có).
2. Lệnh kiểm hồi quy bằng video có ghi sẵn kết quả mong đợi (park1: 0, park1_tamper: 3) + cách tạo `park1_tamper.mp4`.
3. Đo biên an toàn các ngưỡng (edge_keep thấp nhất / obstruct cao nhất lúc bình thường), rồi khoá ngưỡng đã đo bằng test.
4. Thử chạng vạng / IR, ban đêm trên camera thật.
5. Đóng gói cho devops: `cameras.yaml` (cam_id UUID + url), `--check` (bảng OK / LỖI), ảnh gốc chuẩn chốt lúc lắp đặt, Docker 1 lệnh.
6. Chốt hợp đồng event với backend (trường, kênh Kafka / HTTP, phiên bản).
7. Ảnh "trước" làm bằng chứng (1 frame bình thường cuối cùng).
8. Điều tra `LD_LIBRARY_PATH` của deepcv (gốc của cảnh báo libtinfo và lỗi ssh).
9. Về lâu dài: một lớp đọc camera dùng chung cho mọi bài toán (giải mã 1 lần, chia frame).
