# camera_tamper

Phát hiện camera bị phá: `obstructed` (che / mờ / tối / lóa), `scene_changed` (xoay lệch góc / cảnh bị thay), `frozen` (đóng băng hình), `no_signal` (mất stream), `baseline_invalid` (lúc khởi động cảnh không đủ chi tiết — có thể đang bị che).
Detector chỉ dùng opencv + numpy, chạy CPU, không cần model. Đọc camera bằng ffmpeg, giải mã trên GPU (NVDEC).

## Cấu trúc
- `tamper_detector.py` — lõi detector, không IO: `update(frame, ts)` -> event | None
- `frame_source.py` — đọc camera/video bằng ffmpeg subprocess (GPU hoặc CPU), watchdog, tự che user:pass trong log
- `run_tamper.py` — runner đa camera, mỗi camera 1 thread, `--show` để xem live; kiểm config trước khi khởi động
- `config_check.py` — kiểm một mục config: đủ mục, không mục lạ, đúng kiểu, trong khoảng
- `config.yaml` — tham số + danh sách camera (`url_env` trỏ tới biến trong `.env`)

## Config
Mọi tham số **bắt buộc**, không có giá trị mặc định ngầm trong code. Config sai thì chương trình **không khởi động**
(mã thoát 2) và in **mọi** lỗi một lượt: thiếu mục, gõ nhầm tên mục, sai kiểu, ngoài khoảng hợp lý, vùng đệm ngược chiều,
trùng `id` camera, thiếu biến trong `.env`, không thấy file video.
`source` và `detector` khai một lần; camera nào cần khác thì ghi đè trong `cameras[].source` / `cameras[].detector`.
- `.env.example` — mẫu `.env` (link camera, thông tin nhạy cảm)

## Yêu cầu
- ffmpeg/ffprobe hệ thống có hỗ trợ cuda (`ffmpeg -hwaccels` có `cuda`)
- Chạy trong Docker: cần `NVIDIA_DRIVER_CAPABILITIES=compute,utility,video` (thiếu `video` là không có NVDEC)

## Chạy
```bash
cp -n .env.example .env   # lần đầu, điền link camera
pip install python-dotenv pyyaml
python run_tamper.py --log-level DEBUG --show    # bấm q hoặc Ctrl+C để thoát
```
Event in ra log (JSON), ảnh snapshot lưu ở `tamper_events/`.

`--show`: hình camera ở trên, dải thông tin + ảnh gốc ở dưới. Lưới đỏ = ô mất chi tiết, xanh = bình thường, xám = ô trơn không xét, khung cam = vùng `osd_mask`.

## Chạy test
```bash
pip install -r requirements-dev.txt   # lần đầu: cài pytest
python -m pytest -q tests
```
Không cần camera, GPU hay ffmpeg: test tự dựng frame giả và tự truyền thời gian, chạy khoảng 20 giây.
Mỗi test có 3 dòng mô tả: tình huống thật / phần giả lập / đỏ thì hỏng gì. Chạy sau **mọi** thay đổi code.

## Tham số chính cần tune
- `obstruct_frac`, `cell_drop_ratio` — độ nhạy khi bị che
- `edge_keep_thr` — độ nhạy khi bị xoay
- `hold_sec` — nhanh vs ít báo nhầm
- `baseline_min_cells` — kiểm tra lúc học ảnh gốc; đặt 0 cho camera vốn nhìn cảnh trơn (hồ nước, bãi cỏ rộng)
- `grid` — đang dùng 8x8
- `osd_mask` — vùng chữ đồng hồ / logo cố định cần bỏ qua, theo từng camera (khung màu cam trên `--show`)

## Test báo nhầm bằng video công viên
Đặt video (camera cố định, không lia máy) vào `videos/`, khai báo trong `config.park.yaml`, rồi:
```bash
python run_tamper.py --config config.park.yaml
```
Hết video sẽ in tổng kết: thời lượng + số cảnh báo theo loại. Video bình thường thì mong muốn là 0.
- `source.out_width` — giảm khi chạy nhiều camera
