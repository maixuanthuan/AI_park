# camera_tamper

Phát hiện camera bị phá: `obstructed` (che / mờ / tối / lóa), `scene_changed` (xoay lệch góc / cảnh bị thay), `frozen` (đóng băng hình), `no_signal` (mất stream), `baseline_invalid` (lúc khởi động cảnh không đủ chi tiết — có thể đang bị che).
Detector chỉ dùng opencv + numpy, chạy CPU, không cần model. Đọc camera bằng ffmpeg, giải mã trên GPU (NVDEC).

## Cấu trúc
- `tamper_detector.py` — lõi detector, không IO: `update(frame, ts)` -> event | None
- `frame_source.py` — đọc camera/video bằng ffmpeg subprocess (GPU hoặc CPU), watchdog, tự che user:pass trong log
- `run_tamper.py` — runner đa camera, mỗi camera 1 thread, `--show` để xem live; kiểm config trước khi khởi động
- `../common/config_check.py` — kiểm một mục config: đủ mục, không mục lạ, đúng kiểu, trong khoảng (dùng chung với people_count)
- `../common/masking.py` — che user:pass trong URL trước khi in log (dùng chung)
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

## Đọc log
Log là cho **người vận hành / người tiếp nhận** đọc (hệ có khoẻ không, hỏng ở đâu). Cảnh báo cho người quản lý đi theo
đường **event** (dòng JSON `tamper_start` / `tamper_end`), sau này đẩy sang backend.

| Dòng log | Nghĩa | Làm gì |
|---|---|---|
| `INFO thống kê 600s: 3 camera \| ok 2, mất kết nối 1 \| ... frame (2.0 fps/camera) \| cảnh báo mới 0 \| thử kết nối lại 0` | In đều mỗi `stats_sec`, **kể cả khi mọi số bằng 0** | Dòng này ngừng xuất hiện = chương trình treo. `fps/camera` tụt dưới `sample_fps` = camera / mạng chậm |
| `INFO cam_x: hevc 1280x960 -> ... decode=gpu` | Đã kết nối được camera | `decode=cpu` khi `hwaccel: true` = ffmpeg không dùng được GPU |
| `WARNING cam_x: mất kết nối: <lý do>` | Bắt đầu mất kết nối. Các lần thử lại sau **không in nữa**, trừ khi gặp **lý do mới** (mỗi lý do in 1 lần / sự cố) | Đọc lý do: `401` = sai mật khẩu, `404` = sai đường dẫn, `Connection refused` / timeout = mạng / camera tắt (**phía devops**) |
| `INFO cam_x: có hình lại sau 12.3 phút, đã thử 144 lần` | Hết sự cố mất kết nối | Không cần làm gì |
| `WARNING {"event": "tamper_start", ...}` | **Event**: camera bị phá / mất tín hiệu | Đây là cảnh báo nghiệp vụ, xem `reason` và ảnh trong `out_dir` |
| `ERROR ...: không khởi động, N lỗi: ...` (mã thoát 2) | Config sai, chưa mở camera nào | Sửa đúng các mục được liệt kê (**phía người sửa config**) |
| `ERROR cam_x: lỗi code trong thread camera ...` + traceback (mã thoát 1) | Lỗi code, cả chương trình dừng để Docker khởi động lại | Gửi traceback cho **dev** |
| `INFO cam_x: hết video / dừng giữa chừng — đã chạy N phút, K cảnh báo {...}` | Tổng kết khi dừng | Số liệu báo cáo |

Mã thoát: `0` dừng bình thường, `1` lỗi code, `2` config sai.

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
- `recover_grace_sec` — có hình lại sau mất tín hiệu: chờ camera khởi động lại xong (PTZ xoay, lấy nét) mới báo sự cố mới;
  trong lúc chờ không học ảnh gốc, nên "rút dây, che, cắm lại" vẫn bị bắt khi hết thời gian chờ
- `grid` — đang dùng 8x8
- `osd_mask` — vùng chữ đồng hồ / logo cố định cần bỏ qua, theo từng camera (khung màu cam trên `--show`)

## Test báo nhầm bằng video công viên
Đặt video (camera cố định, không lia máy) vào `videos/`, khai báo trong `config.park.yaml`, rồi:
```bash
python run_tamper.py --config config.park.yaml
```
Hết video sẽ in tổng kết: thời lượng + số cảnh báo theo loại. Video bình thường thì mong muốn là 0.
- `source.out_width` — giảm khi chạy nhiều camera
