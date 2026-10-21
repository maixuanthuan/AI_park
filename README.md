# AI_park

Các bài toán AI cho hệ thống camera giám sát công viên. Mỗi bài toán là một thư mục riêng, chạy độc lập.

## Các bài toán

| # | Bài toán | Thư mục | Trạng thái |
|---|---|---|---|
| 1 | Nhận diện & bóc tách biển số xe | — | Đã làm, chưa đưa vào repo này |
| 2 | Nhận diện khuôn mặt | — | Đã làm, chưa đưa vào repo này |
| 3 | Phát hiện camera bị che / làm mờ / xoay / mất tín hiệu | [`camera_tamper/`](camera_tamper/) | Xong giai đoạn chuẩn bị: chạy trên camera thật + video, 57 test |
| 4 | Nhận diện vật nuôi phóng uế | — | Chưa làm |
| 5 | Đo đếm người, vật nuôi trong khuôn viên | [`people_count/`](people_count/) | Engine `overview` (DeepStream) nhiều camera + đếm hiện diện; chờ chạy thử trên robo |
| 6 | Nhận diện đám đông, biểu tình / gây mất trật tự | — | Chưa làm |
| 7 | Phát hiện lấn chiếm vỉa hè | — | Chưa làm |

Hệ thống AI chỉ **phát hiện và phát sự kiện**. Người quản lý xem hình qua đầu ghi / phần mềm camera như bình thường; backend nhận sự kiện để thông báo.

## Cấu trúc

```
AI_park/
├── common/          code dùng chung (kiểm config, che mật khẩu), tách khi bài thứ hai cần
├── camera_tamper/   mỗi bài toán một thư mục:
│   ├── README.md      cách chạy
│   ├── CONTEXT.md     bối cảnh, thiết kế, quyết định và lý do
│   └── ...
└── README.md        file này
```

## Nguyên tắc chung

- **Video, ảnh, log không vào repo** (xem `.gitignore`): vừa nặng, vừa có thể lộ hình ảnh hiện trường.
- Đọc camera bằng **ffmpeg, giải mã trên GPU (NVDEC)** để tiết kiệm CPU khi chạy nhiều camera.
- Mỗi bài toán ghi rõ quyết định thiết kế **kèm lý do và số đo** trong `CONTEXT.md` của nó.

Kiến trúc cả hệ (profile, config, vùng, event): `docs/KIEN_TRUC.md` trong Project "AI in town park".

## Chạy test
Mỗi bài toán chạy test **riêng** (hai thư mục có file test trùng tên): `python -m pytest -q camera_tamper/tests`, `python -m pytest -q people_count/tests`.

## Đóng gói gửi đi
`tools/pack.sh`: một lệnh, lấy file theo git (bỏ mọi thứ `.gitignore` chặn), dừng nếu gói có `.env` hoặc giá trị lấy từ `.env`.
Không nén tay thư mục (lần nén tay 2026-10-21 lọt link camera có mật khẩu).

## Bắt đầu

Xem `README.md` trong thư mục của từng bài toán, ví dụ [`camera_tamper/README.md`](camera_tamper/README.md).
