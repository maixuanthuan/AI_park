# AI_park

Các bài toán AI cho hệ thống camera giám sát công viên. Mỗi bài toán là một thư mục riêng, chạy độc lập.

## Các bài toán

| # | Bài toán | Thư mục | Trạng thái |
|---|---|---|---|
| 1 | Nhận diện & bóc tách biển số xe | — | Đã làm, chưa đưa vào repo này |
| 2 | Nhận diện khuôn mặt | — | Đã làm, chưa đưa vào repo này |
| 3 | Phát hiện camera bị che / làm mờ / xoay / mất tín hiệu | [`camera_tamper/`](camera_tamper/) | Chạy được trên camera thật và video; đang hoàn thiện (test, cấu hình, log) |
| 4 | Nhận diện vật nuôi phóng uế | — | Chưa làm |
| 5 | Đo đếm người, vật nuôi trong khuôn viên | — | Chưa làm |
| 6 | Nhận diện đám đông, biểu tình / gây mất trật tự | — | Chưa làm |
| 7 | Phát hiện lấn chiếm vỉa hè | — | Chưa làm |

Hệ thống AI chỉ **phát hiện và phát sự kiện**. Người quản lý xem hình qua đầu ghi / phần mềm camera như bình thường; backend nhận sự kiện để thông báo.

## Cấu trúc

```
AI_park/
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

## Bắt đầu

Xem `README.md` trong thư mục của từng bài toán, ví dụ [`camera_tamper/README.md`](camera_tamper/README.md).
