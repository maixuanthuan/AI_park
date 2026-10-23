# AI_park

Các bài toán AI cho hệ thống camera giám sát công viên. Mỗi bài toán là một thư mục riêng, chạy độc lập.

## Các bài toán

| # | Bài toán | Thư mục | Trạng thái |
|---|---|---|---|
| 1 | Nhận diện & bóc tách biển số xe | — | Đã làm, chưa đưa vào repo này |
| 2 | Nhận diện khuôn mặt | — | Đã làm, chưa đưa vào repo này |
| 3 | Phát hiện camera bị che / làm mờ / xoay / mất tín hiệu | [`camera_tamper/`](camera_tamper/) | Xong giai đoạn chuẩn bị: chạy trên camera thật + video, 57 test |
| 4 | Nhận diện vật nuôi phóng uế | [`demo/`](demo/) (`phong_ue`) | PoC: keypoint DeepLabCut, động vật ngồi xổm + ảnh bằng chứng; đúng khi con vật đủ to, nhìn ngang |
| 5 | Đo đếm người, vật nuôi trong khuôn viên | [`people_count/`](people_count/), [`demo/`](demo/) (`count`) | Engine `overview` (DeepStream) chạy đúng trên robo (đo tới 32 camera) + PoC đếm hiện diện |
| 6 | Nhận diện đám đông, biểu tình / gây mất trật tự | [`demo/`](demo/) (`crowd`, `bang_ron`) | PoC: đám đông + băng rôn cầm tay; luật kết hợp "biểu tình" chưa có |
| 7 | Phát hiện lấn chiếm vỉa hè | [`demo/`](demo/) (`lan_chiem`) | PoC: logic xong (vật đứng yên trùng chỗ); chưa chạy trên camera cố định |

Hệ thống AI chỉ **phát hiện và phát sự kiện**. Người quản lý xem hình qua đầu ghi / phần mềm camera như bình thường; backend nhận sự kiện để thông báo.

## Demo cho khách (PoC)
[`demo/`](demo/): mỗi video / camera gán chức năng riêng (`count`, `crowd`, `bang_ron`, `lan_chiem`, `phong_ue`), ra video đã vẽ + event
+ ảnh bằng chứng. Python thuần (ultralytics, DeepLabCut), dùng lại logic của `people_count`. Trạng thái + việc mở: `demo/CONTEXT.md`.

## Dữ liệu cho model chung (`overview`)
[`dataset/`](dataset/): `classes.yaml` = danh sách lớp duy nhất; `nhan_dien.py` = thử nhận diện từng lớp trên clip (COCO vs YOLOE) + gán nhãn tự động chuẩn YOLO để sửa trong CVAT.

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
- **Clip thử nằm trong project**: `data/<bài>/` (`.gitignore` chặn `/data/`), không để ở thư mục home.
- Đọc camera bằng **ffmpeg, giải mã trên GPU (NVDEC)** để tiết kiệm CPU khi chạy nhiều camera.
- Mỗi bài toán ghi rõ quyết định thiết kế **kèm lý do và số đo** trong `CONTEXT.md` của nó.

Kiến trúc cả hệ (profile, config, vùng, event): `docs/KIEN_TRUC.md` trong Project "AI in town park".

## Chạy test
Mỗi bài toán chạy test **riêng** (hai thư mục có file test trùng tên): `python -m pytest -q camera_tamper/tests`, `python -m pytest -q people_count/tests`, `demo/tests`, `dataset/tests`.

## Đóng gói gửi đi
`tools/pack.sh`: một lệnh, lấy file theo git (bỏ mọi thứ `.gitignore` chặn), dừng nếu gói có `.env` hoặc giá trị lấy từ `.env`.
Không nén tay thư mục (lần nén tay 2026-10-21 lọt link camera có mật khẩu).

## Bắt đầu

Xem `README.md` trong thư mục của từng bài toán, ví dụ [`camera_tamper/README.md`](camera_tamper/README.md).
