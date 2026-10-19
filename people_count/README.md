# people_count — engine `overview`, module đếm hiện diện

Đếm số **người, chó, mèo đang ở trong từng vùng** của mỗi camera, gửi báo cáo định kỳ (trung vị + max theo lớp).
Mọi camera profile `overview` chạy trong **1 pipeline DeepStream** (1 model dùng chung, batch = số camera).
Kiến trúc chung: `docs/KIEN_TRUC.md` (Project "AI in town park").

## Cấu trúc
| File | Ai sửa | Vai trò |
|---|---|---|
| `cameras.yaml` | devops | camera nào, profile nào, bật chức năng nào |
| `.env` (mẫu `.env.example`) | devops | link camera có user/pass. Không vào git |
| `zones.yaml` | backend (sau này qua API) | vùng (tỉ lệ 0..1) + ngưỡng nghiệp vụ (`report_sec`) |
| `profiles/overview.yaml`, `profiles/nvinfer_yolo26n.txt` | dev | thông số kỹ thuật, model |
| `run_engine.py` | | điểm vào: kiểm config, chạy, log vận hành, thống kê định kỳ, mã thoát 0/1/2 |
| `pipeline.py` | | **file duy nhất dính DeepStream** (Gst, pyds): lắp pipeline, metadata -> `FrameMeta` |
| `meta.py`, `zones.py`, `counter.py`, `config.py`, `events.py` | | logic thuần, không cần GPU |
| `../common/` | | dùng chung với camera_tamper: kiểm config, che mật khẩu |
| `models/` | | ONNX, labels, script xuất ONNX; engine là cache (không commit) |
| `ds/` | | bản thử cũ (`ds_count.py`), giữ để so số, sẽ bỏ |

## Chạy (máy robo, từ thư mục `~/PycharmProjects/AI_park`)
```bash
docker build -t aipark-overview:7.0 people_count
docker run --rm --init --gpus all -v ~/PycharmProjects/AI_park:/work aipark-overview:7.0 python3 /work/people_count/run_engine.py
```
- Camera văn phòng thật: thêm `--cameras cameras.office.yaml --zones zones.office.yaml` (link `CAM_01_URL` trong `.env`).
- Dừng gọn: `docker stop <tên container>` (SIGTERM, như Docker Compose) hoặc Ctrl+C: gửi báo cáo cửa sổ đang dở, ghi xong file debug,
  tối đa 5s. Cần `docker run --init` và image có entrypoint riêng (Dockerfile) để tín hiệu tới thẳng python.
- `--debug-out out/x.mkv`: ghi video có vẽ khung để xem bằng mắt (chỉ để dev, mặc định tắt). Bắt buộc `.mkv`: bị cắt ngang vẫn xem được.
- Engine `models/model_b<N>_gpu0_fp16.engine` theo số camera N: lần đầu dựng 1-3 phút, các lần sau nạp ngay.
- File do container tạo thuộc root: `sudo chown -R $USER:$USER people_count/models people_count/out`.

## Đọc log
| Dòng log | Nghĩa | Làm gì |
|---|---|---|
| `INFO {"schema_version": ..., "event_type": "report", "function": "count", ...}` | **Event** báo cáo đếm của 1 vùng | Dữ liệu nghiệp vụ, sau này đẩy sang backend |
| `INFO thống kê 600s: 3 camera \| 0 camera không có frame \| ... \| fps/camera thấp nhất 24.9 (cam_x), cao nhất 25.1 (cam_y) \| ...` | In đều mỗi `stats_sec`, **kể cả khi bằng 0**; đo từ frame đầu tiên (không tính lúc dựng engine) | Ngừng xuất hiện = treo. `camera không có frame [...]` hoặc fps thấp nhất tụt = camera / mạng (**devops**) |
| `INFO thống kê: 3 camera \| chưa có frame nào ...` | Đang khởi động / dựng engine / chưa camera nào lên | Kéo dài quá vài phút = xem dòng mất kết nối |
| `INFO cam_x: có hình 1920x1080, 2 vùng` | Frame đầu tiên của camera, tỉ lệ khung khớp vùng | |
| `ERROR cam_x: luồng 704x576 khác tỉ lệ khung lúc vẽ vùng ...` | Không đếm camera này, các camera khác vẫn chạy | Sửa link sang luồng đúng tỉ lệ, hoặc vẽ lại vùng |
| `WARNING cam_x: mất kết nối: ...` | Camera rớt, tự thử lại; mỗi sự cố chỉ in lý do mới | `401` sai mật khẩu, `404` sai đường dẫn, timeout = mạng (**devops**) |
| `ERROR không khởi động, N lỗi: ...` (mã 2) | Config sai, chưa dựng pipeline | Sửa đúng các mục được liệt kê |
| `ERROR ...: lỗi, dừng toàn bộ (mã 1)` / traceback | Lỗi code hoặc pipeline | Gửi log cho **dev** |

## Chạy test (không cần GPU, DeepStream, video)
```bash
python -m pytest -q people_count/tests
```
Mỗi test có 3 dòng mô tả: tình huống thật / phần giả lập / đỏ thì hỏng gì.