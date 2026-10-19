# CONTEXT — people_count (engine `overview`, đếm người / chó / mèo trong vùng)

Cập nhật: 2026-10-19. Cách làm việc: `docs/CACH_LAM_VIEC.md`. Kiến trúc cả hệ: `docs/KIEN_TRUC.md` (Project "AI in town park").

# PHẦN 1 — ĐỌC HẾT MỖI PHIÊN

## Hệ là gì (một câu)
Engine `overview` đầu tiên: mọi camera profile `overview` chạy trong **1 pipeline DeepStream** (1 model, batch = số camera);
module `count` đếm **hiện diện** (bao nhiêu người, chó, mèo đang ở trong từng vùng), báo cáo trung vị + max mỗi `report_sec`.

- Bài toán 5/7 của AI_park (GitHub public `maixuanthuan/AI_park`). Dev và chạy trên **host robo**.
- Module sau (phóng uế, lấn chiếm...) sẽ cắm vào **cùng engine này**: thêm module + config, không dựng pipeline mới.

## Đang ở đâu
- **Engine chạy được trên robo** (2026-10-19, 2 video, mã thoát 0, đủ 20740 frame). 43 test logic.
- **Đã đo: nút cổ chai là GIẢI MÃ (NVDEC 100%)**, không phải model (sm ~80%) hay CPU (~2% 1 nhân / camera 30 FPS).
- Gói v2 (2026-10-19): sửa 3 lỗi lộ ra khi chạy thật (median luôn float, thống kê đo từ frame đầu + fps từng camera, đếm cả báo cáo lúc dừng), config camera văn phòng.
- Gói v3 (2026-10-19): **dừng gọn bằng SIGTERM / SIGINT** (trước đó lệnh dừng không tới được python), video debug **mkv**. 48 test.
- **Camera văn phòng thật (2026-10-19, ~5 phút, `office2`)**: kết nối, nhận diện (khung `person 1` bám đúng người), báo cáo 30s đều,
  rút dây -> tự kết nối lại, cửa sổ đóng đúng chỗ mất hình, mkv đọc đủ 5010 frame. Còn 2 lỗi -> gói v4 (chờ thử):
  dừng bằng SIGTERM với RTSP phải chờ 5s rồi bỏ (EOS bị nvurisrcbin nuốt); báo nhầm "mất kết nối" lúc khởi động.
- Đã tách `common/` (config_check, masking); camera_tamper đổi import, 57 test vẫn xanh.
- **Chưa commit** `people_count/`, `common/`.

## Quyết định quan trọng nhất
1. Tách **hệ chạy đúng** (kiểm bằng test dữ liệu giả, làm ngay) với **model nhận đúng** (chỉ đo trên camera thật,
   bằng sai lệch so với đếm tay; sửa bằng dữ liệu / model, **không vá luật từng ca trong code**).
2. Đếm **hiện diện** trước. **Đếm lượt** đánh giá khi có camera thật (cần tracking ổn định; nên làm bằng vượt vạch ở cổng).

## Phiên tới làm gì
1. Thử gói v4 với camera văn phòng (dừng gọn < 5s, không báo nhầm lúc khởi động). Đạt -> **commit** `common/` + `people_count/` (mốc chạy ổn đầu tiên).
2. Câu hỏi còn mở: camera 4:3 bị mux 1920x1080 kéo giãn -> model có kém đi không. Nhìn ảnh chưa thấy méo rõ; chỉ kết luận được bằng đếm tay so sánh.

---

# PHẦN 2 — TRA KHI CẦN

## Cấu trúc
| File | Vai trò |
|---|---|
| `cameras.yaml` (devops) | camera, profile, chức năng bật. Video thử khai `url` (tính từ thư mục people_count) |
| `zones.yaml` (backend sau này) | vùng tỉ lệ 0..1, `ref_size`, `params.report_sec` |
| `profiles/overview.yaml` (dev) | mux, tracker, rtsp, lớp đếm, `aspect_tol`, `stats_sec` |
| `profiles/nvinfer_yolo26n.txt` | nvinfer: YOLO26n FP16, `cluster-mode=4`, chỉ giữ person/cat/dog, `topk=300`. Đường dẫn tương đối theo file này |
| `run_engine.py` | kiểm config (mã 2) -> chọn camera overview -> đứng ở `models/` -> dựng pipeline -> chia frame cho module, thống kê |
| `pipeline.py` | **duy nhất import Gst/pyds**: `nvurisrcbin` x N -> `nvstreammux` -> `nvinfer` -> `nvtracker` -> probe -> fakesink / (`--debug-out`) tiler-osd-mp4 |
| `meta.py` | `FrameMeta(cam_id, ts, src_w, src_h, objects)`, `Obj(cls, conf, bbox tỉ lệ, track_id)` |
| `zones.py` | kiểm zones.yaml, `anchor` (giữa cạnh đáy), `contains` (trên vạch = trong), `aspect_ok` |
| `counter.py` | `CountModule.update(meta) -> [báo cáo]`, `flush()` |
| `config.py` | kiểm cameras / zones / profile, mọi lỗi một lượt; `PROFILES` = module mỗi profile đã có |
| `events.py` | envelope event bản 0.1 |
| `Dockerfile` | `aipark-overview:7.0` = DeepStream 7.0 + pyds 1.1.11 + parser DeepStream-Yolo build trong image |
| `ds/` | bản thử cũ (`ds_count.py`, config cũ, `nvdsanalytics`): giữ để so số, bỏ khi engine mới chạy đúng |
| `models/` | `yolo26n.pt/.onnx`, `labels.txt`, `export_yolo26.py`, engine `model_b<N>_gpu0_fp16.engine` (không commit) |

## Thuật toán đếm
- Điểm đại diện = **giữa cạnh đáy** bbox (chân). Trên cạnh / đỉnh vùng tính là **trong**. bbox kẹp về 0..1.
- Mỗi frame, mỗi vùng: đếm theo lớp. **Mọi lớp ghi ở mọi frame (0 nếu không thấy)**.
- Hết `report_sec` (tính từ frame đầu cửa sổ): báo cáo `{window_start, window_end, frames, counts{lớp: median, max}}`.
- Mất hình lâu hơn `report_sec`: đóng cửa sổ cũ tại frame cuối, không kéo dài qua khoảng mất.
- Frame đầu tiên của camera: so tỉ lệ khung với `ref_size` mọi vùng; lệch > `aspect_tol` -> ERROR 1 lần, **không đếm camera đó**, camera khác chạy tiếp.
- Vùng xét ở logic, **không dùng nvdsanalytics** (vùng đổi lúc chạy; test không GPU).
- `ts`: file = PTS; camera = giờ máy lúc probe nhận frame.

## Số đo (ngày, nguồn, cỡ mẫu)
| Số đo | Giá trị | Nguồn |
|---|---|---|
| **Engine mới**: tốc độ 2 nguồn 1080p (park1 25 FPS + park1_2min 30 FPS), tracker NvSORT bật | **~630 FPS tổng** | robo, 20740 frame / 32.8s (dấu giờ log, từ "có hình" tới EOS), 2026-10-19, 2 lần chạy |
| **Engine mới**: GPU lúc chạy | **dec 93–100%, sm 78–82%, mem 47–51%**, enc 0 | `nvidia-smi dmon -s u`, 15 mẫu / 15s, 2026-10-19, 1 lần |
| **Engine mới**: CPU | ~45% 1 nhân tổng ở 630 FPS -> ~0.7 ms CPU / frame -> **~2% 1 nhân / camera 30 FPS**; luồng giải mã `nvv4l2d` ~0.6% | `top -H`, 1 mẫu 5s (mẫu thứ 2), 2026-10-19 |
| Dựng engine batch 2 | **4 phút 40 giây** (không phải 1-3 phút) | robo, lần chạy đầu, 2026-10-19 |
| Suy ra capacity robo (chưa kiểm trên camera thật) | giải mã bão hoà ~630 frame 1080p/s -> **~21 camera 1080p@30** | chia thẳng; camera thật là H.264/H.265 khác video thử -> phải đo lại |
| Bản thử: 1 nguồn 1080p, YOLO26n FP16 | ~530 FPS | robo, file, `sync=False`, park1 17139 frame, 2026-10-16 |
| CPU | ~4% 1 nhân / camera 30 FPS | `top -H`, lần đo thứ 2, 1 lần |
| Chạy lại cùng engine | giống hệt từng số | 2 lần, park1 |
| Dựng lại engine | lệch nhẹ (max 13 vs 14) | giữ cố định 1 engine để so sánh |
| Lỗi trung vị bản thử | chó có mặt 5/61 frame -> trung vị 1 (đúng: 0) | chạy class `Counts` gốc với dữ liệu giả, 2026-10-19; khoá bằng test C2 |

## Hiện tượng đã thấy (chưa sửa, chờ dữ liệu thật)
- **Người cúi / khom bị nhận là chó** (park1: 5 lần dog, 3 nhầm). Độ tin không tách được (nhầm 0.73, chó thật 0.28–0.41).
  Ứng viên khi có dữ liệu thật: thấy liên tục N giây (tracker), model lớn hơn, huấn luyện thêm.
- Video thử là phóng sự TV đổi góc liên tục -> **không dùng để chỉnh model**.

## Môi trường
- robo: RTX A3000 Laptop, driver 535, 16 nhân, 31 GB RAM, conda `deepcv`.
- DeepStream 7.0 (TensorRT 8.6, CUDA 12.2) trong Docker. Mount cả `AI_park` vào `/work` (cần `common/`).
- DeepStream-Yolo commit 2894bab (`~/my_deepstream_test/DeepStream-Yolo`). Image build lại parser từ GitHub cùng commit.

## Đã kiểm khi chạy thật (2026-10-19) / còn chưa kiểm
- Đã chạy đúng: `nvurisrcbin` với file, pad-added, parser build trong image, `config_tracker_NvSORT.yml` có trong image,
  nvinfer nhận engine + batch từ code, engine lưu ở `models/`, EOS nhiều nguồn, flush cuối.
- **Chưa kiểm**: property RTSP của `nvurisrcbin` (`rtsp-reconnect-interval`, `latency`, `select-rtp-protocol`) — chỉ chạy với camera thật; `--debug-out`.
- Cảnh báo vô hại khi chạy: GStreamer không nạp được vài plugin (rivermax, mpg123...), TensorRT báo trọng số FP16 subnormal,
  `No decoder available for type 'audio/mpeg'` (video có tiếng; camera thật có tiếng cũng sẽ in -> Việc tiếp theo 1).
- **Entrypoint gốc của image** (`deepstream-7.0/entrypoint.sh`): `docker top` thấy `docker-init -> bash entrypoint.sh -> python3`
  -> tín hiệu dừng dừng ở bash, python không biết (2026-10-19). Sửa: entrypoint riêng `exec` thẳng `nvidia_entrypoint.sh`, python
  thay thế shell; python bắt SIGTERM/SIGINT bằng `GLib.unix_signal_add` (phải `signal.signal(sig, SIG_DFL)` trước, không thì PyGObject
  chèn handler Ctrl+C của nó đè lên) -> gửi EOS -> chờ tối đa 5s (< 10s của docker stop).
- **EOS gửi vào pipeline bị nvurisrcbin (RTSP) nuốt**: nó thấy hết dữ liệu, tự kết nối lại (log: "No data from source since
  last 5 sec" ngay sau SIGTERM). Sửa v4: gửi EOS thẳng vào từng `sink_<i>` của nvstreammux. Lúc đang dừng không in lỗi nguồn.
- Lúc khởi động, nạp model ~5s >= `rtsp.reconnect_sec` 5 -> nvurisrcbin báo "No data from source" trước frame đầu.
  v4: trước frame đầu của camera, câu đó in INFO "đang chờ hình đầu tiên"; lỗi khác (sai mật khẩu...) in WARNING "chưa kết nối được".
- **Camera PTZ sau khi mất điện quay lên trần**, không về góc cũ -> đếm 0 dù hệ chạy đúng. Yêu cầu lắp đặt: đặt home preset.
  camera_tamper sẽ báo `scene_changed` cho trường hợp này (hai chức năng bổ trợ nhau trên cùng camera).
- Camera văn phòng thực tế **25 fps** (751 frame / cửa sổ 30s), không phải 30 như camera_tamper/CONTEXT ghi. Keyframe đều **2s** (ffprobe 20s).
- **Bẫy công cụ đo**: `ffmpeg -ss` nhảy giây trong mkv chưa có mục lục -> giải mã từ giữa GOP -> ảnh **vệt xanh giả**
  (Claude kết luận nhầm "đầu vào hỏng" 2026-10-19). Trích ảnh từ mkv debug: giải mã tuần tự `-vf "select=eq(n\,N)"`.
- **mp4 bị cắt ngang không xem được** (mục lục `moov` chỉ ghi lúc EOS) -> video debug bắt buộc `.mkv`.
- Đo bằng lệnh: entrypoint image tách lại `bash -c "..."` -> không bọc lệnh docker bằng `bash -c`; `kill -0` tiến trình root
  từ user thường báo "không có quyền" -> dùng `[ -d /proc/$P ]`.

## Hướng mới (2026-10-19 chiều): DeepStream engineer — nhiều camera, tận dụng GPU, tối ưu tài nguyên
Thuận muốn tập trung vào pipeline DeepStream ở mức engineer, nghiệp vụ chỉ là module cắm cuối. Lộ trình, mỗi bước là thí nghiệm
(dự đoán trước -> đo -> giải thích):
- A. Nhìn pipeline thật: A1 sơ đồ (`GST_DEBUG_DUMP_DOT_DIR`, `_dump_dot` trong pipeline.py), A2 độ trễ từng khối.
- B. Giả lập N camera (MediaMTX phát lại video thành RTSP đúng thời gian thực) + script benchmark N = 1, 2, 4, 8, 16: fps / dec% / sm% / CPU / VRAM.
- C. Đòn bẩy tối ưu, mỗi lần một biến: giải mã (độ phân giải, chỉ keyframe), inference (interval, batch, INT8, model n/s), hậu xử lý GPU, tracker.
- D. Production: một hay nhiều tiến trình, giám sát từng luồng, soak test 24h, Docker Compose.
- B đang làm: `tools/bench.sh` (relay MediaMTX nhân 1 camera văn phòng thành N luồng; Thuận đề xuất nhân bản camera thật thay vì quay
  video — vừa đo tải vừa thử relay của KIEN_TRUC 3.2). Dự đoán của Claude trước khi đo (tính từ số đã có): model trần ~790 frame/s
  (630 fps ở sm ~80%), model luôn co về 640x640 nên không phụ thuộc độ phân giải camera -> ~31 camera 25 fps; giải mã 1280x960 H.265
  nhẹ hơn video 1080p -> fps tụt ở ~25-30 camera, model hết sức trước (ngược sáng nay); hoặc VRAM GPU laptop hết trước.
- **Kết quả B đầu tiên (2026-10-19, camera văn phòng qua relay, H.265 1280x960 25 fps, đo 60s/N, 1 lần):**
  N=1: fps 25.0, sm 9%, dec 1%, VRAM 365 MB, CPU 9.5% 1 nhân. N=4: fps 25.0, sm 23%, dec 5%, VRAM 616 MB, CPU 9.3%. Relay mất gói 0.
  Đã kiểm toàn pipeline chạy: n4 có 12 báo cáo, mỗi camera 3. Đọc: batching -> 4 camera tốn 23% chứ không 36% (~9% -> ~6%/camera);
  giải mã ~1.2%/camera (H.265 1280x960 nhẹ, không phải nút cổ chai); VRAM ~84 MB/camera thêm. Ước lượng (chưa đo): sm chạm ~100% ở > 40 camera.
  Dựng engine batch 4: 255s. Relay mặc định kéo camera bằng UDP -> "1 RTP packet lost" -> đã ép TCP (`MTX_PATHS_CAM_RTSPTRANSPORT=tcp`).
- **Capacity robo, camera văn phòng nhân bản qua relay (2026-10-19, 60s/N, 1 lần mỗi N, relay mất gói 0):**
  | N | fps thấp nhất | sm% | dec% | VRAM MB | RAM python MB (đầu->cuối 60s) | CPU % 1 nhân |
  |---|---|---|---|---|---|---|
  | 1 | 25.0 | 16 | 1 | 365 | 2220->2221 | 5.7 |
  | 4 | 25.0 | 24 | 5 | 606 | 2408->2412 | 10.1 |
  | 8 | 25.0 | 34 | 10 | 928 | 3029->3038 | 16.8 |
  | 16 | 25.0 | 50 | 18 | 1571 | 3527->3543 | 29.8 |
  | 24 | 25.0 | 65 | 27 | 2205 | 4015->4042 | 53.1 |
  | 32 | 24.9 | 84 | 39 | 2842 | 4510->4548 | 65.4 |
  | 40 | 39/40 camera có hình sau 900s; camera còn lại "Resetting source" 90 lần | | | | | |
  N=40 (log): cả 39 camera tụt còn 17-22 fps, tổng ~800 frame/s (20-25.6 nghìn / 30s) -> **trần thông lượng model ~800 frame/s**
  (khớp ước lượng sáng 790) -> **capacity robo = 32 camera 25 fps, sát trần**; giao vận hành (chừa 20-30%) ~24-25 camera.
  Camera vào sau cùng không kịp frame đầu trong 5s -> "Resetting source" mãi: hệ quá tải "chết lặng" (vài camera không bao giờ lên), không sập.
  **Bài học (Claude sai 2026-10-19): % sử dụng (sm 84%) KHÔNG phải sức còn dư** -> đo THÔNG LƯỢNG (frame/s xử lý vs frame/s đến).
  Nhỏ: "đang chờ hình đầu tiên" in 2 lần cho cùng camera (câu của nvurisrcbin có lẽ khác nhau chút) -> chưa sửa.
  Mỗi camera thêm (16->32): sm ~2.1%, dec ~1.3%, VRAM ~79 MB, CPU ~2.2% 1 nhân -> model hết sức trước
  (ngược video 1080p H.264 sáng: giải mã trước) -> nút cổ chai do loại camera quyết định. Dựng engine b8/b16/b24/b32: 280-345s.
  Mở: (a) [đã trả lời: đúng, quá tải] (b) RAM python tăng trong 60s, tăng theo N (+1 MB ở N=1, +38 MB ở N=32): ấm bộ đệm hay rò bộ nhớ -> soak N=24 1 giờ;
  (c) sm N=1 lần này 16% vs 9% lần trước: mỗi điểm 1 lần chạy, ±vài %.
  Vì model là nút cổ chai: `nvinfer interval` (bỏ frame ở model, tracker nội suy) giờ CÓ tác dụng -> dự đoán nút cổ chai tiếp theo
  là giải mã / VRAM ~70+ camera (thí nghiệm C).
- **Giả thuyết của Thuận**: công viên đông người -> tốn hơn văn phòng (VRAM ~5 GB -> ~6 GB). Theo cơ chế: model chính quét cả ảnh 640x640
  mọi frame bất kể số người -> GPU (sm) và VRAM gần như KHÔNG đổi; tracker + probe + đếm (CPU) tăng theo số người; model phụ (sau này)
  tăng GPU theo số đối tượng. Thí nghiệm 1 biến: cắt park1 thành `park_dong` (~0.5-5.5 phút, trung vị 4-8 người) và `park_vang`
  (~7-11 phút, 0-1 người), cùng codec / độ phân giải / fps -> bench cùng N, so sm / VRAM / CPU.
- **Kết quả đông vs vắng (2026-10-19, N=16, park1 1080p H.264 25 fps, 60s, 1 lần):** đông / vắng: sm 39 / 44%, dec 62 / 61%,
  VRAM 1315 / 1315 MB, **CPU 50.2 / 33.0% 1 nhân (+~50%)**, fps thấp nhất 24.2 / 24.7. -> Giả thuyết Thuận đúng một nửa:
  đông người tốn hơn ở CPU (tracker + probe + đếm theo từng người), KHÔNG ở GPU / VRAM (model quét cả ảnh mọi frame).
  CPU vẫn xa trần (~3% 1 nhân / camera lúc đông, robo 16 nhân). Model phụ sau này sẽ tăng GPU theo số đối tượng -> đo lại.
  Phụ: cùng N=16, dec 62% (1080p H.264) vs 18% (camera văn phòng H.265 1280x960) -> loại luồng camera quyết định giải mã ~3x;
  camera 1080p H.264 thì giải mã chạm trần ~25 camera, trước model. -> **cần độ phân giải + codec luồng camera công viên để tính capacity.**
- **Soak 1 giờ, N=24 camera văn phòng (2026-10-19 15:50-16:50):** fps thấp nhất suốt 1 giờ 24.9; VRAM 2195 -> 2195 MB (không rò GPU);
  mất kết nối 0, relay mất gói 0; RAM python 3620 -> 3671 MB (+51 MB) — CHƯA kết luận: 2 điểm không phân biệt "ấm lên rồi đứng yên"
  với "rò đều 51 MB/h (~1.2 GB/ngày)". -> bench.sh ghi RAM mỗi 60s (`ram.tsv`) + cột `ram_doc_MB/h` (hồi quy nửa sau; kiểm bằng
  dữ liệu mẫu: đứng yên -> 0.0, rò đều 51 -> 51.1).
- Tiếp: soak qua đêm N=24 10 giờ (`BENCH_MEASURE=36000`, nohup + systemd-inhibit, laptop cắm sạc) -> thí nghiệm C1 `nvinfer interval`.
Dự đoán A1 của Thuận (ghi trước khi đo): nvurisrcbin ~4 khối (đọc luồng, kết nối lại, ghi theo sự kiện, giải mã); toàn pipeline trên GPU, Python chỉ metadata.

## Việc tiếp theo (theo thứ tự)
1. **Camera văn phòng thật** (Phiên tới 1-2). Mux theo tỉ lệ camera: thử `mux` 1280x960 vs 1920x1080 trên cùng camera (một biến),
   nhìn video debug; nếu cần thì bàn cách cho nhiều tỉ lệ khung trong 1 batch (đệm viền + đổi toạ độ). Dòng cảnh báo audio nếu camera có tiếng.
2. **Thí nghiệm giải mã** (khi biết thông số luồng camera thật) (nút cổ chai đã đo): cùng nội dung 1080p vs 360p -> FPS, `dec%`. Đòn bẩy giải mã là
   **độ phân giải / fps của luồng camera** (môi trường, config), KHÔNG phải bỏ frame ở model: bỏ frame ở nvinfer
   không giảm giải mã (H.264/H.265 phải giải mã mọi frame vì frame sau dựa vào frame trước). Đáng thử thêm: giải mã chỉ keyframe (cần kiểm property).
3. So trung vị / max với `ds_count.py` cùng video (person phải giống; dog có thể thấp hơn do sửa lỗi trung vị).
4. Đo CPU / FPS tracker bật vs tắt (CPU không phải nút cổ chai -> ưu tiên thấp).
5. RTSP camera văn phòng: rút dây xem log mất kết nối / có hình lại, đo thời gian kết nối lại.
6. Công cụ đo độ chính xác: clip + đếm tay -> MAE. Dùng khi có camera thật.
7. Commit `common/`, `people_count/` (kèm sửa import camera_tamper) ở mốc chạy ổn.
8. Về sau: đẩy event sang backend; module thứ hai trên engine này (phóng uế / lấn chiếm); vẽ vùng lên `--debug-out`.

## Việc dở dang ngoài bài này
- Kiểm DeepStream 7.0 + parser trên V100 trước khi chốt image triển khai (KIEN_TRUC việc mở 1).
- Thuận có ghi chú tự tổng hợp về cách làm việc với DeepStream -> gửi để Claude đọc, chỉnh.
- Thí nghiệm hiểu DeepStream (khi rảnh): dump sơ đồ pipeline, thêm `time.sleep(0.01)` vào probe.