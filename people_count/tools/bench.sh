#!/bin/bash
# Đo tải engine overview: 1 nguồn -> MediaMTX (relay, rtsp://127.0.0.1:8554/cam) -> nhân thành N "camera" cho engine đọc.
# Mỗi N: chờ mọi camera có hình, đo BENCH_MEASURE giây, dừng gọn, sang N tiếp.
#
#   people_count/tools/bench.sh [N ...]                                  nguồn = camera văn phòng (CAM_01_URL trong .env)
#   BENCH_SOURCE=people_count/videos/x.mp4 people_count/tools/bench.sh 16  nguồn = file video, phát lặp đúng thời gian thực
#   BENCH_MEASURE=36000 people_count/tools/bench.sh 24                   chạy lâu (soak test): 10 giờ, xem cột ram_doc_MB/h
# Mặc định N: 1 4 8 16 24 32; BENCH_MEASURE: 60 giây.
#
# Đọc: CAM_01_URL trong people_count/.env (không in ra) hoặc file BENCH_SOURCE. Camera chỉ có 1 kết nối (MediaMTX).
# Ghi: people_count/out/bench/<nguồn>/ (log, số đo, ketqua.tsv); engine models/model_b<N>_gpu0_fp16.engine lần đầu mỗi N.
# Dừng giữa chừng: Ctrl+C (tự dừng engine, relay, ffmpeg phát file).
set -u
HERE=$(cd "$(dirname "$0")/.." && pwd)   # people_count
ROOT=$(dirname "$HERE")                  # AI_park
IMAGE=aipark-overview:7.0
RELAY_IMAGE=bluenviron/mediamtx:1
RELAY=aipark-relay
ENGINE=aipark-bench
RTSP=rtsp://127.0.0.1:8554/cam
READY_TIMEOUT=900   # chờ N camera có hình; gồm dựng engine batch N lần đầu (batch 4: 255s — đo 2026-10-19)
MEASURE=${BENCH_MEASURE:-60}   # stats_sec của profiles/bench.yaml = 30 -> nên >= 60 để có >= 2 dòng thống kê
SOURCE=${BENCH_SOURCE:-}
NS=${*:-1 4 8 16 24 32}
FFPID=""

mask() { sed -E 's#//[^/[:space:]]*@#//***@#g'; }
die() { echo "LỖI: $*" >&2; exit 1; }
cleanup() {
    docker stop "$ENGINE" >/dev/null 2>&1
    [ -n "$FFPID" ] && kill "$FFPID" 2>/dev/null
    if docker inspect "$RELAY" >/dev/null 2>&1; then
        docker logs "$RELAY" 2>&1 | mask > "$OUT/relay.log"
        docker stop "$RELAY" >/dev/null 2>&1
    fi
}
trap cleanup EXIT
trap 'exit 130' INT TERM

# ---- nguồn: camera thật (relay kéo) hoặc file (ffmpeg trên máy phát vào relay) ----
docker rm -f "$RELAY" "$ENGINE" >/dev/null 2>&1
if [ -z "$SOURCE" ]; then
    TAG=camera; REF_W=1280; REF_H=960   # khung camera văn phòng (đo 2026-10-19)
    [ -f "$HERE/.env" ] || die "thiếu $HERE/.env (cần dòng CAM_01_URL=...)"
    CAM_01_URL=$(grep -m1 '^CAM_01_URL=' "$HERE/.env" | cut -d= -f2-)
    CAM_01_URL=${CAM_01_URL%\"}; CAM_01_URL=${CAM_01_URL#\"}; CAM_01_URL=${CAM_01_URL%\'}; CAM_01_URL=${CAM_01_URL#\'}
    [ -n "$CAM_01_URL" ] || die "$HERE/.env chưa có CAM_01_URL"
    # rtspTransport=tcp: mặc định "automatic" kéo camera bằng UDP -> mất gói (thấy 2026-10-19: "1 RTP packet lost" sau 15s)
    # -> hình hỏng tới keyframe sau. Engine cũng đọc bằng TCP vì lý do này.
    docker run -d --rm --name "$RELAY" --network host -e MTX_PATHS_CAM_SOURCE="$CAM_01_URL" \
        -e MTX_PATHS_CAM_RTSPTRANSPORT=tcp "$RELAY_IMAGE" >/dev/null \
        || die "không chạy được MediaMTX ($RELAY_IMAGE). Cổng 8554 đang bị chiếm?"
else
    [ -f "$SOURCE" ] || die "không thấy file $SOURCE"
    TAG=$(basename "${SOURCE%.*}")
    # csv mặc định phân cách bằng dấu phẩy; ffprobe KHÔNG nhận dấu cách làm phân cách (s=' ' -> lỗi, thấy 2026-10-19)
    read -r REF_W REF_H < <(ffprobe -v error -select_streams v:0 -show_entries stream=width,height -of csv=p=0 "$SOURCE" \
        2>/dev/null | tr ',' ' ')
    if [ -z "${REF_H:-}" ]; then
        ffprobe -v error -select_streams v:0 -show_entries stream=width,height -of csv=p=0 "$SOURCE" 2>&1 \
            | grep -v "no version information" | tail -3
        die "ffprobe không đọc được kích thước khung của $SOURCE (lỗi của ffprobe ở trên)"
    fi
    docker run -d --rm --name "$RELAY" --network host "$RELAY_IMAGE" >/dev/null \
        || die "không chạy được MediaMTX ($RELAY_IMAGE). Cổng 8554 đang bị chiếm?"
    sleep 2
    # -re: phát đúng tốc độ thời gian thực như camera; -c copy: không giải mã / nén lại (gần như không tốn CPU)
    ffmpeg -nostdin -loglevel error -re -stream_loop -1 -i "$SOURCE" -an -c copy -f rtsp -rtsp_transport tcp "$RTSP" \
        2> >(grep -v "no version information" >&2) &
    FFPID=$!
fi
OUT=$HERE/out/bench/$TAG
mkdir -p "$OUT"
for _ in $(seq 30); do
    docker logs "$RELAY" 2>&1 | grep -qE "\[path cam\].*(online|ready)|publishing to path 'cam'" && break
    sleep 1
done
docker logs "$RELAY" 2>&1 | grep -qE "\[path cam\].*(online|ready)|publishing to path 'cam'" \
    || { docker logs "$RELAY" 2>&1 | mask | tail -5; die "relay chưa có hình sau 30s (xem 5 dòng trên)"; }
echo "relay: có hình từ nguồn '$TAG' (${REF_W}x${REF_H}) -> $RTSP; đo ${MEASURE}s mỗi N"

GPU_MB=$(nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits | head -1)
RES="$OUT/ketqua.tsv"
printf "N\ttrang_thai\tfps_thap_nhat\tfps_cao_nhat\tsm_tb%%\tdec_tb%%\tvram_dau_MB\tvram_cuoi_MB(/%s)\tram_dau_MB\tram_cuoi_MB\tram_doc_MB/h\tcpu_%%1nhan\tmat_ket_noi\n" "$GPU_MB" > "$RES"

for N in $NS; do
    D=$OUT/n$N; rm -rf "$D"; mkdir -p "$D"
    # N camera cùng 1 link (relay), mỗi camera 1 vùng cả khung, ref_size = khung nguồn
    { echo "cameras:"
      for i in $(seq 1 "$N"); do
          printf '  - {cam_id: bench-%02d, name: "bench %02d", url_env: BENCH_URL, profile: overview, functions: [count]}\n' "$i" "$i"
      done; } > "$D/cameras.yaml"
    { echo "zones:"
      for i in $(seq 1 "$N"); do
          printf '  - {zone_id: z-bench-%02d, cam_id: bench-%02d, function: count, polygon: [[0,0],[1,0],[1,1],[0,1]], ref_size: [%s, %s], params: {report_sec: 30}, version: 1}\n' "$i" "$i" "$REF_W" "$REF_H"
      done; } > "$D/zones.yaml"

    echo "== N=$N: khởi động engine (lần đầu với N này sẽ dựng engine ~5 phút)..."
    docker run -d --rm --init --name "$ENGINE" --network host --gpus all -e BENCH_URL="$RTSP" -v "$ROOT":/work "$IMAGE" \
        python3 /work/people_count/run_engine.py --cameras "out/bench/$TAG/n$N/cameras.yaml" \
        --zones "out/bench/$TAG/n$N/zones.yaml" --profile profiles/bench.yaml >/dev/null || die "không chạy được engine"
    docker logs -f "$ENGINE" > "$D/engine.log" 2>&1 &
    LOGPID=$!

    status=ok; waited=0
    while :; do
        up=$(grep -c ": có hình [0-9]" "$D/engine.log")
        [ "$up" -ge "$N" ] && break
        if ! docker inspect "$ENGINE" >/dev/null 2>&1; then status="engine_dừng_sớm"; break; fi
        if [ "$waited" -ge "$READY_TIMEOUT" ]; then status="quá_${READY_TIMEOUT}s_chưa_đủ_hình_($up/$N)"; break; fi
        sleep 5; waited=$((waited + 5))
    done

    fps_lo=-; fps_hi=-; sm=-; dec=-; fb0=-; fb1=-; ram0=-; ram1=-; slope=-; cpu=-; lost=-
    if [ "$status" = ok ]; then
        echo "   đủ $N camera có hình sau ${waited}s, đo ${MEASURE}s..."
        PID=$(pgrep -f "^python3 /work/people_count/run_engine.py" | head -1)
        start=$(wc -l < "$D/engine.log")
        ram0=$(( $(ps -o rss= -p "$PID") / 1024 ))
        nvidia-smi dmon -s um -c "$MEASURE" > "$D/dmon.txt" 2>&1 &
        DM=$!
        # RAM python mỗi 60s -> ram.tsv (giây, MB): 2 điểm đầu / cuối không phân biệt được "ấm lên rồi đứng yên"
        # với "tăng đều = rò bộ nhớ" (soak 1 giờ 2026-10-19: +51 MB, không kết luận được)
        ( t=0; while [ -d "/proc/$PID" ] && [ "$t" -le "$MEASURE" ]; do
              printf "%s\t%s\n" "$t" "$(( $(ps -o rss= -p "$PID" 2>/dev/null || echo 0) / 1024 ))"
              sleep 60; t=$((t + 60)); done ) > "$D/ram.tsv" &
        RP=$!
        LC_ALL=C top -b -n 2 -d 30 -p "$PID" > "$D/top.txt" 2>&1
        wait "$DM"
        kill "$RP" 2>/dev/null
        # độ dốc RAM (MB/giờ): hồi quy tuyến tính trên NỬA SAU (bỏ giai đoạn ấm lên); cần >= 4 điểm (>= ~7 phút đo)
        slope=$(awk '{t[NR]=$1; r[NR]=$2} END{h=int(NR/2); n=0; for(i=h+1;i<=NR;i++){n++; sx+=t[i]; sy+=r[i]; sxx+=t[i]*t[i]; sxy+=t[i]*r[i]}
            if(n<4){print "-"; exit} d=n*sxx-sx*sx; if(d==0){print "-"; exit} printf "%.1f\n", (n*sxy-sx*sy)/d*3600}' "$D/ram.tsv")
        ram1=$(( $(ps -o rss= -p "$PID" 2>/dev/null || echo 0) / 1024 ))
        if ! docker inspect "$ENGINE" >/dev/null 2>&1; then status="engine_chết_lúc_đo"; fi
        # chỉ xét log từ lúc bắt đầu đo: fps thấp nhất / cao nhất qua MỌI dòng thống kê (chạy lâu thì thấy được tụt dần)
        tail -n +"$((start + 1))" "$D/engine.log" > "$D/do.log"
        read -r fps_lo fps_hi < <(sed -n 's/.*thấp nhất \([0-9.]*\) .*cao nhất \([0-9.]*\) .*/\1 \2/p' "$D/do.log" |
            awk 'NR==1{lo=$1; hi=$2} {if($1<lo) lo=$1; if($2>hi) hi=$2} END{if(NR) print lo, hi; else print "- -"}')
        lost=$(grep -c "mất kết nối" "$D/do.log")
        # dmon: cột theo tên ở dòng tiêu đề "# gpu sm mem enc dec ... fb ..."; dòng số liệu lệch 1 cột (không có "#")
        # VRAM: trung bình 10 mẫu đầu vs 10 mẫu cuối -> tăng dần khi chạy lâu = rò bộ nhớ GPU
        read -r sm dec fb0 fb1 < <(awk '/^# gpu/{for(i=2;i<=NF;i++) c[$i]=i-1; next} /^#/{next}
            NF{n++; s+=$c["sm"]; d+=$c["dec"]; f[n]=$c["fb"]+0}
            END{if(!n){print "- - - -"; exit} k=(n<10?n:10); for(i=1;i<=k;i++){a+=f[i]; b+=f[n-k+i]}
                printf "%.0f %.0f %.0f %.0f\n", s/n, d/n, a/k, b/k}' "$D/dmon.txt")
        cpu=$(awk -v p="$PID" '$1==p{c=$9} END{print (c==""?"-":c)}' "$D/top.txt")
    fi
    docker stop "$ENGINE" >/dev/null 2>&1
    wait "$LOGPID" 2>/dev/null
    printf "%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n" "$N" "$status" "${fps_lo:--}" "${fps_hi:--}" "$sm" "$dec" \
        "$fb0" "$fb1" "$ram0" "$ram1" "$slope" "$cpu" "$lost" >> "$RES"
    echo "   $(tail -1 "$RES")"
    if [ "$status" != ok ]; then
        echo "   dừng dãy N: $status (10 dòng cuối log: $D/engine.log)"
        tail -10 "$D/engine.log" | mask
        break
    fi
done

echo
echo "== KẾT QUẢ nguồn '$TAG' ($RES)"
column -t -s $'\t' "$RES" 2>/dev/null || cat "$RES"
lost=$(docker logs "$RELAY" 2>&1 | grep -c "packet.* lost")
echo "relay báo mất gói: $lost lần (phải là 0; khác 0 thì hình đầu vào có thể hỏng, xem $OUT/relay.log)"