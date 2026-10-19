"""Thử DeepStream bằng Python: đếm đối tượng theo TỪNG LỚP trong vùng đếm, in số đếm + FPS.

Python chỉ lắp pipeline và đọc metadata ở 1 điểm chặn (probe); giải mã, TensorRT, vùng đếm, vẽ, ghi file
đều chạy trong các khối C/CUDA của DeepStream trên GPU. Probe KHÔNG kéo ảnh về CPU.

python3 ds_count.py --uri file:///work/videos/park1_2min.mp4 --out /work/out/park1_py.mp4
"""
import argparse
import statistics
import sys
import time
from collections import Counter, defaultdict

import gi

gi.require_version("Gst", "1.0")
from gi.repository import GLib, Gst  # noqa: E402

import pyds  # noqa: E402

WINDOW_SEC = 30  # in số trung vị mỗi 30 giây VIDEO
WATCH_GAP_SEC = 1.0  # lớp hiếm (--watch): in 1 dòng khi mới xuất hiện, không in lại nếu vẫn thấy liên tục


class Counts:
    """Số đối tượng trong vùng theo từng frame; cuối mỗi cửa sổ in trung vị theo lớp."""

    def __init__(self):
        self.frames = 0
        self.t_first = self.t_last = None  # giờ máy lúc frame đầu / cuối: FPS không tính thời gian nạp model
        self.window = defaultdict(list)   # (vùng, lớp) -> [số đếm từng frame]
        self.window_start = None

    def add(self, ts, per_frame):
        self.frames += 1
        self.t_last = time.time()
        if self.t_first is None:
            self.t_first = self.t_last
        if self.window_start is None:
            self.window_start = ts
        for key in set(self.window) | set(per_frame):
            self.window[key].append(per_frame.get(key, 0))
        if ts - self.window_start >= WINDOW_SEC:
            self.flush(ts)

    def flush(self, ts):
        if not self.window:
            return
        parts = [f"{zone}/{label}: trung vị {statistics.median(v):g}, max {max(v)}"
                 for (zone, label), v in sorted(self.window.items())]
        print(f"[video {self.window_start:6.1f}s -> {ts:6.1f}s] " + " | ".join(parts), flush=True)
        self.window.clear()
        self.window_start = ts


def analytics_probe(pad, info, ctx):
    counts, watch, last_seen = ctx
    buf = info.get_buffer()
    batch = pyds.gst_buffer_get_nvds_batch_meta(hash(buf))
    l_frame = batch.frame_meta_list
    while l_frame is not None:
        frame = pyds.NvDsFrameMeta.cast(l_frame.data)
        ts = frame.buf_pts / 1e9
        per_frame = Counter()
        l_obj = frame.obj_meta_list
        while l_obj is not None:
            obj = pyds.NvDsObjectMeta.cast(l_obj.data)
            if obj.obj_label in watch:  # lớp hiếm: in thời điểm để tua video kiểm bằng mắt
                if ts - last_seen.get(obj.obj_label, -1e9) > WATCH_GAP_SEC:
                    r = obj.rect_params
                    print(f"  THẤY {obj.obj_label} lúc {ts:.2f}s, độ tin {obj.confidence:.2f}, "
                          f"khung x={r.left:.0f} y={r.top:.0f} w={r.width:.0f} h={r.height:.0f}", flush=True)
                last_seen[obj.obj_label] = ts
            l_user = obj.obj_user_meta_list
            while l_user is not None:
                um = pyds.NvDsUserMeta.cast(l_user.data)
                if um.base_meta.meta_type == pyds.nvds_get_user_meta_type("NVIDIA.DSANALYTICSOBJ.USER_META"):
                    info_obj = pyds.NvDsAnalyticsObjInfo.cast(um.user_meta_data)
                    for zone in info_obj.roiStatus:      # các vùng mà đối tượng này nằm trong
                        per_frame[(zone, obj.obj_label)] += 1
                l_user = l_user.next
            l_obj = l_obj.next
        counts.add(ts, per_frame)
        l_frame = l_frame.next
    return Gst.PadProbeReturn.OK


def make(factory, name, **props):
    e = Gst.ElementFactory.make(factory, name)
    if e is None:
        sys.exit(f"không tạo được khối GStreamer '{factory}'")
    for k, v in props.items():
        e.set_property(k.replace("_", "-"), v)
    return e


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--uri", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--analytics", default="/work/ds/config_nvdsanalytics.txt")
    ap.add_argument("--pgie", default="/work/ds/config_infer_yolo26.txt", help="config model (nvinfer)")
    ap.add_argument("--watch", default="dog,cat", help="lớp hiếm cần in thời điểm xuất hiện, cách nhau dấu phẩy")
    args = ap.parse_args()

    Gst.init(None)
    pipe = Gst.Pipeline()
    src = make("uridecodebin", "src", uri=args.uri)
    mux = make("nvstreammux", "mux", batch_size=1, width=1920, height=1080, batched_push_timeout=40000)
    pgie = make("nvinfer", "pgie", config_file_path=args.pgie)
    ana = make("nvdsanalytics", "ana", config_file=args.analytics)
    conv1 = make("nvvideoconvert", "conv1")
    osd = make("nvdsosd", "osd")
    conv2 = make("nvvideoconvert", "conv2")
    caps = make("capsfilter", "caps", caps=Gst.Caps.from_string("video/x-raw(memory:NVMM), format=I420"))
    enc = make("nvv4l2h264enc", "enc", bitrate=4000000)
    parse = make("h264parse", "parse")
    mp4 = make("qtmux", "mp4")
    sink = make("filesink", "sink", location=args.out, sync=False)
    chain = [mux, pgie, ana, conv1, osd, conv2, caps, enc, parse, mp4, sink]
    for e in [src] + chain:
        pipe.add(e)
    for a, b in zip(chain, chain[1:]):
        if not a.link(b):
            sys.exit(f"không nối được {a.get_name()} -> {b.get_name()}")

    def on_pad(_src, pad):  # uridecodebin tạo pad video sau khi đọc file: nối vào mux
        if pad.query_caps(None).to_string().startswith("video/x-raw"):
            pad.link(mux.request_pad_simple("sink_0"))

    src.connect("pad-added", on_pad)

    counts = Counts()
    watch = {x.strip() for x in args.watch.split(",") if x.strip()}
    ana.get_static_pad("src").add_probe(Gst.PadProbeType.BUFFER, analytics_probe, (counts, watch, {}))

    loop = GLib.MainLoop()

    def on_msg(_bus, msg):
        if msg.type == Gst.MessageType.EOS:
            loop.quit()
        elif msg.type == Gst.MessageType.ERROR:
            err, dbg = msg.parse_error()
            print(f"LỖI: {err}: {dbg}", flush=True)
            loop.quit()

    bus = pipe.get_bus()
    bus.add_signal_watch()
    bus.connect("message", on_msg)

    pipe.set_state(Gst.State.PLAYING)
    try:
        loop.run()
    finally:
        pipe.set_state(Gst.State.NULL)
    counts.flush(counts.window_start or 0)
    if counts.frames > 1:
        dt = counts.t_last - counts.t_first
        print(f"XONG: {counts.frames} frame trong {dt:.1f}s = {(counts.frames - 1) / dt:.1f} FPS "
              f"(tính từ frame đầu, không gồm nạp model). File: {args.out}", flush=True)
    else:
        print("XONG: không có frame nào đi qua (xem dòng LỖI ở trên)", flush=True)


if __name__ == "__main__":
    main()