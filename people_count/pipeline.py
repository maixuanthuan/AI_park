"""Phần DUY NHẤT dính DeepStream (Gst + pyds): lắp pipeline N camera, đổi metadata sang FrameMeta (meta.py).

nvurisrcbin x N -> nvstreammux (batch N) -> nvinfer -> [nvtracker] -> probe -> fakesink
                                                                       └-> (--debug-out) tiler -> osd -> mkv
Probe KHÔNG kéo ảnh về CPU, chỉ đọc metadata. Logic (vùng, đếm) không import file này (KIEN_TRUC mục 3.4):
đổi phiên bản DeepStream thì chỉ sửa ở đây.
"""
import logging
import math
import os
import re
import signal
import time

import gi

gi.require_version("Gst", "1.0")
from gi.repository import GLib, Gst  # noqa: E402

import pyds  # noqa: E402

from common.masking import mask  # noqa: E402
from meta import FrameMeta, Obj  # noqa: E402

log = logging.getLogger("engine")
TRACKER_LIB = "/opt/nvidia/deepstream/deepstream/lib/libnvds_nvmultiobjecttracker.so"
TCP_ONLY = 4  # nvurisrcbin select-rtp-protocol: 4 = chỉ TCP (như -rtsp_transport tcp của camera_tamper)
_ADDR = re.compile(r"0x[0-9a-fA-F]+")
DOT_AFTER_SEC = 15  # xuất sơ đồ sau 15s: nạp model ~5s + camera nối xong, các khối con của nvurisrcbin đã được tạo
GIVE_UP_SEC = 5  # nhận lệnh dừng mà pipeline chưa kết thúc sau 5s thì dừng hẳn; phải < 10s mặc định của docker stop


def _make(factory, name, **props):
    e = Gst.ElementFactory.make(factory, name)
    if e is None:
        raise RuntimeError(f"không tạo được khối GStreamer '{factory}' (thiếu plugin DeepStream?)")
    for k, v in props.items():
        e.set_property(k.replace("_", "-"), v)
    return e


def _clamp(v):
    return min(max(v, 0.0), 1.0)


class Pipeline:
    """cams: [{"cam_id", "uri", "live"}]; prof: profile đã kiểm; on_frames(list[FrameMeta]) gọi trong probe."""

    def __init__(self, cams, prof, pgie_config, engine_file, on_frames, debug_out=None):
        Gst.init(None)
        self.cams, self.on_frames = cams, on_frames
        self.w, self.h = prof["mux"]["width"], prof["mux"]["height"]
        self.tracking = prof["tracker"]["enable"]
        self.reconnect_sec = prof["rtsp"]["reconnect_sec"]
        self.stopping = False
        self.down = {}  # cam_id -> các lý do mất kết nối đã in trong sự cố đang diễn ra
        self.seen = set()  # camera đã có ít nhất 1 frame (trước đó: "đang chờ hình đầu tiên", không phải "mất kết nối")
        self.fatal = False
        self.loop = GLib.MainLoop()
        n = len(cams)
        p = self.pipe = Gst.Pipeline()

        # enable-padding=0: co giãn về khung mux không đệm viền -> toạ độ tỉ lệ 0..1 vẫn đúng theo khung gốc
        mux = _make("nvstreammux", "mux", batch_size=n, width=self.w, height=self.h, enable_padding=0,
                    batched_push_timeout=prof["mux"]["batched_push_timeout_us"],
                    live_source=int(any(c["live"] for c in cams)))
        pgie = _make("nvinfer", "pgie", config_file_path=pgie_config, batch_size=n, model_engine_file=engine_file)
        chain = [mux, pgie]
        if self.tracking:
            t = prof["tracker"]
            chain.append(_make("nvtracker", "tracker", ll_lib_file=TRACKER_LIB, ll_config_file=t["config"],
                               tracker_width=t["width"], tracker_height=t["height"]))
        probe_at = chain[-1]

        if debug_out:  # phần phụ cho dev xem bằng mắt: mặc định tắt, bỏ đi thì lõi vẫn chạy y nguyên
            rows = int(math.ceil(math.sqrt(n)))
            chain += [_make("nvmultistreamtiler", "tiler", rows=rows, columns=int(math.ceil(n / rows)),
                            width=self.w, height=self.h),
                      _make("nvvideoconvert", "conv1"), _make("nvdsosd", "osd"), _make("nvvideoconvert", "conv2"),
                      _make("capsfilter", "caps",
                            caps=Gst.Caps.from_string("video/x-raw(memory:NVMM), format=I420")),
                      _make("nvv4l2h264enc", "enc", bitrate=4000000), _make("h264parse", "parse"),
                      # mkv, KHÔNG mp4: mp4 chỉ ghi mục lục (moov) lúc kết thúc -> bị cắt ngang là không xem được
                      # (gặp 2026-10-19 với camera thật). mkv ghi từng đoạn: dừng kiểu gì cũng xem được tới đoạn cuối.
                      _make("matroskamux", "mkv"), _make("filesink", "sink", location=debug_out, sync=False)]
        else:
            chain.append(_make("fakesink", "sink", sync=False))

        for e in chain:
            p.add(e)
        for a, b in zip(chain, chain[1:]):
            if not a.link(b):
                raise RuntimeError(f"không nối được {a.get_name()} -> {b.get_name()}")

        self.mux_sinks = []  # đầu vào của nvstreammux, mỗi camera 1 pad: lúc dừng gửi EOS thẳng vào đây
        for i, cam in enumerate(cams):
            props = dict(uri=cam["uri"])
            if cam["live"]:
                props.update(rtsp_reconnect_interval=prof["rtsp"]["reconnect_sec"],
                             latency=prof["rtsp"]["latency_ms"], select_rtp_protocol=TCP_ONLY)
            src = _make("nvurisrcbin", f"src_{i}", **props)
            sink_pad = mux.request_pad_simple(f"sink_{i}")
            self.mux_sinks.append(sink_pad)
            src.connect("pad-added", self._on_pad, sink_pad)
            p.add(src)

        probe_at.get_static_pad("src").add_probe(Gst.PadProbeType.BUFFER, self._probe, None)
        bus = p.get_bus()
        bus.add_signal_watch()
        bus.connect("message", self._on_msg)

    # ---------- chạy ----------
    def run(self, stats_sec, on_stats):
        """Chạy tới khi hết mọi nguồn (EOS), Ctrl+C, hoặc lỗi. Trả về True nếu dừng vì lỗi (mã thoát 1)."""
        GLib.timeout_add_seconds(int(stats_sec), lambda: on_stats() or True)
        # SIGTERM = docker stop / docker compose down / khởi động lại máy; SIGINT = Ctrl+C. Cả hai: dừng GỌN
        # (gửi EOS -> báo cáo cửa sổ đang dở, file debug ghi xong). Bắt bằng GLib để chạy trong vòng lặp chính.
        for sig in (signal.SIGTERM, signal.SIGINT):
            # bỏ handler Python trước: còn handler mặc định của Python thì PyGObject tự chèn handler Ctrl+C riêng
            # của nó lúc loop.run(), đè lên handler GLib bên dưới
            signal.signal(sig, signal.SIG_DFL)
            GLib.unix_signal_add(GLib.PRIORITY_HIGH, sig, self._on_signal, sig)
        self.pipe.set_state(Gst.State.PLAYING)
        if os.environ.get("GST_DEBUG_DUMP_DOT_DIR"):  # công cụ cho dev nhìn pipeline thật, mặc định tắt
            GLib.timeout_add_seconds(DOT_AFTER_SEC, self._dump_dot)
        try:
            self.loop.run()
        finally:
            self.pipe.set_state(Gst.State.NULL)
        return self.fatal

    def _on_signal(self, sig):
        name = signal.Signals(sig).name
        if self.stopping:
            log.warning("nhận %s lần nữa, dừng ngay", name)
            self.loop.quit()
            return True
        self.stopping = True
        log.info("nhận %s, đang dừng gọn (tối đa %ds)...", name, GIVE_UP_SEC)
        # EOS gửi THẲNG vào đầu vào của nvstreammux, không qua nguồn camera: gửi vào pipeline thì EOS đi qua nvurisrcbin,
        # nó thấy RTSP hết dữ liệu và tự kết nối lại -> EOS bị nuốt, phải chờ 5s rồi bỏ (đo 2026-10-19, camera văn phòng).
        # Mux nhận EOS ở mọi đầu vào -> đẩy EOS xuôi pipeline -> file debug ghi xong -> bus báo EOS -> dừng.
        for pad in self.mux_sinks:
            pad.send_event(Gst.Event.new_eos())
        GLib.timeout_add_seconds(GIVE_UP_SEC, self._give_up)
        return True

    def _dump_dot(self):
        """Ghi sơ đồ pipeline đang chạy (mọi khối, kể cả khối con tự sinh bên trong bin, kèm caps trên từng liên kết)
        ra GST_DEBUG_DUMP_DOT_DIR/pipeline.dot. Ứng dụng tự viết phải tự gọi; chỉ gst-launch mới tự xuất."""
        Gst.debug_bin_to_dot_file(self.pipe, Gst.DebugGraphDetails.ALL, "pipeline")
        log.info("đã xuất sơ đồ pipeline: %s/pipeline.dot", os.environ["GST_DEBUG_DUMP_DOT_DIR"])
        return False

    def _give_up(self):
        log.warning("pipeline không kết thúc trong %ds sau lệnh dừng, dừng hẳn", GIVE_UP_SEC)
        self.loop.quit()
        return False

    def stop(self):
        GLib.idle_add(self.loop.quit)

    # ---------- nội bộ ----------
    @staticmethod
    def _on_pad(_src, pad, sink_pad):
        if "video" in pad.query_caps(None).to_string() and not sink_pad.is_linked():
            pad.link(sink_pad)

    def _cam_of(self, obj):
        """Camera nào gây ra message (tìm bin cha tên src_<i>); None = khối chung của pipeline."""
        while obj is not None:
            name = obj.get_name() or ""
            if name.startswith("src_") and name[4:].isdigit():
                return self.cams[int(name[4:])]
            obj = obj.get_parent()
        return None

    def _on_msg(self, _bus, msg):
        if msg.type == Gst.MessageType.EOS:
            log.info("pipeline kết thúc (EOS: hết mọi video, hoặc sau Ctrl+C)")
            self.loop.quit()
        elif msg.type in (Gst.MessageType.ERROR, Gst.MessageType.WARNING):
            err, dbg = msg.parse_error() if msg.type == Gst.MessageType.ERROR else msg.parse_warning()
            cam = self._cam_of(msg.src)
            text = mask(f"{err.message} | {dbg or ''}")
            if cam is not None and self.stopping:
                return True  # đang dừng: nguồn camera báo lỗi / mất dữ liệu là đương nhiên, không in
            if cam is not None and cam["live"]:
                # camera rớt / chưa lên: nvurisrcbin tự thử kết nối lại, các camera khác chạy tiếp.
                # Mỗi sự cố chỉ in LÝ DO MỚI (bỏ địa chỉ bộ nhớ trước khi so), như camera_tamper; có hình lại in 1 dòng.
                cid = cam["cam_id"]
                seen = self.down.setdefault(cid, set())
                key = _ADDR.sub("0x", err.message)
                if key not in seen:
                    seen.add(key)
                    if cid in self.seen:
                        log.warning("%s: mất kết nối: %s (tự thử lại mỗi %ss, chỉ báo lý do mới)",
                                    cid, text, self.reconnect_sec)
                    elif "No data from source" in err.message:
                        # câu của nvurisrcbin khi quá rtsp-reconnect-interval chưa có dữ liệu: lúc khởi động là bình
                        # thường (nạp model ~5s), không phải mất kết nối (báo nhầm thấy 2026-10-19)
                        log.info("%s: đang chờ hình đầu tiên (nvurisrcbin tự thử lại mỗi %ss)", cid, self.reconnect_sec)
                    else:  # sai mật khẩu, sai đường dẫn, camera tắt... ngay từ đầu: phải lộ ra
                        log.warning("%s: chưa kết nối được: %s (tự thử lại mỗi %ss, chỉ báo lý do mới)",
                                    cid, text, self.reconnect_sec)
            elif msg.type == Gst.MessageType.ERROR:
                who = cam["cam_id"] if cam else "pipeline"
                log.error("%s: lỗi, dừng toàn bộ (mã 1): %s", who, text)
                self.fatal = True
                self.loop.quit()
            else:
                log.warning("cảnh báo GStreamer: %s", text)
        return True

    def _probe(self, _pad, info, _):
        try:
            batch = pyds.gst_buffer_get_nvds_batch_meta(hash(info.get_buffer()))
            frames = []
            l_frame = batch.frame_meta_list
            while l_frame is not None:
                f = pyds.NvDsFrameMeta.cast(l_frame.data)
                cam = self.cams[f.source_id]
                if cam["cam_id"] in self.down:
                    del self.down[cam["cam_id"]]
                    if cam["cam_id"] in self.seen:  # lần đầu có hình thì run_engine in "có hình WxH", không in ở đây
                        log.info("%s: có hình lại", cam["cam_id"])
                self.seen.add(cam["cam_id"])
                objs = []
                l_obj = f.obj_meta_list
                while l_obj is not None:
                    o = pyds.NvDsObjectMeta.cast(l_obj.data)
                    r = o.rect_params
                    # bbox về tỉ lệ 0..1 theo khung mux (= khung gốc vì không đệm viền); khung lấn ra mép thì kẹp lại
                    bbox = (_clamp(r.left / self.w), _clamp(r.top / self.h),
                            _clamp((r.left + r.width) / self.w), _clamp((r.top + r.height) / self.h))
                    objs.append(Obj(o.obj_label, float(o.confidence), bbox,
                                    int(o.object_id) if self.tracking else None))
                    l_obj = l_obj.next
                ts = time.time() if cam["live"] else f.buf_pts / 1e9
                frames.append(FrameMeta(cam["cam_id"], ts, f.source_frame_width, f.source_frame_height, tuple(objs)))
                l_frame = l_frame.next
            self.on_frames(frames)
        except Exception:
            # lỗi code = hệ hỏng, không phải đầu vào xấu: dừng toàn bộ, mã 1 (như camera_tamper)
            log.exception("lỗi code khi xử lý metadata, dừng toàn bộ chương trình (mã 1)")
            self.fatal = True
            self.stop()
        return Gst.PadProbeReturn.OK