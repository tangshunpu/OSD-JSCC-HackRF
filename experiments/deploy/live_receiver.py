"""Live camera demo, receiver side (RTX PRO 6000): Wi-Fi or HackRF -> TensorRT decoder -> web page.

Pairs with ``live_sender.py`` on the MacBook. Run from the repo root:

    W=~/.cache/trt_wheels
    RUN="uv run --with $W/tensorrt_cu13_libs-10.16.1.11-py3-none-manylinux_2_28_x86_64.whl \
         --with $W/tensorrt_cu13_bindings-10.16.1.11-cp312-none-manylinux_2_28_x86_64.whl python"
    $RUN experiments/deploy/live_receiver.py --link rf     # real 2.38 GHz HackRF link
    $RUN experiments/deploy/live_receiver.py               # Wi-Fi + simulated AWGN

and open http://<this host>:8777 in a browser (the rate buttons, and with
``--link wifi`` the SNR slider, are sent back to the sender).

``--link rf``: the symbols go only over the air, through the 10 MHz OFDM PHY of
``ofdm_phy.py`` (continuous HackRF
streams on both ends, this HackRF receiving the whole time). TCP carries only
per-frame metadata (the burst's position in the sender's TX stream, rate,
timestamps), the receiver's SNR feedback, and a JPEG of the camera frame for
display. The TX gain is fixed on the sender; the SNR is the real one: every
frame is decoded at the SNR estimated from its own packet headers, and the
sender conditions its encoder on the latest estimate it got back (both
clamped to the 0-20 dB training range; the page shows the unclamped value).
Lost packets are zero-filled; a frame with no packet at all is counted as
lost and the page keeps the previous frame.

Temporal smoothing (``--smooth``, default on; switch on the page): every frame
is decoded on its own, so static background edges and textures come out a
little different each frame and the video flickers. Before display, each
pixel is blended with the previous displayed frame, weighted by how much the
area changed (motion-adaptive, see ``TemporalSmoother``); it uses only the
received frames, adds ~0.5 ms, and frames.csv records whether it was on.

``--link wifi``: the fp16 symbols go over TCP and AWGN at the SNR set on the
page is added here.

Decoding uses the TensorRT FP16 engines of ``receiver.py`` (JSCC decoder, one
routed U-Net step, image decoder; ``--models tiny`` = the pruned image decoder
of OSD-JSCC-P (tiny), ``osd`` = the full VAE decoder) as one CUDA graph. Only
the newest frame is processed; one overtaken while waiting is dropped.

Per connection the receiver writes ``<out-dir>/<time>/frames.csv`` (every
timestamp plus the stage times below) and ``summary.json`` (fps, latency and,
over the air, SNR and loss statistics); ``--record`` adds a side-by-side
``record.mp4``. The page's Record button saves the dashboard itself as a video
(``live_record.py``: headless Chromium on this host, H.264) under
``<out-dir>/recordings/``, with a download link on the page.

Stages of one frame (ms; sender times are mapped to this clock with the
PING/PONG offset of the lowest round trip):
    wait      camera frame delivered -> picked by the encoder loop
    prep      crop/resize/mirror on the Mac
    encode    Core ML encoder (uint8 RGB -> fp16 symbols in host memory)
  Wi-Fi:
    network   encoder output -> whole frame received here (send + Wi-Fi + recv)
    queue     received -> decoder start (incl. the simulated AWGN)
  RF:
    modulate  symbols -> OFDM int8 I/Q (C++ kernel)
    tx_queue  I/Q ready -> burst started in the TX stream (previous burst, USB transfer boundary)
    radio     burst started in the TX stream -> its last sample captured here
              (TX USB queue, air time, RX USB buffering; a busy receiver adds its wait)
    demod     decimation, packet detection/equalization, de-mapping (last pass)
  decode    TensorRT decoder (symbols in host memory -> uint8 RGB in host memory)
  smooth    temporal smoothing for display (0 when switched off)
  output    JPEG for the web page
  e2e       camera frame delivered -> JPEG published
The camera's own exposure/ISP delay before OpenCV returns the frame and the
browser's display delay are not included.
"""

from __future__ import annotations

import argparse
from collections import deque
import csv
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
from pathlib import Path
import shutil
import signal
import socket
import statistics
import subprocess
import sys
import threading
import time

import cv2
import numpy as np

HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parents[1]
sys.path.insert(0, str(HERE))
from live_link import ACK, DEFAULT_PORT, FRAME, PING, PONG, PREVIEW, Channel  # noqa: E402

SNR_RANGE = (0.0, 20.0)  # training range of the model's SNR conditioning
STAGES = {
    "wifi": ("wait", "prep", "encode", "network", "queue", "decode", "smooth", "output"),
    # tx_wait: encoded frame waiting for the sender's transmitter thread (pipelined encode/modulate, 11-09)
    "rf": ("wait", "prep", "encode", "tx_wait", "modulate", "tx_queue", "radio", "demod", "decode", "smooth", "output"),
}
GROUPS = {  # display groups of the stages (web page, console)
    "wifi": [("Capture", ("wait", "prep")), ("Encode", ("encode",)), ("Network", ("network",)),
             ("Decode", ("decode",)), ("Queue + display", ("queue", "smooth", "output"))],
    "rf": [("Capture", ("wait", "prep")), ("Encode", ("encode",)), ("Modulate", ("tx_wait", "modulate", "tx_queue")),
           ("Radio link", ("radio",)), ("Demodulate", ("demod",)), ("Decode", ("decode",)), ("Smooth + JPEG", ("smooth", "output"))],
}


FIELDS = {  # frames.csv columns (a lost RF frame leaves its timing columns empty)
    "wifi": ["seq", "camera_index", "lost", "smoothing", "snr_db", "rate", "channels", "payload_bytes", "channel_uses",
             *(f"{k}_ms" for k in STAGES["wifi"]), "e2e_ms", "t_capture", "t_pick", "t_encode_start",
             "t_encode_end", "t_recv", "t_decode_start", "t_decode_end", "t_out", "clock_offset_ms",
             "sync_rtt_ms", "jpeg_bytes"],
    "rf": ["seq", "camera_index", "lost", "model", "video_key", "smoothing", "snr_db", "decoder_snr_db", "encoder_snr_db", "rate",
           "channels",
           "channel_uses", "packets", "total_packets", "inferred_packets", "tx_gain", "air_ms", *(f"{k}_ms" for k in STAGES["rf"]),
           "e2e_ms", "window", "demod_passes", "rx_wait_ms", "demod_total_ms", "rx_queue_ms", "metadata_ms",
           "rx_rail_fraction", "tx_underruns", "t_capture", "t_pick", "t_encode_start", "t_encode_end",
           "t_modulate_start", "t_modulate_end", "t_tx_start", "t_recv", "t_work", "t_rx_ready", "t_demod_end", "t_decode_end",
           "t_out", "clock_offset_ms", "sync_rtt_ms", "jpeg_bytes"],
}


def clamp_snr(value: float) -> float:
    return float(min(max(value, SNR_RANGE[0]), SNR_RANGE[1]))


# ------------------------------------------------------------------ helpers

class Hub:
    """Latest JPEG per stream name, with waiting for a newer one (MJPEG + snapshots)."""

    def __init__(self) -> None:
        self.cond = threading.Condition()
        self.items: dict[str, tuple[int, bytes]] = {}

    def publish(self, name: str, jpeg: bytes) -> None:
        with self.cond:
            count = self.items.get(name, (0, b""))[0] + 1
            self.items[name] = (count, jpeg)
            self.cond.notify_all()

    def wait(self, name: str, last: int, timeout: float):
        with self.cond:
            self.cond.wait_for(lambda: self.items.get(name, (0, b""))[0] > last, timeout)
            return self.items.get(name)


class Clock:
    """Receiver clock minus sender clock from PING/PONG, lowest-RTT sample of a sliding window."""

    def __init__(self) -> None:
        self.samples: deque = deque(maxlen=120)
        self.lock = threading.Lock()

    def add(self, t_a: int, t_b: int, t_c: int) -> None:
        with self.lock:
            self.samples.append((t_c - t_a, (t_a + t_c) // 2 - t_b))

    def best(self) -> tuple[int, int] | None:
        with self.lock:
            if not self.samples:
                return None
            rtt, offset = min(self.samples)
            return offset, rtt


def pct(values, q: float) -> float:
    values = sorted(v for v in values if v == v)
    if not values:
        return float("nan")
    k = (len(values) - 1) * q
    lo, hi = int(k), min(int(k) + 1, len(values) - 1)
    return values[lo] + (values[hi] - values[lo]) * (k - lo)


def describe(values) -> dict:
    values = [v for v in values if v is not None and v == v]
    if not values:
        return {}
    return {"mean": statistics.fmean(values), "median": pct(values, 0.5), "p95": pct(values, 0.95),
            "p99": pct(values, 0.99), "min": min(values), "max": max(values)}


def fps_of(times_ns) -> float:
    times_ns = list(times_ns)
    if len(times_ns) < 3 or times_ns[-1] == times_ns[0]:
        return 0.0
    return (len(times_ns) - 1) / ((times_ns[-1] - times_ns[0]) / 1e9)


class TemporalSmoother:
    """Motion-adaptive temporal smoothing of decoded frames, for display, on the GPU.

    out_t = out_{t-1} + A * (x_t - out_{t-1}), per pixel. A is ``alpha_min`` where the
    picture is static and 1 where it moves: the change of each 8x8 block (block means,
    |x_t - out_{t-1}| averaged over RGB) against the frame's median block change, which is
    the decoder's frame-to-frame flicker level since most of a frame is static (clamped
    to 0.5-4 levels): A ramps from alpha_min at ``k_lo`` x that level to 1 at ``k_hi`` x,
    dilated by one block, bilinearly upsampled. Tuned 11-08 on the recorded camera session
    (CR 96, 10 dB, 1500 frames, camera frame as reference): flicker in static areas
    3.29 -> 0.95 levels/frame (camera itself 0.31), PSNR +0.52 dB, moving areas -0.11 dB.

    ``alpha_snr`` (lo, hi): when a frame comes with its SNR, alpha_min is lo at <= ``snr_range[0]``
    dB and hi at >= ``snr_range[1]`` dB, linear in between, so a frame received at low SNR leans
    more on the previous output. 11-09, Tiny video model on two live RF SNR traces (720p, 192
    frames, static + camera): (0.1, 0.25) beat the fixed 0.2 on all four sequences, flicker
    -3..-6 %, p95 -8..-16 %, LPIPS -0.001..-0.002, PSNR +0.04..+0.05 dB
    (results/deploy/video_snr_policy_20261109). Without an SNR, ``alpha_min`` applies.
    """

    def __init__(self, alpha_min: float = 0.2, k_lo: float = 3.0, k_hi: float = 8.0, block: int = 8,
                 floor: float = 0.5, cap: float = 4.0, alpha_snr: tuple[float, float] | None = None,
                 snr_range: tuple[float, float] = (5.0, 20.0)) -> None:
        self.alpha_min, self.k_lo, self.k_hi, self.block = alpha_min, k_lo, k_hi, block
        self.floor, self.cap = floor, cap
        self.alpha_snr, self.snr_range = alpha_snr, snr_range
        self.prev = self.prev_small = None

    def alpha_for(self, snr_db: float | None) -> float:
        if snr_db is None or self.alpha_snr is None:
            return self.alpha_min
        (lo, hi), (s0, s1) = self.alpha_snr, self.snr_range
        return lo + (hi - lo) * float(np.clip((snr_db - s0) / (s1 - s0), 0.0, 1.0))

    def reset(self) -> None:
        self.prev = self.prev_small = None

    def __call__(self, image: np.ndarray, snr_db: float | None = None) -> np.ndarray:
        import torch
        import torch.nn.functional as F

        with torch.inference_mode():
            cur = torch.from_numpy(np.ascontiguousarray(image)).cuda().permute(2, 0, 1)[None].float()
            small = F.avg_pool2d(cur, self.block)
            if self.prev is None or self.prev.shape != cur.shape:
                self.prev, self.prev_small = cur, small
                return image
            d = (small - self.prev_small).abs().mean(1, keepdim=True)
            noise = d.median().clamp(self.floor, self.cap)
            a = ((d - self.k_lo * noise) / ((self.k_hi - self.k_lo) * noise)).clamp(0, 1)
            a = F.max_pool2d(a, 3, stride=1, padding=1)
            alpha = self.alpha_for(snr_db)
            a = alpha + (1 - alpha) * a
            out = self.prev + F.interpolate(a, size=cur.shape[-2:], mode="bilinear", align_corners=False) \
                * (cur - self.prev)
            self.prev, self.prev_small = out, F.avg_pool2d(out, self.block)
            return out.add(0.5).clamp(0, 255).to(torch.uint8)[0].permute(1, 2, 0).contiguous().cpu().numpy()


# ------------------------------------------------------------------ RF front end

class RadioFrontEnd:
    """This host's HackRF capturing continuously, and burst recovery per frame.

    Locating a burst uses the stream sample offset: once the RX/TX sample offset of
    the sender's TX stream is known, only the predicted span (+-``--window-margin``) is read
    as soon as it has been captured, then up to ``--slip`` later (TX underruns delay the
    stream); a span holding only the first packets is re-read shifted by the delay they show.
    The stream is searched around the position at which the frame's metadata arrived only
    for the first burst of a TX stream or after ``--search-after`` empty spans in a row.
    """

    def __init__(self, args) -> None:
        import hackrf_io as io
        import ofdm_phy as phy

        self.io, self.phy, self.args = io, phy, args
        self.lo_hz = args.rf_hz - phy.OFFSET
        Path(args.out_dir).mkdir(parents=True, exist_ok=True)
        # forked reader: must start before CUDA is initialized in this process
        self.stream = io.RxStream(args.rx_serial, self.lo_hz, args.lna, args.vga,
                                  Path(args.out_dir) / "rx_hackrf.log", args.ring_seconds)
        self.stream.wait_started()
        self.offsets: dict[int, int] = {}   # TX stream id -> RX sample minus TX sample
        self.empty_streak: dict[int, int] = {}
        self.perms: dict[int, np.ndarray] = {}

    def setup_gpu(self) -> None:
        import torch
        from rf_frontend import TorchBaseband

        torch.backends.cudnn.allow_tf32 = False  # keep the decimation filter in true fp32
        self.baseband = TorchBaseband()
        self.baseband(self.stream.read(0, 1 << 16))
        if self.phy.native_kernel() is None:
            print("ofdm_phy: C++ kernel unavailable, demodulating with numpy", flush=True)

    def check(self, hello: dict) -> None:
        if hello.get("lo_hz") != self.lo_hz or hello.get("sample_rate") != self.io.RATE:
            raise ValueError(f"tuning mismatch: sender {hello.get('lo_hz')} Hz at {hello.get('sample_rate')} S/s")
        if hello.get("phy") != self.phy.__doc__.splitlines()[0]:
            raise ValueError(f"PHY mismatch: sender {hello.get('phy')!r}")

    def _demod(self, start: int, stop: int, frame: dict):
        iq = self.stream.read(start, stop)
        t0 = time.perf_counter()
        packets, report = self.phy.recover(self.baseband(iq), frame["total_packets"], frame["header_seq"],
                                           infer_ids=self.args.infer_header_errors)
        return iq, packets, 1e3 * (time.perf_counter() - t0)

    def _offset(self, frame: dict, start: int, packets: dict) -> int:
        starts = frame["waveform"]["packet_starts"]
        return int(np.median([start + self.phy.UP * item["start_baseband_sample"]
                              - (frame["tx"]["tx_stream_sample"] + starts[pid])
                              for pid, (_, item) in packets.items()]))

    def capture(self, frame: dict, p_arrival: int) -> dict:
        rate, args, tx = self.io.RATE, self.args, frame["tx"]
        sid, total = tx["tx_stream_id"], frame["total_packets"]
        offset = self.offsets.get(sid)
        margin = int(args.window_margin * rate)
        packets, window, passes, wait_ms, demod_ms, iq, start = {}, None, 0, 0.0, 0.0, None, 0
        t_ready = None

        def take(span_start: int, span_stop: int, label: str):
            nonlocal passes, wait_ms, demod_ms, t_ready, window
            waited = time.perf_counter()
            self.stream.wait_until(span_stop, timeout=3.0)
            wait_ms += 1e3 * (time.perf_counter() - waited)
            t_ready = time.time_ns()
            got = self._demod(span_start, span_stop, frame)
            passes, demod_ms, window = passes + 1, demod_ms + got[2], label
            return got

        if offset is not None:
            start = tx["tx_stream_sample"] + offset - margin
            end = start + frame["waveform"]["samples"] + 2 * margin
            for stop, label in ((end, "predicted"), (end + int(args.slip * rate), "slipped")):
                iq, packets, _ = take(start, stop, label)
                if packets:
                    break
            if packets and len(packets) < total:  # underruns since the offset was learned: shifted burst
                shift = self._offset(frame, start, packets) - offset
                if shift > margin // 2:
                    iq2, packets2, _ = take(start + shift, stop + shift, "shifted")
                    if len(packets2) > len(packets):
                        iq, packets, start = iq2, packets2, start + shift
            self.empty_streak[sid] = 0 if packets else self.empty_streak.get(sid, 0) + 1
        if offset is None or (not packets and self.empty_streak[sid] >= args.search_after):
            self.empty_streak[sid] = 0
            start = max(0, p_arrival - int(args.search_before * rate), self.stream.position()
                        - int(0.9 * args.ring_seconds * rate))
            iq, packets, _ = take(start, p_arrival + int(args.search_ahead * rate), "search")
        if packets:
            self.offsets[sid] = self._offset(frame, start, packets)
        return {"packets": packets, "window": window, "passes": passes, "rx_wait_ms": wait_ms,
                "demod_total_ms": demod_ms, "t_rx_ready": t_ready,
                "rail_fraction": float(np.mean(np.abs(iq) >= 127)) if iq is not None and len(iq) else 0.0}

    def symbols(self, frame: dict, packets: dict) -> np.ndarray:
        phy = self.phy
        seed = frame["permutation_seed"]
        if seed not in self.perms:
            self.perms[seed] = phy.permutation(seed)
        rows = phy.assemble(packets, frame["total_packets"], self.perms[seed], frame["scramble_seed"])
        received = phy.rows_to_symbols(rows, frame["layers"], frame["height"], frame["width"])
        return np.where(np.isfinite(received), received, 0.0).astype(np.float16)  # lost packets -> zeros

    def close(self) -> None:
        self.stream.close()


# ------------------------------------------------------------------ session

class Session:
    """One sender connection: read frames, decode the newest, log, acknowledge."""

    def __init__(self, app: "App", channel: Channel, hello: dict) -> None:
        self.app, self.channel, self.hello = app, channel, hello
        self.link = app.args.link
        self.models = list(hello.get("models") or [hello["model"]])
        size = (hello["height"], hello["width"])
        self.decoders = {m: app.witt[size] if m == "swinjscc" else app.video[size] if m == "video"
                         else app.receivers[(m, *size)] for m in self.models}
        if "video" in self.models:
            self.decoders["video"].reset()
        # Region of the coded frame that holds the picture (the sender pads e.g. 1080 -> 1152 rows);
        # the decoded frame is cropped to it before smoothing, display and recording.
        self.valid = (int(hello.get("valid_height", hello["height"])), int(hello.get("valid_width", hello["width"])))
        self.clock = Clock()
        self.cond = threading.Condition()
        self.pending = None
        self.closed = threading.Event()
        self.rows: list[dict] = []
        self.drops = self.lost = 0
        self.last_snr = None
        args = app.args
        self.smoother = TemporalSmoother(args.smooth_alpha, *args.smooth_k,
                                         alpha_snr=tuple(args.smooth_alpha_snr) if args.smooth_snr else None)
        self.smooth_rate = None
        self.dir = Path(app.args.out_dir) / time.strftime("%Y%m%d-%H%M%S")
        self.dir.mkdir(parents=True, exist_ok=True)
        (self.dir / "hello.json").write_text(json.dumps(hello, indent=2) + "\n")
        self.csv_file = open(self.dir / "frames.csv", "w", newline="")
        self.writer = None
        self.rng = np.random.default_rng(app.args.seed)
        self.recorder = Recorder(self.dir / "record.mp4", app) if app.args.record else None

    # -- threads
    def pinger(self) -> None:
        count = 0
        while not self.closed.is_set():
            try:
                self.channel.send(b"PING", PING.pack(time.time_ns()))
            except OSError:
                return
            count += 1
            time.sleep(0.02 if count < 30 else 0.5)

    def decoder(self) -> None:
        while True:
            with self.cond:
                self.cond.wait_for(lambda: self.pending is not None or self.closed.is_set())
                if self.pending is None:
                    return
                item, self.pending = self.pending, None
            try:
                (self.rf_frame if self.link == "rf" else self.wifi_frame)(*item)
            except (TimeoutError, ValueError, RuntimeError) as error:
                print(f"frame {item[0]['seq'] if isinstance(item[0], dict) else item[0][0]} failed: {error!r}",
                      flush=True)

    # -- per frame
    def _finish(self, image: np.ndarray, rate, snr_db: float | None = None) -> tuple[np.ndarray, bytes, int, int, int]:
        """Smooth (if on), JPEG, publish -> (bgr, jpeg, t_smooth_end, t_out, smoothing).
        ``rate`` is any key of the stream's look (CR, or (model, CR)): a change restarts the smoother."""
        smoothing = int(self.app.smoothing)
        if not smoothing or rate != self.smooth_rate:  # a new rate/model: start from its own frames
            self.smoother.reset()
            self.smooth_rate = rate
        if smoothing:
            image = self.smoother(image, snr_db)
        t_smooth = time.time_ns()
        bgr = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        ok, jpeg = cv2.imencode(".jpg", bgr, [cv2.IMWRITE_JPEG_QUALITY, self.app.args.jpeg_quality])
        jpeg = jpeg.tobytes()
        self.app.hub.publish("rx", jpeg)
        return bgr, jpeg, t_smooth, time.time_ns(), smoothing

    def _log(self, row: dict, bgr: np.ndarray | None) -> None:
        if self.writer is None:
            self.writer = csv.DictWriter(self.csv_file, fieldnames=FIELDS[self.link], extrasaction="ignore",
                                         restval="")
            self.writer.writeheader()
        self.writer.writerow(row)
        self.rows.append(row)
        if len(self.rows) % 30 == 0:
            self.csv_file.flush()
        if bgr is not None:
            self.app.stats.add(row)
            if self.recorder is not None:
                self.recorder.offer(row["seq"], bgr, row)

    def wifi_frame(self, header: tuple, symbols: np.ndarray, t_recv: int, _p=None) -> None:
        seq, cam, t_cap, t_pick, t_enc0, t_enc1, snr_db, rate, height, width, channels = header
        sigma = np.sqrt(0.5 * 10 ** (-snr_db / 10))  # unit-power complex symbols
        noise = self.rng.standard_normal(symbols.shape, dtype=np.float32)
        received = (symbols.astype(np.float32) + sigma * noise).astype(np.float16)
        t_dec0 = time.time_ns()
        image = self.decoders[self.models[0]].decode(received[None], snr_db, int(rate))[0][:self.valid[0], :self.valid[1]]
        t_dec1 = time.time_ns()
        bgr, jpeg, t_smooth, t_out, smoothing = self._finish(image, int(rate), float(snr_db))
        try:
            self.channel.send(b"ACK!", ACK.pack(seq, t_recv, t_dec0, t_dec1, t_out, 0))
        except OSError:
            pass
        offset, rtt = self.clock.best() or (0, -1)
        self._log({"seq": seq, "camera_index": cam, "lost": 0, "smoothing": smoothing, "snr_db": round(snr_db, 3),
                   "rate": int(rate),
                   "channels": channels, "payload_bytes": symbols.nbytes, "channel_uses": symbols.size // 2,
                   "wait_ms": (t_pick - t_cap) / 1e6, "prep_ms": (t_enc0 - t_pick) / 1e6,
                   "encode_ms": (t_enc1 - t_enc0) / 1e6, "network_ms": (t_recv - offset - t_enc1) / 1e6,
                   "queue_ms": (t_dec0 - t_recv) / 1e6, "decode_ms": (t_dec1 - t_dec0) / 1e6,
                   "smooth_ms": (t_smooth - t_dec1) / 1e6, "output_ms": (t_out - t_smooth) / 1e6,
                   "e2e_ms": (t_out - offset - t_cap) / 1e6,
                   "t_capture": t_cap, "t_pick": t_pick, "t_encode_start": t_enc0, "t_encode_end": t_enc1,
                   "t_recv": t_recv, "t_decode_start": t_dec0, "t_decode_end": t_dec1, "t_out": t_out,
                   "clock_offset_ms": offset / 1e6, "sync_rtt_ms": rtt / 1e6, "jpeg_bytes": len(jpeg)}, bgr)

    def rf_frame(self, frame: dict, _blob, t_recv: int, p_arrival: int) -> None:
        radio = self.app.radio
        t_work = time.time_ns()
        got = radio.capture(frame, p_arrival)
        packets, total = got["packets"], frame["total_packets"]
        offset, rtt = self.clock.best() or (0, -1)
        tx = frame["tx"]
        estimate = radio.phy.snr_estimate([item for _, item in packets.values()]) if packets else None
        if estimate is not None:
            self.last_snr = estimate
        base = {"seq": frame["seq"], "camera_index": frame["camera_index"], "model": frame.get("model", self.models[0]),
                "video_key": int(frame["video"]["key"]) if "video" in frame else "",
                "snr_db": None if estimate is None else round(estimate, 3), "rate": frame["rate"],
                "channels": frame["channels"], "packets": len(packets), "total_packets": total,
                "inferred_packets": sum(1 for _, item in packets.values() if item.get("header_crc") is False),
                "encoder_snr_db": frame["encoder_snr_db"], "tx_gain": tx.get("tx_gain"),
                "air_ms": 1e3 * frame["waveform"]["seconds"],
                "channel_uses": frame["channels"] * (frame["height"] // 16) * (frame["width"] // 16) // 2,
                "window": got["window"], "demod_passes": got["passes"], "rx_wait_ms": got["rx_wait_ms"],
                "demod_total_ms": got["demod_total_ms"], "rx_rail_fraction": got["rail_fraction"],
                "tx_underruns": tx.get("underruns"), "metadata_ms": (t_recv - offset - tx["t_write_start"]) / 1e6,
                "rx_queue_ms": (t_work - t_recv) / 1e6, "clock_offset_ms": offset / 1e6, "sync_rtt_ms": rtt / 1e6}
        if not packets:  # nothing arrived: keep showing the previous frame
            self.lost += 1
            self.app.stats.lost += 1
            self._send_ack({"seq": frame["seq"], "lost": 1, "snr_estimate_db": None, "packets": 0, "total": total})
            self._log({**base, "lost": 1}, None)
            return
        received = radio.symbols(frame, packets)
        decoder_snr = clamp_snr(estimate)
        t_demod = time.time_ns()
        model = frame.get("model", self.models[0])
        if model == "video":  # Tiny video model: keyframe/previous state on the receiver
            info = frame["video"]
            if info.get("start"):
                self.decoders[model].reset()
            image, why = self.decoders[model].decode(received, decoder_snr, int(frame["rate"]), info)
            if image is None:  # P frame without its keyframe: not shown, ask the sender for a keyframe
                self.lost += 1
                self.app.stats.lost += 1
                self._send_ack({"seq": frame["seq"], "lost": 1, "snr_estimate_db": estimate, "packets": len(packets),
                                "total": total, "need_key": True})
                self._log({**base, "lost": 1, "video_key": 0}, None)
                return
        elif model == "swinjscc":  # WITT C16 decoder: flat unit-power features [1, L*16] (zeros for lost packets)
            image = self.decoders[model](received=received.reshape(1, -1).astype(np.float16),
                                         snr_db=np.array([decoder_snr], np.float32))["image"][0]
        else:
            image = self.decoders[model].decode(received[None], decoder_snr, int(frame["rate"]))[0]
        image = image[:self.valid[0], :self.valid[1]]
        t_dec1 = time.time_ns()
        bgr, jpeg, t_smooth, t_out, smoothing = self._finish(image, (model, int(frame["rate"])), decoder_snr)
        self._send_ack({"seq": frame["seq"], "lost": 0, "snr_estimate_db": estimate, "packets": len(packets),
                        "total": total, "t_out": t_out})
        t_cap, t_write = frame["t_capture"], tx["t_write_start"]
        t_mod0 = frame.get("t_modulate_start", frame["t_encode_end"])  # older senders modulate right after encoding
        self._log({**base, "lost": 0, "smoothing": smoothing, "decoder_snr_db": decoder_snr,
                   "wait_ms": (frame["t_pick"] - t_cap) / 1e6,
                   "prep_ms": (frame["t_encode_start"] - frame["t_pick"]) / 1e6,
                   "encode_ms": (frame["t_encode_end"] - frame["t_encode_start"]) / 1e6,
                   "tx_wait_ms": (t_mod0 - frame["t_encode_end"]) / 1e6,
                   "modulate_ms": (frame["t_modulate_end"] - t_mod0) / 1e6,
                   "tx_queue_ms": (t_write - frame["t_modulate_end"]) / 1e6,
                   "radio_ms": (got["t_rx_ready"] - offset - t_write) / 1e6,
                   "demod_ms": (t_demod - got["t_rx_ready"]) / 1e6,
                   "decode_ms": (t_dec1 - t_demod) / 1e6, "smooth_ms": (t_smooth - t_dec1) / 1e6,
                   "output_ms": (t_out - t_smooth) / 1e6,
                   "e2e_ms": (t_out - offset - t_cap) / 1e6,
                   "t_capture": t_cap, "t_pick": frame["t_pick"], "t_encode_start": frame["t_encode_start"],
                   "t_encode_end": frame["t_encode_end"], "t_modulate_start": t_mod0,
                   "t_modulate_end": frame["t_modulate_end"],
                   "t_tx_start": t_write, "t_recv": t_recv, "t_work": t_work, "t_rx_ready": got["t_rx_ready"],
                   "t_demod_end": t_demod, "t_decode_end": t_dec1, "t_out": t_out, "jpeg_bytes": len(jpeg)}, bgr)

    def _send_ack(self, body: dict) -> None:
        try:
            self.channel.send_json(b"RACK", body)
        except OSError:
            pass

    def run(self) -> None:
        threads = [threading.Thread(target=self.pinger, daemon=True),
                   threading.Thread(target=self.decoder, daemon=True)]
        for thread in threads:
            thread.start()
        if self.app.desired:
            self.channel.send_json(b"CTRL", self.app.desired)
        try:
            while True:
                kind, body = self.channel.recv()
                item = None
                if kind == b"FRAM":
                    header = FRAME.unpack_from(body)
                    height, width, channels = header[8:11]
                    symbols = np.frombuffer(body, np.float16, offset=FRAME.size).reshape(
                        channels, height // 16, width // 16)
                    item = (header, symbols, time.time_ns(), None)
                elif kind == b"RFFR":
                    t_recv = time.time_ns()
                    item = (json.loads(body), None, t_recv, self.app.radio.stream.position())
                elif kind == b"PONG":
                    t_a, t_b = PONG.unpack(body)
                    self.clock.add(t_a, t_b, time.time_ns())
                elif kind == b"BYE!":
                    break
                if item is not None:
                    with self.cond:
                        dropped, self.pending = self.pending, item
                        self.cond.notify()
                    if dropped is not None:
                        self.drops += 1
                        self.app.stats.drops += 1
                        if kind == b"FRAM":
                            self.channel.send(b"ACK!", ACK.pack(dropped[0][0], dropped[2], 0, 0, 0, 1))
                        else:
                            self._send_ack({"seq": dropped[0]["seq"], "lost": 0, "dropped": 1,
                                            "snr_estimate_db": None})
        except (ConnectionError, OSError) as error:
            print(f"connection ended: {error}", flush=True)
        finally:
            self.closed.set()
            with self.cond:
                self.cond.notify_all()
            threads[1].join(timeout=3)
            self.channel.close()
            self.finish()

    def finish(self) -> None:
        self.csv_file.close()
        if self.recorder is not None:
            self.recorder.close()
        shown = [r for r in self.rows if not r["lost"]]
        if not shown:
            return
        shown = shown[1:] if len(shown) > 10 else shown  # first frame pays graph/socket/search warm-up
        stages = STAGES[self.link]

        def block(subset, all_rows) -> dict:
            out = {"frames_shown": len(subset), "frames_lost": sum(r["lost"] for r in all_rows),
                   "fps": fps_of(r["t_out"] for r in subset),
                   "camera_fps": (subset[-1]["camera_index"] - subset[0]["camera_index"])
                   / max((subset[-1]["t_capture"] - subset[0]["t_capture"]) / 1e9, 1e-9),
                   **{f"{k}_ms": describe(r[f"{k}_ms"] for r in subset) for k in (*stages, "e2e")}}
            if self.link == "rf":
                out.update(snr_db=describe(r["snr_db"] for r in subset),
                           packet_loss=1 - sum(r["packets"] for r in all_rows)
                           / max(sum(r["total_packets"] for r in all_rows), 1),
                           rx_wait_ms=describe(r["rx_wait_ms"] for r in subset),
                           windows={w: sum(r["window"] == w for r in all_rows)
                                    for w in {r["window"] for r in all_rows}},
                           tx_underruns=max((r["tx_underruns"] or 0) for r in all_rows),
                           rx_rail_fraction_max=max(r["rx_rail_fraction"] for r in all_rows))
            else:
                out["payload_bytes"] = subset[-1]["payload_bytes"]
            return out

        key = (lambda r: (r["snr_db"], r["rate"])) if self.link == "wifi" else (lambda r: r["rate"])
        segments, current = [], []
        for row in self.rows[1:]:  # contiguous runs of one setting
            if current and key(row) != key(current[-1]):
                segments.append(current)
                current = []
            current.append(row)
        segments.append(current)
        summary = {"hello": self.hello, "link": self.link, "model": self.hello["model"],
                   "drops_at_receiver": self.drops,
                   "camera_frames_skipped": shown[-1]["camera_index"] - shown[0]["camera_index"] + 1
                   - len([r for r in self.rows if r["camera_index"] >= shown[0]["camera_index"]]),
                   "duration_s": (shown[-1]["t_out"] - shown[0]["t_out"]) / 1e9,
                   "clock_offset_ms": shown[-1]["clock_offset_ms"], "sync_rtt_ms": shown[-1]["sync_rtt_ms"],
                   "overall": block(shown, self.rows),
                   "segments": [{"rate": s[0]["rate"], **({"snr_db": s[0]["snr_db"]} if self.link == "wifi" else {}),
                                 **block([r for r in s if not r["lost"]], s)}
                                for s in segments if len([r for r in s if not r["lost"]]) >= 3]}
        (self.dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
        o = summary["overall"]
        snr = f", SNR median {o['snr_db']['median']:.1f} dB, packet loss {100 * o['packet_loss']:.2f} %" \
            if self.link == "rf" and o.get("snr_db") else ""
        print(f"session {self.dir.name}: {o['frames_shown']} frames shown ({o['frames_lost']} lost) in "
              f"{summary['duration_s']:.1f} s, {o['fps']:.1f} fps, e2e median {o['e2e_ms']['median']:.1f} ms "
              f"(p95 {o['e2e_ms']['p95']:.1f}){snr} -> {self.dir}", flush=True)


class Recorder:
    """Side-by-side mp4 (camera preview | received) with a one-line overlay, in its own thread."""

    def __init__(self, path: Path, app: "App") -> None:
        self.path, self.app, self.writer = path, app, None
        self.queue: deque = deque(maxlen=8)
        self.cond = threading.Condition()
        self.done = False
        self.thread = threading.Thread(target=self._run, daemon=True)
        self.thread.start()

    def offer(self, seq: int, bgr: np.ndarray, row: dict) -> None:
        with self.cond:
            self.queue.append((seq, bgr, row))
            self.cond.notify()

    def _run(self) -> None:
        while True:
            with self.cond:
                self.cond.wait_for(lambda: self.queue or self.done)
                if not self.queue:
                    return
                seq, bgr, row = self.queue.popleft()
            jpeg = self.app.previews.get(seq)
            left = cv2.imdecode(np.frombuffer(jpeg, np.uint8), cv2.IMREAD_COLOR) if jpeg else np.zeros_like(bgr)
            frame = np.concatenate([left, bgr], axis=1)
            link = "2.38 GHz HackRF, measured" if self.app.args.link == "rf" else "simulated AWGN"
            link += ", smoothed" if row.get("smoothing") else ""
            text = (f"CR {row['rate']}  SNR {row['snr_db']:.1f} dB ({link})  {self.app.stats.fps():.1f} fps  "
                    f"e2e {row['e2e_ms']:.0f} ms")
            cv2.putText(frame, text, (12, frame.shape[0] - 14), cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                        (255, 255, 255), 2, cv2.LINE_AA)
            if self.writer is None:
                self.writer = cv2.VideoWriter(str(self.path), cv2.VideoWriter_fourcc(*"mp4v"), 30.0,
                                              (frame.shape[1], frame.shape[0]))
            self.writer.write(frame)

    def close(self) -> None:
        with self.cond:
            self.done = True
            self.cond.notify()
        self.thread.join(timeout=5)
        if self.writer is not None:
            self.writer.release()


class DashboardRecorder:
    """Record button: runs ``live_record.py`` (headless Chromium + ffmpeg) as a child process."""

    def __init__(self, app: "App") -> None:
        self.app, self.proc, self.lock = app, None, threading.Lock()
        self.status, self.started, self.file, self.last = "idle", 0.0, None, None
        self.dir = Path(app.args.out_dir) / "recordings"

    def start(self, theme: str = "dark") -> None:
        with self.lock:
            if self.proc is not None:
                return
            self.dir.mkdir(parents=True, exist_ok=True)
            self.file = self.dir / f"dashboard_{time.strftime('%Y%m%d-%H%M%S')}.mp4"
            uv = shutil.which("uv") or str(Path.home() / ".local/bin/uv")
            command = [uv, "run", "--no-project", "--with", "playwright", "--with", "imageio-ffmpeg", "python",
                       str(HERE / "live_record.py"), "--url", f"http://127.0.0.1:{self.app.args.http_port}/",
                       "--out", str(self.file), "--theme", theme if theme in ("dark", "light") else "dark"]
            self.proc = subprocess.Popen(command, stdin=subprocess.PIPE, stdout=subprocess.PIPE, text=True,
                                         start_new_session=True)  # Ctrl-C of the receiver is ours to forward
            self.status, self.started = "starting", time.time()
            threading.Thread(target=self._read, args=(self.proc,), daemon=True).start()
        print(f"dashboard recording -> {self.file}", flush=True)

    def _read(self, proc) -> None:
        result = None
        for line in proc.stdout:
            try:
                message = json.loads(line)
            except ValueError:
                continue
            if message.get("status") == "recording":
                self.status, self.started = "recording", time.time()
            else:
                result = message
        proc.wait()
        with self.lock:
            if result is None or result.get("status") != "done" or not self.file.exists():
                self.last = {"error": f"recorder exited ({proc.returncode})"}
            else:
                self.last = {"name": self.file.name, "url": f"/recordings/{self.file.name}",
                             "path": str(self.file), "seconds": result["seconds"], "frames": result["frames"],
                             "screencast_frames": result.get("screencast_frames"),
                             "mb": round(result["bytes"] / 1e6, 1), "size": result["size"]}
                print(f"dashboard recording saved: {self.file} ({self.last['seconds']:.0f} s, "
                      f"{self.last['mb']} MB)", flush=True)
            self.proc, self.status = None, "idle"

    def stop(self) -> None:
        with self.lock:
            proc = self.proc
            if proc is None or self.status == "finishing":
                return
            self.status = "finishing"
        try:
            proc.stdin.write("stop\n")
            proc.stdin.close()
        except (BrokenPipeError, OSError):
            proc.terminate()

    def wait(self, timeout: float = 20.0) -> None:
        proc = self.proc
        if proc is not None:
            try:
                proc.wait(timeout=timeout)
            except subprocess.TimeoutExpired:
                proc.kill()

    def state(self) -> dict:
        return {"status": self.status, "last": self.last,
                "seconds": time.time() - self.started if self.status == "recording" else 0.0}


class Stats:
    """Rolling numbers for the web page and the console."""

    def __init__(self, link: str) -> None:
        self.link, self.stages, self.groups = link, STAGES[link], GROUPS[link]
        self.rows: deque = deque(maxlen=600)
        self.lock = threading.Lock()
        self.drops = self.lost = 0
        self.lost_times: deque = deque(maxlen=600)

    def add(self, row: dict) -> None:
        with self.lock:
            self.rows.append(row)

    def recent(self, seconds: float = 2.0) -> list[dict]:
        now = time.time_ns()
        with self.lock:
            return [r for r in self.rows if now - r["t_out"] < seconds * 1e9]

    def fps(self) -> float:
        return fps_of(r["t_out"] for r in self.recent())

    def snapshot(self) -> dict:
        recent = self.recent()
        with self.lock:
            history = list(self.rows)[-150:]
        base = {"link": self.link, "groups": [g for g, _ in self.groups], "drops": self.drops, "lost": self.lost}
        if not recent:
            return {**base, "live": False, "history": []}
        last = recent[-1]
        span = (recent[-1]["t_capture"] - recent[0]["t_capture"]) / 1e9
        stages = {k: pct([r[f"{k}_ms"] for r in recent], 0.5) for k in self.stages}

        def grouped(values):
            return [round(sum(max(0.0, values[k]) for k in keys), 2) for _, keys in self.groups]

        out = {
            **base, "live": True, "fps": fps_of(r["t_out"] for r in recent),
            "camera_fps": (recent[-1]["camera_index"] - recent[0]["camera_index"]) / span if span > 0 else 0.0,
            "e2e_median": pct([r["e2e_ms"] for r in recent], 0.5),
            "e2e_p95": pct([r["e2e_ms"] for r in recent], 0.95),
            "stages": stages, "group_values": grouped(stages),
            "snr_db": pct([r["snr_db"] for r in recent], 0.5), "rate": last["rate"], "model": last.get("model"),
            "channels": last["channels"], "channel_uses": last["channel_uses"],
            "sync_rtt_ms": last["sync_rtt_ms"], "seq": last["seq"],
            "history": [[r["seq"], *grouped({k: r[f"{k}_ms"] for k in self.stages}), round(r["e2e_ms"], 2)]
                        for r in history],
        }
        if self.link == "rf":
            out.update(packets=sum(r["packets"] for r in recent), total_packets=sum(r["total_packets"] for r in recent),
                       packets_per_frame=last["total_packets"], air_ms=last["air_ms"], tx_gain=last["tx_gain"],
                       encoder_snr_db=last["encoder_snr_db"], decoder_snr_db=last["decoder_snr_db"])
        else:
            out.update(payload_bytes=last["payload_bytes"],
                       mbps=8 * last["payload_bytes"] * out["fps"] / 1e6)
        return out


# ------------------------------------------------------------------ web

class Handler(BaseHTTPRequestHandler):
    app: "App"

    def log_message(self, *_):
        pass

    def _send(self, body: bytes, content_type: str, code: int = 200) -> None:
        self.send_response(code)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self) -> None:
        path = self.path.split("?")[0]
        try:
            if path == "/":
                self._send((HERE / "live_viewer.html").read_bytes(), "text/html; charset=utf-8")
            elif path in ("/rx.mjpg", "/tx.mjpg"):
                self._mjpeg(path[1:3])
            elif path in ("/rx.jpg", "/tx.jpg"):
                item = self.app.hub.items.get(path[1:3])
                self._send(item[1], "image/jpeg") if item else self._send(b"no frame", "text/plain", 404)
            elif path == "/events":
                self._events()
            elif path.startswith("/recordings/"):
                file = self.app.recorder.dir / Path(path).name  # no directories: only this folder
                if file.suffix == ".mp4" and file.is_file():
                    self.send_response(200)
                    self.send_header("Content-Type", "video/mp4")
                    self.send_header("Content-Length", str(file.stat().st_size))
                    self.send_header("Content-Disposition", f'attachment; filename="{file.name}"')
                    self.end_headers()
                    with open(file, "rb") as handle:
                        shutil.copyfileobj(handle, self.wfile)
                else:
                    self._send(b"not found", "text/plain", 404)
            elif path == "/state":
                self._send(json.dumps(self.app.state()).encode(), "application/json")
            else:
                self._send(b"not found", "text/plain", 404)
        except (BrokenPipeError, ConnectionResetError):
            pass

    def do_POST(self) -> None:
        if self.path not in ("/control", "/record"):
            return self._send(b"not found", "text/plain", 404)
        body = json.loads(self.rfile.read(int(self.headers.get("Content-Length", 0))) or b"{}")
        if self.path == "/record":
            if body.get("action") == "start":
                self.app.recorder.start(body.get("theme", "dark"))
            elif body.get("action") == "stop":
                self.app.recorder.stop()
            return self._send(json.dumps(self.app.state()).encode(), "application/json")
        control = {}
        if "snr_db" in body and self.app.args.link == "wifi":  # over the air the SNR is measured, not set
            control["snr_db"] = max(-10.0, min(30.0, float(body["snr_db"])))
        if "smoothing" in body:  # display-side only: not sent to the sender
            self.app.smoothing = bool(body["smoothing"])
        if "rate" in body and int(body["rate"]) in self.app.rates:
            control["rate"] = int(body["rate"])
        session = self.app.session
        if "model" in body and session is not None and body["model"] in session.models:
            control["model"] = body["model"]
        self.app.control(control)
        self._send(json.dumps(self.app.state()).encode(), "application/json")

    def _mjpeg(self, name: str) -> None:
        self.send_response(200)
        self.send_header("Content-Type", "multipart/x-mixed-replace; boundary=frame")
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(b"--frame\r\n")
        last = 0
        while not self.app.stopping.is_set():
            item = self.app.hub.wait(name, last, timeout=1.0)
            if item is None or item[0] == last:
                continue
            last, jpeg = item
            # The closing boundary right after the JPEG lets the browser show it at once.
            self.wfile.write(b"Content-Type: image/jpeg\r\nContent-Length: %d\r\n\r\n" % len(jpeg)
                             + jpeg + b"\r\n--frame\r\n")
            self.wfile.flush()

    def _events(self) -> None:
        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream")
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        while not self.app.stopping.is_set():
            payload = {**self.app.state(), **self.app.stats.snapshot()}
            self.wfile.write(b"data: " + json.dumps(payload).encode() + b"\n\n")
            self.wfile.flush()
            time.sleep(0.2)


# ------------------------------------------------------------------ app

class App:
    def __init__(self, args) -> None:
        self.args = args
        self.hub, self.stats = Hub(), Stats(args.link)
        self.stopping = threading.Event()
        self.session: Session | None = None
        self.desired: dict = {}
        self.previews: dict[int, bytes] = {}
        self.recorder = DashboardRecorder(self)
        self.smoothing = bool(args.smooth)
        self.radio = RadioFrontEnd(args) if args.link == "rf" else None  # before CUDA (forked RX reader)
        meta = json.loads((REPO_ROOT / "runs/deploy/osd_jscc_p/meta.json").read_text())
        self.rates = [int(r) for r in meta["rates"]]
        from receiver import TRTReceiver

        self.receivers = {}  # (model, H, W) -> TRTReceiver; the sender's HELO picks one
        for model in args.models:
            for size in args.sizes:
                height, width = map(int, size.lower().split("x"))
                rx = TRTReceiver(meta, args.precision, height, width, vae_suffix="_tiny" if model == "tiny" else "")
                dummy = np.zeros((1, meta["c_max"], height // 16, width // 16), np.float16)
                for _ in range(10):  # warm-up + CUDA graph capture
                    rx.decode(dummy, 10.0, 96)
                self.receivers[(model, height, width)] = rx
        self.witt = {}  # (H, W) -> SwinJSCC (WITT C16) TensorRT decoder, for senders that switch models
        swin_root = REPO_ROOT / "runs/deploy/swinjscc/engines/rtx6000"
        for size in args.sizes:
            height, width = map(int, size.lower().split("x"))
            path = swin_root / f"witt_decode_{height}x{width}_fp16.plan"
            if path.exists():
                from rf_frontend import TRTGraph

                graph = TRTGraph(path)
                for _ in range(3):
                    graph(received=np.zeros((1, 16 * (height // 16) * (width // 16)), np.float16),
                          snr_db=np.array([10.0], np.float32))
                self.witt[(height, width)] = graph
        self.video = {}  # (H, W) -> Tiny video model receiver (video_codec.VideoReceiver), for --models ... video
        for size in args.sizes:
            height, width = map(int, size.lower().split("x"))
            if (REPO_ROOT / "runs/deploy/tiny_video/engines/rtx6000" / f"video_rx_jscc_{height}x{width}_fp16.plan").exists():
                from video_codec import VideoReceiver

                self.video[(height, width)] = VideoReceiver(height, width)
        if self.radio is not None:
            self.radio.setup_gpu()
        print(f"decoders ready: {list(self.receivers)} + SwinJSCC {list(self.witt)} + video {list(self.video)} "
              f"({args.precision}); link {args.link}"
              + (f", RX {args.rx_serial[-8:]} at {self.radio.lo_hz / 1e6:.4f} MHz (LNA {args.lna}, VGA {args.vga})"
                 if self.radio else ""), flush=True)

    def state(self) -> dict:
        session = self.session
        directory = None
        if session is not None:
            directory = str(session.dir.resolve().relative_to(REPO_ROOT)) \
                if session.dir.resolve().is_relative_to(REPO_ROOT) else str(session.dir)
        return {"connected": session is not None and not session.closed.is_set(),
                "hello": session.hello if session else None, "desired": self.desired,
                "session_dir": directory, "frames_logged": len(session.rows) if session else 0,
                "rates": self.rates, "models": sorted({m for m, _, _ in self.receivers}), "link": self.args.link,
                "rf_mhz": self.args.rf_hz / 1e6 if self.args.link == "rf" else None,
                "recording": self.recorder.state(), "smoothing": self.smoothing}

    def control(self, control: dict) -> None:
        self.desired.update(control)
        session = self.session
        if session is not None and not session.closed.is_set() and control:
            try:
                session.channel.send_json(b"CTRL", control)
            except OSError:
                pass

    def preview_server(self) -> None:
        server = socket.create_server(("0.0.0.0", self.args.port + 1), reuse_port=True)
        while True:
            conn, _ = server.accept()
            threading.Thread(target=self._preview, args=(Channel(conn),), daemon=True).start()

    def _preview(self, channel: Channel) -> None:
        try:
            while True:
                kind, body = channel.recv()
                if kind == b"PREV":
                    seq, _ = PREVIEW.unpack_from(body)
                    jpeg = body[PREVIEW.size:]
                    self.hub.publish("tx", jpeg)
                    if self.args.record:
                        self.previews[seq] = jpeg
                        self.previews.pop(seq - 60, None)
        except (ConnectionError, OSError):
            channel.close()

    def console(self) -> None:
        while not self.stopping.is_set():
            time.sleep(1.0)
            s = self.stats.snapshot()
            if s.get("live"):
                stages = " ".join(f"{k} {v:.1f}" for k, v in s["stages"].items())
                link = (f"SNR {s['snr_db']:.1f} dB (measured) | packets {s['packets']}/{s['total_packets']} "
                        f"| lost frames {s['lost']}") if s["link"] == "rf" else f"SNR {s['snr_db']:g} dB"
                print(f"rx {s['fps']:5.1f} fps (camera {s['camera_fps']:4.1f}) | e2e {s['e2e_median']:5.1f} ms "
                      f"p95 {s['e2e_p95']:5.1f} | {stages} | CR {s['rate']} {link} | drops {s['drops']}",
                      flush=True)

    def serve(self) -> None:
        Handler.app = self
        http = ThreadingHTTPServer(("0.0.0.0", self.args.http_port), Handler)
        http.daemon_threads = True
        for target in (http.serve_forever, self.preview_server, self.console):
            threading.Thread(target=target, daemon=True).start()
        server = socket.create_server(("0.0.0.0", self.args.port), reuse_port=True)
        print(f"stream on :{self.args.port} (preview :{self.args.port + 1}), "
              f"viewer http://{socket.gethostname()}:{self.args.http_port}", flush=True)
        while True:
            conn, address = server.accept()
            channel = Channel(conn)
            kind, body = channel.recv()
            if kind != b"HELO":
                channel.close()
                continue
            hello = json.loads(body)
            problem = None
            size = (hello.get("height"), hello.get("width"))
            missing = [m for m in (hello.get("models") or [hello.get("model")])
                       if (size not in self.witt if m == "swinjscc" else size not in self.video if m == "video"
                           else (m, *size) not in self.receivers)]
            if missing:
                problem = (f"no decoder for {missing} at {size[0]}x{size[1]} "
                           f"(--models {self.args.models} --sizes {self.args.sizes}; SwinJSCC engines in runs/deploy/swinjscc)")
            elif hello.get("link", "wifi") != self.args.link:
                problem = f"sender link {hello.get('link', 'wifi')!r}, receiver link {self.args.link!r}"
            elif self.radio is not None:
                try:
                    self.radio.check(hello)
                except ValueError as error:
                    problem = str(error)
            if problem:
                print(f"refused {address}: {problem}", flush=True)
                channel.close()
                continue
            print(f"sender {address[0]} ({hello.get('sender')}): {hello}", flush=True)
            self.session = Session(self, channel, hello)
            self.session.run()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--link", choices=("wifi", "rf"), default="wifi")
    parser.add_argument("--port", type=int, default=DEFAULT_PORT, help="stream port; preview on port + 1")
    parser.add_argument("--http-port", type=int, default=8777)
    parser.add_argument("--models", nargs="+", choices=("tiny", "osd"), default=["tiny"])
    parser.add_argument("--precision", default="fp16")
    parser.add_argument("--sizes", nargs="+", default=["512x768"], metavar="HxW",
                        help="coded frame sizes to load engines for (e.g. 512x768 1152x1920 for 1080p)")
    parser.add_argument("--jpeg-quality", type=int, default=85)
    parser.add_argument("--record", action="store_true", help="also write a side-by-side record.mp4")
    parser.add_argument("--seed", type=int, default=0, help="Wi-Fi link: simulated channel noise seed")
    parser.add_argument("--smooth", action=argparse.BooleanOptionalAction, default=True,
                        help="temporal smoothing of the displayed frames at start (switch on the page)")
    parser.add_argument("--smooth-alpha", type=float, default=0.2,
                        help="smoothing: weight of a static pixel's new value (with --no-smooth-snr)")
    parser.add_argument("--smooth-snr", action=argparse.BooleanOptionalAction, default=True,
                        help="smoothing weight from each frame's (decoder) SNR, see --smooth-alpha-snr")
    parser.add_argument("--smooth-alpha-snr", type=float, nargs=2, default=(0.1, 0.25), metavar=("LOW", "HIGH"),
                        help="static-pixel weight at <= 5 dB and >= 20 dB, linear in between")
    parser.add_argument("--smooth-k", type=float, nargs=2, default=(3.0, 8.0), metavar=("LO", "HI"),
                        help="smoothing: change (x flicker level) where the weight starts rising / reaches 1")
    rf = parser.add_argument_group("--link rf")
    rf.add_argument("--rx-serial", help="HackRF RX serial from hackrf_info (required for RF)")
    rf.add_argument("--infer-header-errors", action=argparse.BooleanOptionalAction, default=True,
                    help="keep packets whose header failed the CRC when their slot on the burst grid and their "
                         "decoded header bits identify them (ofdm_phy.recover infer_ids); logged as inferred_packets")
    rf.add_argument("--rf-hz", type=int, default=2_380_000_000, help="band centre (= LO, DC null carrier)")
    rf.add_argument("--lna", type=int, default=40, help="RX LNA dB (0-40, step 8; default: maximum)")
    rf.add_argument("--vga", type=int, default=20)
    rf.add_argument("--ring-seconds", type=float, default=8.0)
    rf.add_argument("--window-margin", type=float, default=0.0002, help="+- seconds around the predicted burst")
    rf.add_argument("--slip", type=float, default=0.03, help="look this much later if the predicted span is empty")
    rf.add_argument("--search-after", type=int, default=3, help="search after this many empty spans in a row")
    rf.add_argument("--search-before", type=float, default=0.12,
                    help="search: seconds before the stream position at which the frame's metadata arrived")
    rf.add_argument("--search-ahead", type=float, default=0.08, help="search: seconds after that position")
    parser.add_argument("--out-dir", default=str(REPO_ROOT / "results/deploy/live"))
    args = parser.parse_args()
    if args.link == "rf" and not args.rx_serial:
        parser.error("--rx-serial is required for --link rf")

    def stop(*_):
        signal.signal(signal.SIGTERM, signal.SIG_IGN)  # `uv run` forwards a second SIGTERM
        raise KeyboardInterrupt

    app = App(args)
    signal.signal(signal.SIGTERM, stop)  # after App: the forked RX reader keeps the default handler
    try:
        app.serve()
    except KeyboardInterrupt:
        app.stopping.set()
        app.recorder.stop()
        app.recorder.wait()
        if app.session is not None and not app.session.closed.is_set():
            app.session.closed.set()
            app.session.channel.close()
            app.session.finish()
    finally:
        if app.radio is not None:
            app.radio.close()


if __name__ == "__main__":
    main()
