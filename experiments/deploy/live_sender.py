"""Live sender: Jetson camera -> TensorRT encoder -> HackRF or TCP.

Run from the repository root. See README.md for engine setup and commands.
In RF mode the learned symbols travel over HackRF; TCP carries control,
feedback, timestamps and an optional reference preview.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import queue
import socket
import statistics
import sys
import threading
import time

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from live_link import ACK, DEFAULT_PORT, FRAME, PING, PONG, PREVIEW, Channel  # noqa: E402

ROOT = Path("runs/deploy/osd_jscc_p")
SWIN_ROOT = Path("runs/deploy/swinjscc")  # SwinJSCC baseline = WITT C16 (export_baselines.py): fixed CR 96
SWIN_CHANNELS = 16
SNR_RANGE = (0.0, 20.0)  # training range of the encoder's SNR conditioning
KODAK = Path("dataset/Kodak")
HEIGHT, WIDTH = 512, 768


# ------------------------------------------------------------------ sources

class LatestFrame:
    """Single-slot frame buffer: producers overwrite, the consumer waits for a newer one."""

    def __init__(self) -> None:
        self.cond = threading.Condition()
        self.frame, self.t_capture, self.index = None, 0, 0
        self.stop = threading.Event()
        self.times: list[int] = []

    def put(self, frame: np.ndarray, t_capture: int) -> None:
        with self.cond:
            self.frame, self.t_capture = frame, t_capture
            self.index += 1
            self.times.append(t_capture)
            self.cond.notify_all()

    def newer(self, index: int, timeout: float):
        with self.cond:
            if not self.cond.wait_for(lambda: self.index > index or self.stop.is_set(), timeout):
                return None
            if self.stop.is_set():
                return None
            return self.frame, self.t_capture, self.index

    def fps(self, window_s: float = 2.0) -> float:
        now = time.time_ns()
        recent = [t for t in self.times[-200:] if now - t < window_s * 1e9]
        return (len(recent) - 1) / ((recent[-1] - recent[0]) / 1e9) if len(recent) > 2 else 0.0


class Camera(LatestFrame):
    def __init__(self, index: int | str, width: int, height: int, fps: float) -> None:
        super().__init__()
        if sys.platform == "darwin":
            self.cap = cv2.VideoCapture(index, cv2.CAP_AVFOUNDATION)
        else:  # Linux / Jetson: V4L2 with MJPG (USB cameras give full rate only compressed)
            self.cap = cv2.VideoCapture(index, cv2.CAP_V4L2)
            self.cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
            self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
            # Raw MJPEG + cv2.imdecode: OpenCV's built-in V4L2 conversion managed only 21.5 fps of the
            # DJI's 30 on the AGX (11-09); imdecode of 1280x720 takes 8.3 ms.
            self.cap.set(cv2.CAP_PROP_CONVERT_RGB, 0)
        if not self.cap.isOpened():
            raise SystemExit(f"Camera {index!r} not available. On macOS the camera is granted per app: run this "
                             "in Terminal.app (not over ssh) and allow the camera prompt.")
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
        self.cap.set(cv2.CAP_PROP_FPS, fps)
        self.settings = {"camera_index": index,
                         "camera_width": int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
                         "camera_height": int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
                         "camera_fps": float(self.cap.get(cv2.CAP_PROP_FPS))}
        threading.Thread(target=self._run, daemon=True).start()

    def _run(self) -> None:
        while not self.stop.is_set():
            ok, frame = self.cap.read()
            t_capture = time.time_ns()
            if ok and (frame.ndim != 3 or frame.shape[0] == 1):  # raw MJPEG bytes (Linux)
                frame = cv2.imdecode(frame.reshape(-1), cv2.IMREAD_COLOR)
                ok = frame is not None
            if ok:
                self.put(frame, t_capture)
            else:
                time.sleep(0.005)
        self.cap.release()


class Playback(LatestFrame):
    """Kodak images or a video file, paced at ``fps``; each image is shown ``hold`` frames."""

    def __init__(self, spec: str, fps: float, hold: int) -> None:
        super().__init__()
        if spec == "kodak":
            frames = [cv2.imread(str(p)) for p in sorted(KODAK.glob("kodim*.png"))]
            self.frames = [f for f in frames if f is not None and f.shape[1] > f.shape[0]]
            self.video = None
        else:
            self.video, self.frames = cv2.VideoCapture(spec.removeprefix("video:")), []
            fps = fps or self.video.get(cv2.CAP_PROP_FPS) or 30.0
        if not self.frames and self.video is None:
            raise SystemExit(f"No frames in {spec}.")
        self.period, self.hold = 1.0 / fps, hold
        self.settings = {"playback": spec, "playback_fps": fps, "hold": hold}
        threading.Thread(target=self._run, daemon=True).start()

    def _next(self, count: int) -> np.ndarray:
        if self.video is None:
            return self.frames[(count // self.hold) % len(self.frames)]
        ok, frame = self.video.read()
        if not ok:
            self.video.set(cv2.CAP_PROP_POS_FRAMES, 0)
            ok, frame = self.video.read()
        return frame

    def _run(self) -> None:
        start, count = time.perf_counter(), 0
        while not self.stop.is_set():
            frame = self._next(count)
            delay = start + count * self.period - time.perf_counter()
            if delay > 0:
                time.sleep(delay)
            self.put(frame, time.time_ns())
            count += 1


def fit_size(h: int, w: int) -> tuple[int, int]:
    """--fit pad: the picture size inside the HEIGHTxWIDTH coded frame (aspect kept)."""
    scale = min(WIDTH / w, HEIGHT / h)
    return min(HEIGHT, round(h * scale)), min(WIDTH, round(w * scale))


def prepare(frame: np.ndarray, mirror: bool, fit: str = "crop") -> np.ndarray:
    """crop: center-crop to the coded aspect, resize to HEIGHTxWIDTH (BGR uint8).
    pad: resize to fit (aspect kept), reflect-pad bottom/right to HEIGHTxWIDTH (e.g. 1080p -> 1152x1920)."""
    if fit == "pad":
        vh, vw = fit_size(*frame.shape[:2])
        if frame.shape[:2] != (vh, vw):
            shrink = vh < frame.shape[0]
            frame = cv2.resize(frame, (vw, vh), interpolation=cv2.INTER_AREA if shrink else cv2.INTER_LINEAR)
        if mirror:
            frame = cv2.flip(frame, 1)
        return cv2.copyMakeBorder(frame, 0, HEIGHT - vh, 0, WIDTH - vw, cv2.BORDER_REFLECT_101)
    h, w = frame.shape[:2]
    if w * HEIGHT > h * WIDTH:  # too wide
        crop = h * WIDTH // HEIGHT
        frame = frame[:, (w - crop) // 2:(w - crop) // 2 + crop]
    elif w * HEIGHT < h * WIDTH:
        crop = w * HEIGHT // WIDTH
        frame = frame[(h - crop) // 2:(h - crop) // 2 + crop]
    if frame.shape[:2] != (HEIGHT, WIDTH):
        frame = cv2.resize(frame, (WIDTH, HEIGHT), interpolation=cv2.INTER_AREA)
    return cv2.flip(frame, 1) if mirror else frame


# ------------------------------------------------------------------ sender

class Preview:
    """JPEG copies of the encoded frames on a second socket (display only, newest wins)."""

    def __init__(self, host: str, port: int, quality: int) -> None:
        self.channel = Channel(socket.create_connection((host, port), timeout=10))
        self.channel.sock.settimeout(None)
        self.quality, self.queue = quality, queue.Queue(maxsize=1)
        threading.Thread(target=self._run, daemon=True).start()

    def offer(self, seq: int, t_capture: int, bgr: np.ndarray) -> None:
        try:
            self.queue.get_nowait()
        except queue.Empty:
            pass
        self.queue.put_nowait((seq, t_capture, bgr))

    def _run(self) -> None:
        while True:
            seq, t_capture, bgr = self.queue.get()
            ok, jpeg = cv2.imencode(".jpg", bgr, [cv2.IMWRITE_JPEG_QUALITY, self.quality])
            try:
                self.channel.send(b"PREV", PREVIEW.pack(seq, t_capture), jpeg.tobytes())
            except OSError:
                return


class RadioTx:
    """This host's HackRF as a continuous TX stream; one OFDM burst per frame."""

    def __init__(self, args, meta: dict) -> None:
        import hackrf_io as io
        import ofdm_phy as phy

        self.io, self.phy, self.args = io, phy, args
        if phy.native_kernel() is None:  # compile/load the C++ modulator before the first burst
            print("ofdm_phy: C++ kernel unavailable, modulating with numpy", flush=True)
        self.lo_hz = args.rf_hz - phy.OFFSET
        self.layers = {int(r): [tuple(c) for c in meta["layer_channels"] if c[1] <= int(ch)]
                       for r, ch in meta["sent_channels"].items()}
        log = Path(args.log_dir) / "tx_hackrf.log"
        self.stream = io.open_tx_stream(args.tx_serial, self.lo_hz, args.tx_gain, log, (0, 1),
                                        args.tx_transfer_kib, amp=args.tx_amp)
        self.first_seq = int(time.time()) % 60000
        self.hello = {"tx_serial": args.tx_serial, "rf_hz": args.rf_hz, "lo_hz": self.lo_hz,
                      "sample_rate": io.RATE, "tx_gain": args.tx_gain, "tx_amp": args.tx_amp,
                      "phy": phy.__doc__.splitlines()[0],
                      "tx_transfer_kib": args.tx_transfer_kib}

    def send(self, symbols: np.ndarray, rate: int, seq: int, layers=None) -> dict:
        phy, layers = self.phy, layers or self.layers[rate]
        rows = phy.air_rows(symbols, layers, self.args.permutation_seed, self.args.scramble_seed)
        header_seq = (self.first_seq + seq) % 65536
        iq, info = phy.modulate(rows, header_seq)
        t_mod = time.time_ns()
        tx = self.stream.send(iq)
        return {"header_seq": header_seq, "total_packets": len(rows) // phy.ROWS, "layers": layers,
                "permutation_seed": self.args.permutation_seed, "scramble_seed": self.args.scramble_seed,
                "waveform": info, "tx": tx, "t_modulate_end": t_mod}

    def close(self) -> None:
        self.stream.close()


def load_sender(args, model: str):
    """Load the OSD-JSCC-P TensorRT sender for the selected image size."""
    from agx_sender import Sender
    from trt_runtime import Cudart

    suffix = "_tiny" if model == "tiny" else ""
    name = f"sender_{HEIGHT}x{WIDTH}_{args.precision}{suffix}.plan"
    engine = ROOT / "engines" / args.engine_tag / name
    if not engine.exists():
        raise SystemExit(f"No {model} sender engine {engine}")
    return Sender(Cudart(), engine), str(engine)


def main() -> None:
    global HEIGHT, WIDTH  # --size sets the coded frame size for prepare()/fit_size()/the engine path
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--host", required=True, help="RTX receiver hostname or IP address")
    parser.add_argument("--link", choices=("wifi", "rf"), default="wifi")
    parser.add_argument("--port", type=int, default=DEFAULT_PORT, help="stream port; preview uses port + 1")
    parser.add_argument("--model", choices=("tiny", "osd"), default="tiny", help="model at start")
    parser.add_argument("--models", nargs="+", choices=("tiny", "osd"),
                        help="load these encoders; the page switches between them per frame (--link rf only)")
    parser.add_argument("--size", default=f"{HEIGHT}x{WIDTH}", metavar="HxW",
                        help="coded frame size (engines must exist for it), e.g. 1152x1920 for 1080p with --fit pad")
    parser.add_argument("--fit", choices=("crop", "pad"), default="crop",
                        help="crop = center-crop to the coded aspect; pad = keep the whole picture, pad bottom/right")
    parser.add_argument("--precision", default="fp16")
    parser.add_argument("--backend", choices=("trt",), default="trt",
                        help="TensorRT sender engine")
    parser.add_argument("--engine-tag", default="agx_orin", help="--backend trt: engines/<tag>/ folder")
    parser.add_argument("--compute-units", default="cpu_and_gpu", help="--backend coreml only")
    parser.add_argument("--source", default="camera", help="camera | kodak | video:<path>")
    parser.add_argument("--camera", default="0", help="camera index, or a device path (Linux)")
    parser.add_argument("--camera-size", type=int, nargs=2, default=(1280, 720), metavar=("W", "H"))
    parser.add_argument("--fps", type=float, default=30.0, help="camera request / playback rate")
    parser.add_argument("--hold", type=int, default=1, help="playback: frames per Kodak image")
    parser.add_argument("--no-mirror", action="store_true")
    parser.add_argument("--snr-db", type=float, default=10.0, help="initial SNR (the web page changes it)")
    parser.add_argument("--rate", type=int, default=96, help="initial CR (the web page changes it)")
    parser.add_argument("--max-inflight", type=int, default=2)
    parser.add_argument("--no-preview", action="store_true")
    parser.add_argument("--preview-quality", type=int, default=80)
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--duration", type=float, default=0, help="stop after N seconds (0 = Ctrl-C)")
    parser.add_argument("--log-dir", default="results/live")
    rf = parser.add_argument_group("--link rf")
    rf.add_argument("--tx-gain", type=int, default=47, help="fixed TX VGA dB (0-47; default: maximum)")
    rf.add_argument("--tx-amp", action=argparse.BooleanOptionalAction, default=True,
                    help="TX RF amplifier (+~10 dB measured; on by default, --no-tx-amp at short range)")
    rf.add_argument("--tx-serial", help="HackRF TX serial from hackrf_info (required for RF)")
    rf.add_argument("--rf-hz", type=int, default=2_380_000_000, help="band centre (= LO, DC null carrier)")
    rf.add_argument("--tx-transfer-kib", type=float, default=32, help="libhackrf TX transfer size")
    rf.add_argument("--permutation-seed", type=int, default=20261106)
    rf.add_argument("--scramble-seed", type=int, default=20261107)
    args = parser.parse_args()
    if args.link == "rf" and not args.tx_serial:
        parser.error("--tx-serial is required for --link rf")
    HEIGHT, WIDTH = map(int, args.size.lower().split("x"))
    if HEIGHT % 128 or WIDTH % 128:
        raise SystemExit("--size: H and W must be multiples of 128")

    meta = json.loads((ROOT / "meta.json").read_text())
    channels_of = {int(r): int(c) for r, c in meta["sent_channels"].items()}
    if args.rate not in channels_of:
        raise SystemExit(f"--rate must be one of {sorted(channels_of)}")
    models = list(dict.fromkeys(args.models or [args.model]))
    if args.model not in models:
        args.model = models[0]
    if args.link != "rf" and (len(models) > 1 or {"swinjscc", "video"} & set(models)):
        raise SystemExit("SwinJSCC, the video model and per-frame model switching need --link rf")
    encoders, encoder_files = {}, {}
    dummy = np.zeros((HEIGHT, WIDTH, 3), np.uint8)
    for model in models:
        encoders[model], encoder_files[model] = load_sender(args, model)
        for _ in range(args.warmup):
            if model == "video":
                encoders[model].encode(dummy, args.snr_db, True)
                encoders[model].encode(dummy, args.snr_db, False, 1.0)
            else:
                encoders[model].encode(dummy, args.snr_db)
        print(f"encoder {model}: {encoder_files[model]}", flush=True)
    Path(args.log_dir).mkdir(parents=True, exist_ok=True)
    radio = RadioTx(args, meta) if args.link == "rf" else None
    camera = int(args.camera) if args.camera.isdigit() else args.camera
    source = (Camera(camera, *args.camera_size, args.fps) if args.source == "camera"
              else Playback(args.source, args.fps, args.hold))
    print(f"source {source.settings}", flush=True)
    first = source.newer(0, timeout=10.0)
    if first is None:
        raise SystemExit("No frame from the source within 10 s")
    valid = fit_size(*first[0].shape[:2]) if args.fit == "pad" else (HEIGHT, WIDTH)
    print(f"coded {HEIGHT}x{WIDTH}, picture {valid[0]}x{valid[1]} ({args.fit})", flush=True)

    channel = Channel(socket.create_connection((args.host, args.port), timeout=10))
    channel.sock.settimeout(None)
    preview = None if args.no_preview else Preview(args.host, args.port + 1, args.preview_quality)
    channel.send_json(b"HELO", {"model": args.model, "height": HEIGHT, "width": WIDTH,
                                "precision": args.precision, "backend": args.backend,
                                "encoder": encoder_files[args.model], "models": models, "encoders": encoder_files,
                                "compute_units": args.compute_units if args.backend == "coreml" else None,
                                "mirror": not args.no_mirror, "sender": socket.gethostname(),
                                "fit": args.fit, "valid_height": valid[0], "valid_width": valid[1],
                                "max_inflight": args.max_inflight, "link": args.link,
                                **(radio.hello if radio else {}), **source.settings})
    state = {"snr_db": args.snr_db, "rate": args.rate, "model": args.model}
    # video model GOP state: the counter runs over video frames only; a run (start, model switch) or the
    # receiver's need_key starts with a keyframe
    video = {"index": 0, "key_index": None, "force_key": False, "previous_model": None}
    inflight = threading.Semaphore(args.max_inflight)
    rows: dict[int, dict] = {}
    closed = threading.Event()

    def reader() -> None:
        try:
            while True:
                kind, body = channel.recv()
                if kind == b"PING":
                    (t_a,) = PING.unpack(body)
                    channel.send(b"PONG", PONG.pack(t_a, time.time_ns()))
                elif kind == b"ACK!":
                    seq, t_recv, t_d0, t_d1, t_out, dropped = ACK.unpack(body)
                    row = rows.get(seq)
                    if row is not None:
                        row.update(t_ack=time.time_ns(), rx_recv=t_recv, rx_decode_start=t_d0,
                                   rx_decode_end=t_d1, rx_out=t_out, dropped=dropped)
                    inflight.release()
                elif kind == b"RACK":
                    ack = json.loads(body)
                    row = rows.get(ack["seq"])
                    if row is not None:
                        row.update(t_ack=time.time_ns(), lost=ack.get("lost"), dropped=ack.get("dropped", 0),
                                   snr_estimate_db=ack.get("snr_estimate_db"), packets=ack.get("packets"))
                    if ack.get("snr_estimate_db") is not None:
                        state["snr_db"] = float(ack["snr_estimate_db"])  # measured: conditions the next frames
                    if ack.get("need_key"):  # the receiver lacks this P frame's keyframe
                        video["force_key"] = True
                    inflight.release()
                elif kind == b"CTRL":
                    control = json.loads(body)
                    if args.link == "rf":
                        control.pop("snr_db", None)  # over the air the SNR is measured, not set
                    if int(control.get("rate", state["rate"])) in channels_of:
                        state["rate"] = int(control.get("rate", state["rate"]))
                    if control.get("model") in encoders:
                        state["model"] = control["model"]
                    state["snr_db"] = float(control.get("snr_db", state["snr_db"]))
                    print(f"control: {state['model']} CR {state['rate']}"
                          + ("" if args.link == "rf" else f", SNR {state['snr_db']:g} dB"), flush=True)
        except (ConnectionError, OSError):
            pass
        closed.set()

    threading.Thread(target=reader, daemon=True).start()
    log_dir = Path(args.log_dir)
    log_dir.mkdir(parents=True, exist_ok=True)
    log_path = log_dir / f"{time.strftime('%Y%m%d-%H%M%S')}_sender.csv"
    seq, last_index, start = 0, 0, time.time()
    encode_ms: list[float] = []
    next_report = time.time() + 1.0

    # --link rf: modulation + TX run in their own thread, so the next frame encodes (GPU) while this one
    # modulates (CPU, C++ OFDM) and streams; a one-slot queue keeps frames in order and backpressures.
    tx_queue: queue.Queue = queue.Queue(maxsize=1)
    tx_errors: list[BaseException] = []

    def transmitter() -> None:
        while True:
            item = tx_queue.get()
            if item is None:
                return
            tx_seq, tx_symbols, tx_rate, tx_layers, meta_json = item
            try:
                t_mod0 = time.time_ns()
                sent = radio.send(tx_symbols, tx_rate, tx_seq, tx_layers)
                rows[tx_seq].update(t_modulate_start=t_mod0, t_modulate_end=sent["t_modulate_end"],
                                    t_tx_start=sent["tx"]["t_write_start"],
                                    tx_stream_sample=sent["tx"]["tx_stream_sample"],
                                    tx_underruns=sent["tx"]["underruns"], air_ms=1e3 * sent["waveform"]["seconds"])
                channel.send_json(b"RFFR", {**meta_json, "t_modulate_start": t_mod0, **sent})
                rows[tx_seq]["t_sent"] = time.time_ns()
            except Exception as error:  # noqa: BLE001 - one bad frame must not end the stream
                print(f"frame {tx_seq} not transmitted: {error!r}", flush=True)
                rows[tx_seq]["tx_error"] = repr(error)
                tx_errors.append(error)
                inflight.release()  # no ACK will come for it

    tx_thread = threading.Thread(target=transmitter, daemon=True) if radio is not None else None
    if tx_thread is not None:
        tx_thread.start()
    try:
        while not closed.is_set():
            if args.duration and time.time() - start > args.duration:
                break
            if not inflight.acquire(timeout=0.5):
                continue
            got = source.newer(last_index, timeout=1.0)
            if got is None:
                inflight.release()
                continue
            frame, t_capture, last_index = got
            t_pick = time.time_ns()
            if args.fit == "pad" and fit_size(*frame.shape[:2]) != valid:
                raise SystemExit(f"source frame size changed to {frame.shape[1]}x{frame.shape[0]}")
            bgr = prepare(frame, not args.no_mirror, args.fit)
            snr_db, rate, model = state["snr_db"], state["rate"], state["model"]
            if radio is not None:
                snr_db = min(max(snr_db, SNR_RANGE[0]), SNR_RANGE[1])
            t_enc0 = time.time_ns()
            video_info = None
            if model == "video":  # Tiny video model: fixed CR 96 (as trained), keyframe every key_interval
                encoder = encoders[model]
                rate, channels, layers = 96, channels_of[96], None
                start_run = video["previous_model"] != "video"
                if start_run or video["force_key"] or video["key_index"] is None \
                        or video["index"] - video["key_index"] >= encoder.key_interval:
                    video["key_index"], video["force_key"] = video["index"], False
                key = video["key_index"] == video["index"]
                symbols = encoder.encode(bgr[..., ::-1], snr_db, key, float(video["index"] - video["key_index"]))[
                    0, :channels]
                video_info = {"index": video["index"], "key": key, "key_index": video["key_index"], "start": start_run}
                video["index"] += 1
            elif model == "swinjscc":  # WITT C16: fixed CR 96, one "layer" of 16 channels for the OFDM framing
                rate, channels, layers = 96, SWIN_CHANNELS, [(0, SWIN_CHANNELS)]
                symbols = encoders[model].encode(bgr[..., ::-1], snr_db)[0].reshape(
                    SWIN_CHANNELS, HEIGHT // 16, WIDTH // 16)  # flat [L*16]; the receiver flattens it back
            else:
                channels, layers = channels_of[rate], None
                symbols = encoders[model].encode(bgr[..., ::-1], snr_db)[0, :channels]  # RGB in; rate = layer prefix
            video["previous_model"] = model
            t_enc1 = time.time_ns()
            rows[seq] = {"seq": seq, "camera_index": last_index, "t_capture": t_capture, "t_pick": t_pick,
                         "t_encode_start": t_enc0, "t_encode_end": t_enc1, "snr_db": snr_db, "model": model,
                         "rate": rate, "channels": channels, "payload_bytes": symbols.nbytes}
            if radio is None:
                channel.send(b"FRAM", FRAME.pack(seq, last_index, t_capture, t_pick, t_enc0, t_enc1,
                                                 snr_db, float(rate), HEIGHT, WIDTH, channels), symbols.tobytes())
                rows[seq]["t_sent"] = time.time_ns()
            else:  # copy: the encoder's output buffer is reused by the next frame while this one modulates
                item = (seq, np.array(symbols, copy=True), rate, layers,
                        {"seq": seq, "camera_index": last_index, "t_capture": t_capture, "t_pick": t_pick,
                         "t_encode_start": t_enc0, "t_encode_end": t_enc1, "encoder_snr_db": snr_db,
                         "model": model, "rate": rate, "channels": channels, "height": HEIGHT, "width": WIDTH,
                         **({"video": video_info} if video_info else {})})
                while not closed.is_set():
                    try:
                        tx_queue.put(item, timeout=0.5)
                        break
                    except queue.Full:
                        continue
            if preview is not None:
                preview.offer(seq, t_capture, bgr[:valid[0], :valid[1]])
            encode_ms.append((t_enc1 - t_enc0) / 1e6)
            seq += 1
            if time.time() >= next_report:
                next_report += 1.0
                recent = [r for r in list(rows.values())[-60:] if "t_ack" in r]
                ack_ms = statistics.median((r["t_ack"] - r["t_encode_end"]) / 1e6 for r in recent) if recent else 0
                snr_text = (f"encoder SNR {snr_db:.1f} dB (measured {state['snr_db']:.1f})" if radio
                            else f"SNR {snr_db:g} dB")
                print(f"camera {source.fps():5.1f} fps | sent {seq} | encode {statistics.median(encode_ms[-30:]):5.1f} ms"
                      f" | encode-end->ack {ack_ms:5.1f} ms | {snr_text} {model} CR {rate}", flush=True)
    except (KeyboardInterrupt, BrokenPipeError, ConnectionError):
        pass
    finally:
        source.stop.set()
        if tx_thread is not None:
            try:
                tx_queue.put(None, timeout=2.0)
            except queue.Full:
                pass
            tx_thread.join(timeout=5.0)
        if tx_errors:
            print(f"transmitter failed: {tx_errors[0]!r}", flush=True)
        if radio is not None:
            radio.close()
        try:
            channel.send(b"BYE!")
        except OSError:
            pass
        time.sleep(0.2)
        channel.close()
        if rows:
            fields = list(dict.fromkeys(k for r in rows.values() for k in r))
            with open(log_path, "w", newline="") as handle:
                writer = csv.DictWriter(handle, fieldnames=fields)
                writer.writeheader()
                writer.writerows(rows.values())
            print(f"{len(rows)} frames, encode median {statistics.median(encode_ms):.1f} ms -> {log_path}")


if __name__ == "__main__":
    main()
