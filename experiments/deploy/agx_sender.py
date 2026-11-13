"""OSD-JSCC-P transmitter on the Jetson AGX Orin (TensorRT, no torch).

Run on the AGX from the deployment folder (same layout as the repo):

    python3 experiments/deploy/agx_sender.py bench          # Kodak latency + accuracy
    python3 experiments/deploy/agx_sender.py bench --gap 2  # isolated images (DVFS ramp-up)
    python3 experiments/deploy/agx_sender.py power          # idle vs. sustained power
    python3 experiments/deploy/agx_sender.py send --host RECEIVER_IP   # live link

Timing follows ``experiments/runtime``: encoding starts from the uint8 RGB
array in host memory and ends with the channel symbols of every layer in host
memory, ready for the socket. PNG reading is not timed. The image and SNR are
copied into pinned buffers, and H2D copy + engine + D2H copy are one CUDA
graph (``--no-graph`` enqueues them one by one instead).

Power comes from the module's INA3221 monitors (the rails tegrastats prints).
The module total is VDD_GPU_SOC + VDD_CPU_CV + VIN_SYS_5V0; VDDQ_VDD2_1V8AO is
reported separately. The sensor averages over ~0.1 s, so power is measured
over a sustained back-to-back loop, never per image.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
from pathlib import Path
import socket
import statistics
import struct
import sys
import threading
import time

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from trt_runtime import Cudart, Engine, host_array, trt  # noqa: E402

ROOT = Path("runs/deploy/osd_jscc_p")
KODAK = Path("dataset/Kodak")
HEADER = struct.Struct("<4sIIHHffIqq32s")   # magic seq index H W snr rate nbytes t0 t1 name
ACK = struct.Struct("<4sIqqqf")             # magic seq recv_done decode_start decode_done decode_ms
SYNC = struct.Struct("<4sqq")


def load_rgb(path: Path) -> np.ndarray:
    try:
        from PIL import Image

        with Image.open(path) as image:
            return np.asarray(image.convert("RGB"), dtype=np.uint8).copy()
    except ImportError:  # pragma: no cover - fall back to OpenCV
        import cv2

        return cv2.cvtColor(cv2.imread(str(path)), cv2.COLOR_BGR2RGB)


def read(path: str) -> str:
    try:
        return Path(path).read_text().strip()
    except OSError:
        return ""


GPU_FREQ = "/sys/devices/platform/17000000.gpu/devfreq/17000000.gpu/cur_freq"


def device_state() -> dict:
    gpu = "/sys/devices/platform/17000000.gpu/devfreq/17000000.gpu"
    mode = os.popen("nvpmodel -q 2>/dev/null").read().split("\n")[0].replace("NV Power Mode: ", "")
    return {
        "model": read("/proc/device-tree/model").rstrip("\x00"),
        "l4t": read("/etc/nv_tegra_release").split("\n")[0],
        "power_mode": mode,
        "gpu_max_freq_mhz": int(read(f"{gpu}/max_freq") or 0) / 1e6,
        "gpu_cur_freq_mhz": int(read(f"{gpu}/cur_freq") or 0) / 1e6,
        "tpc_pg_mask": read("/sys/devices/platform/gpu.0/tpc_pg_mask"),
        "cpu_online": read("/sys/devices/system/cpu/online"),
        "cpu_max_freq_mhz": int(read("/sys/devices/system/cpu/cpu0/cpufreq/scaling_max_freq") or 0) / 1e3,
        "tensorrt": trt.__version__,
    }


# ------------------------------------------------------------------ sender

class Sender:
    """One static sender engine with pinned I/O and an optional CUDA graph."""

    def __init__(self, cuda: Cudart, path: Path, use_graph: bool = True) -> None:
        self.cuda, self.engine = cuda, Engine(path)
        shapes = {n: tuple(self.engine.engine.get_tensor_shape(n)) for n in self.engine.names}
        self.shapes = shapes
        self.height, self.width = shapes["image"][1:3]
        self.stream = cuda.stream()
        self.device, self.host, self.nbytes = {}, {}, {}
        for name in self.engine.names:
            dtype = np.dtype(self.engine.dtype(name))
            nbytes = int(np.prod(shapes[name])) * dtype.itemsize
            self.nbytes[name] = nbytes
            self.device[name] = cuda.malloc(nbytes)
            address = cuda.host_alloc(nbytes)
            self.host[name] = host_array(address, shapes[name], dtype)
        self.engine.bind(self.device)
        self.start_event, self.end_event = cuda.event(), cuda.event()
        self._enqueue_all(timed=False)  # first enqueue initializes the context
        cuda.sync(self.stream)
        self.graph = cuda.capture(self.stream, lambda: self._enqueue_all(False)) if use_graph else None

    def _enqueue_all(self, timed: bool) -> None:
        cuda, s = self.cuda, self.stream
        for name in ("image", "snr_db"):
            cuda.memcpy_async(self.device[name], self.host[name].ctypes.data, self.nbytes[name],
                              Cudart.H2D, s)
        if timed:
            cuda.record(self.start_event, s)
        self.engine.enqueue(s)
        if timed:
            cuda.record(self.end_event, s)
        cuda.memcpy_async(self.host["symbols"].ctypes.data, self.device["symbols"],
                          self.nbytes["symbols"], Cudart.D2H, s)

    def encode(self, image: np.ndarray, snr_db: float) -> np.ndarray:
        np.copyto(self.host["image"][0], image)
        self.host["snr_db"][0] = snr_db
        if self.graph is not None:
            self.cuda.launch(self.graph, self.stream)
        else:
            self._enqueue_all(timed=False)
        self.cuda.sync(self.stream)
        return self.host["symbols"]

    def engine_ms(self, image: np.ndarray, snr_db: float) -> float:
        """GPU time of the engine alone (events around one ungraphed enqueue)."""
        np.copyto(self.host["image"][0], image)
        self.host["snr_db"][0] = snr_db
        self._enqueue_all(timed=True)
        self.cuda.sync(self.stream)
        return self.cuda.elapsed_ms(self.start_event, self.end_event)


def load_senders(args) -> dict[tuple[int, int], Sender]:
    cuda = Cudart()
    senders = {}
    for path in sorted((ROOT / "engines" / args.tag).glob(f"sender_*_{args.precision}{args.suffix}.plan")):
        sender = Sender(cuda, path, use_graph=not args.no_graph)
        senders[(sender.height, sender.width)] = sender
        print(f"loaded {path} ({sender.height}x{sender.width})", flush=True)
    if not senders:
        raise SystemExit("No sender engines found; build them first.")
    return senders


def kodak(limit: int | None = None) -> list[tuple[str, np.ndarray]]:
    paths = sorted(KODAK.glob("kodim*.png"))[:limit] if limit else sorted(KODAK.glob("kodim*.png"))
    return [(path.stem, load_rgb(path)) for path in paths]


# ------------------------------------------------------------------- power

class PowerMonitor:
    """Background sampler of the INA3221 rails (mW = bus mV x mA / 1000)."""

    def __init__(self, interval: float = 0.01) -> None:
        self.rails = {}
        for label in Path("/sys/bus/i2c/drivers/ina3221").glob("*/hwmon/hwmon*/in*_label"):
            name = label.read_text().strip()
            channel = label.name[2:-6]
            current = label.with_name(f"curr{channel}_input")
            if current.exists() and name != "sum of shunt voltages":
                self.rails[name] = (label.with_name(f"in{channel}_input"), current)
        if not self.rails:
            raise RuntimeError("No INA3221 rails found.")
        self.interval, self.samples, self._stop = interval, [], threading.Event()

    def sample(self) -> dict[str, float]:
        return {name: int(v.read_text()) * int(c.read_text()) / 1000.0
                for name, (v, c) in self.rails.items()}

    def _run(self) -> None:
        while not self._stop.is_set():
            self.samples.append((time.perf_counter(), self.sample()))
            time.sleep(self.interval)

    def __enter__(self):
        self.samples, self._stop = [], threading.Event()
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()
        return self

    def __exit__(self, *_):
        self._stop.set()
        self._thread.join()

    def mean(self) -> dict[str, float]:
        values = {name: statistics.fmean(s[name] for _, s in self.samples) for name in self.rails}
        values["module"] = sum(values[n] for n in ("VDD_GPU_SOC", "VDD_CPU_CV", "VIN_SYS_5V0")
                               if n in values)
        return values


# ------------------------------------------------------------------- modes

def bench(args) -> None:
    senders = load_senders(args)
    reference = np.load(ROOT / "reference" / f"sender_kodak_{args.snr_db:g}dB.npz")
    images = kodak(args.limit)
    for sender in senders.values():  # warm-up (also lets DVFS ramp the clocks)
        dummy = np.zeros((sender.height, sender.width, 3), np.uint8)
        for _ in range(args.warmup):
            sender.encode(dummy, args.snr_db)
    rows, accuracy = [], []
    for name, image in images:
        sender = senders[image.shape[:2]]
        for repeat in range(args.repeats):
            if args.gap:  # isolated image: let DVFS lower the clocks first
                time.sleep(args.gap)
            freq = int(read(GPU_FREQ) or 0) / 1e6
            start = time.perf_counter()
            symbols = sender.encode(image, args.snr_db)
            seconds = time.perf_counter() - start
            rows.append({"image": name, "repeat": repeat, "encode_ms": 1e3 * seconds,
                         "gpu_mhz_at_start": freq,
                         "engine_ms": sender.engine_ms(image, args.snr_db)})
        ours, ref = symbols[0].astype(np.float32), reference[name]
        error = np.linalg.norm(ours - ref) / np.linalg.norm(ref)
        accuracy.append({"image": name, "rel_l2": float(error),
                         "error_snr_db": float(-20 * np.log10(max(error, 1e-12))),
                         "max_abs": float(np.abs(ours - ref).max())})

    def per_image(key):
        groups = {}
        for row in rows:
            groups.setdefault(row["image"], []).append(row[key])
        return [statistics.median(v) for v in groups.values()]

    encode, engine_only = per_image("encode_ms"), per_image("engine_ms")
    summary = {
        "precision": args.precision + args.suffix, "cuda_graph": not args.no_graph,
        "images": len(images), "repeats": args.repeats, "snr_db": args.snr_db,
        "idle_gap_s": args.gap,
        "gpu_mhz_at_start_mean": statistics.fmean(r["gpu_mhz_at_start"] for r in rows),
        "encode_ms_mean": statistics.fmean(encode), "encode_ms_min": min(encode),
        "encode_ms_max": max(encode), "engine_ms_mean": statistics.fmean(engine_only),
        "error_snr_db_min": min(a["error_snr_db"] for a in accuracy),
        "error_snr_db_mean": statistics.fmean(a["error_snr_db"] for a in accuracy),
        "rel_l2_max": max(a["rel_l2"] for a in accuracy),
        "payload_bytes_all_layers": int(next(iter(senders.values())).nbytes["symbols"]),
        **device_state(),
    }
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    tag = f"{args.precision}{args.suffix}{'' if not args.no_graph else '_nograph'}"
    tag += f"_gap{args.gap:g}s" if args.gap else ""
    with open(out / f"agx_encode_{tag}.csv", "w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    with open(out / f"agx_accuracy_{tag}.csv", "w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(accuracy[0]))
        writer.writeheader()
        writer.writerows(accuracy)
    (out / f"agx_encode_{tag}.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2), flush=True)


def power(args) -> None:
    senders = load_senders(args)
    name, image = kodak(1)[0]
    sender = senders[image.shape[:2]]
    monitor = PowerMonitor()
    time.sleep(args.settle)
    with monitor:
        time.sleep(args.idle_seconds)
    idle = monitor.mean()
    for _ in range(args.warmup):
        sender.encode(image, args.snr_db)
    count = 0
    with monitor:
        start = time.perf_counter()
        while time.perf_counter() - start < args.busy_seconds:
            sender.encode(image, args.snr_db)
            count += 1
        seconds = time.perf_counter() - start
    busy = monitor.mean()
    per_image_s = seconds / count
    summary = {
        "precision": args.precision + args.suffix, "image": name, "images": count,
        "seconds": seconds, "ms_per_image": 1e3 * per_image_s, "images_per_s": count / seconds,
        "idle_mw": idle, "busy_mw": busy,
        "energy_mj_per_image": busy["module"] * per_image_s,
        "dynamic_energy_mj_per_image": (busy["module"] - idle["module"]) * per_image_s,
        "gpu_energy_mj_per_image": busy["VDD_GPU_SOC"] * per_image_s,
        **device_state(),
    }
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    (out / f"agx_power_{args.precision}{args.suffix}.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2), flush=True)


def clock_offset(sock: socket.socket, rounds: int = 50) -> tuple[int, int]:
    """Receiver clock minus sender clock (ns), from the lowest-RTT exchange."""
    best = None
    for _ in range(rounds):
        t_a = time.time_ns()
        sock.sendall(SYNC.pack(b"SYNC", t_a, 0))
        _, _, t_b = SYNC.unpack(recv_exact(sock, SYNC.size))
        t_c = time.time_ns()
        rtt = t_c - t_a
        if best is None or rtt < best[0]:
            best = (rtt, t_b - (t_a + t_c) // 2)
    sock.sendall(SYNC.pack(b"DONE", 0, 0))
    return best[1], best[0]


def recv_exact(sock: socket.socket, size: int) -> bytes:
    data = bytearray()
    while len(data) < size:
        chunk = sock.recv(size - len(data))
        if not chunk:
            raise ConnectionError("Receiver closed the connection.")
        data += chunk
    return bytes(data)


def send(args) -> None:
    senders = load_senders(args)
    meta = json.loads((ROOT / "meta.json").read_text())
    channels = int(meta["sent_channels"][str(args.rate)])
    images = kodak(args.limit)
    for sender in senders.values():
        dummy = np.zeros((sender.height, sender.width, 3), np.uint8)
        for _ in range(args.warmup):
            sender.encode(dummy, args.snr_db)
    sock = socket.create_connection((args.host, args.port))
    sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
    offset, rtt = clock_offset(sock)
    print(f"clock offset {offset / 1e6:.3f} ms (sync RTT {rtt / 1e6:.3f} ms)", flush=True)
    rows, seq = [], 0
    for repeat in range(args.repeats):
        for index, (name, image) in enumerate(images):
            sender = senders[image.shape[:2]]
            t0 = time.time_ns()
            symbols = sender.encode(image, args.snr_db)
            payload = symbols[0, :channels].tobytes()  # layers are a contiguous prefix
            t1 = time.time_ns()
            header = HEADER.pack(b"OSDP", seq, index, sender.height, sender.width, args.snr_db,
                                 float(args.rate), len(payload), t0, t1, name.encode())
            sock.sendall(header + payload)
            t_sent = time.time_ns()
            magic, ack_seq, recv_done, decode_start, decode_done, decode_ms = ACK.unpack(
                recv_exact(sock, ACK.size))
            t2 = time.time_ns()
            if magic != b"ACK!" or ack_seq != seq:
                raise RuntimeError("Out-of-order acknowledgement.")
            rows.append({
                "seq": seq, "repeat": repeat, "image": name, "height": sender.height,
                "width": sender.width, "rate": args.rate, "snr_db": args.snr_db,
                "payload_bytes": len(payload), "encode_ms": (t1 - t0) / 1e6,
                "send_call_ms": (t_sent - t1) / 1e6,
                "transfer_ms": (recv_done - offset - t1) / 1e6,
                "decode_ms": decode_ms,
                "e2e_ms": (decode_done - offset - t0) / 1e6,
                "rtt_e2e_ms": (t2 - t0) / 1e6,
            })
            print(f"{name} {sender.height}x{sender.width}: encode {rows[-1]['encode_ms']:.1f} ms, "
                  f"link {rows[-1]['transfer_ms']:.2f} ms, decode {decode_ms:.1f} ms, "
                  f"e2e {rows[-1]['e2e_ms']:.1f} ms", flush=True)
            seq += 1
    sock.sendall(HEADER.pack(b"STOP", seq, 0, 0, 0, 0.0, 0.0, 0, 0, 0, b""))
    sock.close()
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    path = out / f"link_cr{args.rate}_{args.snr_db:g}dB_{args.precision}{args.suffix}.csv"
    with open(path, "w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    summary = {k: statistics.fmean(r[k] for r in rows if r["repeat"] > 0 or args.repeats == 1)
               for k in ("encode_ms", "transfer_ms", "decode_ms", "e2e_ms", "rtt_e2e_ms")}
    summary.update(clock_offset_ms=offset / 1e6, sync_rtt_ms=rtt / 1e6,
                   payload_bytes=rows[0]["payload_bytes"], **device_state())
    path.with_suffix(".json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2), flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("mode", choices=("bench", "power", "send"))
    parser.add_argument("--tag", default="agx_orin")
    parser.add_argument("--precision", default="fp16")
    parser.add_argument("--suffix", default="")
    parser.add_argument("--snr-db", type=float, default=12.0)
    parser.add_argument("--rate", type=int, default=96, help="CR of the live link (send mode).")
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--no-graph", action="store_true")
    parser.add_argument("--gap", type=float, default=0.0,
                        help="bench: idle seconds before every timed encode (isolated images).")
    parser.add_argument("--idle-seconds", type=float, default=10.0)
    parser.add_argument("--busy-seconds", type=float, default=30.0)
    parser.add_argument("--settle", type=float, default=5.0)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=50007)
    parser.add_argument("--out-dir", default=str(ROOT / "results"))
    args = parser.parse_args()
    {"bench": bench, "power": power, "send": send}[args.mode](args)


if __name__ == "__main__":
    main()
