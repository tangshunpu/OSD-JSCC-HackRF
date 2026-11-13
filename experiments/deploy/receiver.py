"""TensorRT receiver for OSD-JSCC-P over HackRF.

See README.md for engine setup and the live receiver entry point.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import socket
import statistics
import struct
import subprocess
import sys
import time

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from trt_runtime import Engine, trt  # noqa: E402

ROOT = REPO_ROOT / "runs/deploy/osd_jscc_p"
KODAK = Path("dataset/Kodak")
HEADER = struct.Struct("<4sIIHHffIqq32s")
ACK = struct.Struct("<4sIqqqf")
SYNC = struct.Struct("<4sqq")
TORCH_DTYPES = {np.dtype(np.float32): torch.float32, np.dtype(np.float16): torch.float16,
                np.dtype(np.uint8): torch.uint8, np.dtype(np.int32): torch.int32,
                np.dtype(np.int64): torch.int64}
GRAPHS = ("rx_jscc", "unet_step", "vae_decode")


def load_rgb(path: Path) -> np.ndarray:
    from PIL import Image

    with Image.open(path) as image:
        return np.asarray(image.convert("RGB"), dtype=np.uint8).copy()


def add_awgn(symbols: np.ndarray, snr_db: float, seed: int) -> np.ndarray:
    """Unit-power symbols + AWGN (per real component variance 1/(2 SNR)), as fp16."""
    rng = np.random.default_rng(seed)
    sigma = np.sqrt(0.5 * 10 ** (-snr_db / 10))
    noise = rng.standard_normal(symbols.shape, dtype=np.float32) * sigma
    return (symbols.astype(np.float32) + noise).astype(np.float16)


def gpu_state() -> dict:
    query = "name,driver_version,power.limit,clocks.max.sm,clocks.sm"
    line = subprocess.run(["nvidia-smi", f"--query-gpu={query}", "--format=csv,noheader", "-i", "0"],
                          capture_output=True, text=True).stdout.strip()
    return {"gpu": line, "tensorrt": trt.__version__, "torch": torch.__version__}


class Routing:
    def __init__(self, meta: dict) -> None:
        self.thresholds = {int(k): v for k, v in meta["rate_thresholds"].items()}
        self.sigmas = {int(k): v for k, v in meta["sigmas"].items()}
        self.high, self.low = meta["high_timestep"], meta["low_timestep"]

    def __call__(self, snr_db: float, rate: int) -> tuple[float, float]:
        timestep = self.high if snr_db >= self.thresholds[rate] else self.low
        return float(timestep), self.sigmas[timestep]


# ------------------------------------------------------------ TensorRT receiver

class TRTReceiver:
    """rx_jscc -> unet_step -> vae_decode engines sharing device buffers."""

    def __init__(self, meta: dict, precision: str, height: int, width: int, *,
                 profile: int = 0, max_batch: int = 1, tag: str = "rtx6000",
                 use_graph: bool = True, vae_suffix: str = "") -> None:
        """``vae_suffix`` picks another image-decoder engine with the vae_decode I/O, e.g. "_tiny"
        (the pruned decoder of experiments/taesd); the JSCC decoder and U-Net engines are shared."""
        self.meta, self.routing = meta, Routing(meta)
        self.height, self.width, self.max_batch = height, width, max_batch
        self.profile, self.use_graph = profile, use_graph
        self.engines = {}
        for graph in GRAPHS:
            suffix = vae_suffix if graph == "vae_decode" else ""
            path = ROOT / "engines" / tag / f"{graph}_{height}x{width}_{precision}{suffix}.plan"
            self.engines[graph] = Engine(path, profiles=(profile,))
        self.stream = torch.cuda.Stream()
        shapes = self._shapes(max_batch)
        self.device, self.host = {}, {}
        for graph, engine in self.engines.items():
            for name in engine.names:
                dtype = TORCH_DTYPES[np.dtype(engine.dtype(name))]
                if name in self.device:
                    if self.device[name].dtype != dtype:
                        raise TypeError(f"{name}: dtype mismatch between engines.")
                    continue
                self.device[name] = torch.empty(shapes[name], dtype=dtype, device="cuda")
        for name in ("received", "snr_db", "rate", "timestep", "sigma", "image"):
            self.host[name] = torch.zeros(shapes[name], dtype=self.device[name].dtype,
                                          pin_memory=True)
        self.batch, self.graphs = None, {}
        self.events = [torch.cuda.Event(enable_timing=True) for _ in range(5)]

    def _shapes(self, batch: int) -> dict:
        h, w, meta = self.height, self.width, self.meta
        return {"received": (batch, meta["c_max"], h // 16, w // 16), "snr_db": (batch,),
                "rate": (batch,), "timestep": (batch,), "sigma": (batch,),
                "latent": (batch, 4, h // 8, w // 8),
                "conditioning": (batch, meta["context_tokens"], meta["context_dim"]),
                "vae_latent": (batch, 4, h // 8, w // 8), "image": (batch, h, w, 3)}

    def set_batch(self, batch: int) -> None:
        if batch == self.batch:
            return
        if batch > self.max_batch:
            raise ValueError(f"Batch {batch} exceeds the buffers ({self.max_batch}).")
        shapes = self._shapes(batch)
        for engine in self.engines.values():
            outputs = engine.set_inputs({n: shapes[n] for n in engine.inputs}, self.profile)
            for name, shape in outputs.items():
                if tuple(shape) != tuple(shapes[name]):
                    raise RuntimeError(f"{name}: engine shape {shape} != {shapes[name]}.")
            engine.bind({n: self.device[n].data_ptr() for n in engine.names}, self.profile)
        self.batch = batch

    def _work(self, batch: int, timed: bool = False) -> None:
        stream = torch.cuda.current_stream().cuda_stream
        for name in ("received", "snr_db", "rate", "timestep", "sigma"):
            self.device[name][:batch].copy_(self.host[name][:batch], non_blocking=True)
        if timed:
            self.events[0].record()
        for index, graph in enumerate(GRAPHS):
            self.engines[graph].enqueue(stream, self.profile)
            if timed:
                self.events[index + 1].record()
        self.host["image"][:batch].copy_(self.device["image"][:batch], non_blocking=True)
        if timed:
            self.events[4].record()

    def stage(self, received: np.ndarray, snr_db, rate) -> int:
        """Copy received prefixes [B, ch, h, w] (fp16) into the zero-filled staging buffer."""
        batch, channels = received.shape[:2]
        snr_db = np.broadcast_to(np.asarray(snr_db, np.float32), (batch,))
        rate = np.broadcast_to(np.asarray(rate, np.int64), (batch,))
        host = self.host["received"].numpy()
        host[:batch, channels:] = 0
        host[:batch, :channels] = received
        self.host["snr_db"].numpy()[:batch] = snr_db
        self.host["rate"].numpy()[:batch] = rate
        routes = [self.routing(float(s), int(r)) for s, r in zip(snr_db, rate)]
        self.host["timestep"].numpy()[:batch] = [t for t, _ in routes]
        self.host["sigma"].numpy()[:batch] = [s for _, s in routes]
        return batch

    def run(self, batch: int) -> torch.Tensor:
        self.set_batch(batch)
        with torch.cuda.stream(self.stream):
            if not self.use_graph:
                self._work(batch)
            else:
                if batch not in self.graphs:
                    self._work(batch)  # TensorRT needs one enqueue before capture
                    self.stream.synchronize()
                    graph = torch.cuda.CUDAGraph()
                    with torch.cuda.graph(graph, stream=self.stream):
                        self._work(batch)
                    self.graphs[batch] = graph
                self.graphs[batch].replay()
        self.stream.synchronize()
        return self.host["image"][:batch]

    def decode(self, received: np.ndarray, snr_db, rate) -> np.ndarray:
        return self.run(self.stage(received, snr_db, rate)).numpy()

    def breakdown(self, received: np.ndarray, snr_db, rate) -> dict:
        batch = self.stage(received, snr_db, rate)
        self.set_batch(batch)
        start = torch.cuda.Event(enable_timing=True)
        with torch.cuda.stream(self.stream):
            start.record()
            self._work(batch, timed=True)
        self.stream.synchronize()
        e = self.events
        return {"h2d_ms": start.elapsed_time(e[0]), "jscc_ms": e[0].elapsed_time(e[1]),
                "unet_ms": e[1].elapsed_time(e[2]), "vae_ms": e[2].elapsed_time(e[3]),
                "d2h_ms": e[3].elapsed_time(e[4])}


# ------------------------------------------------------------- PyTorch receiver
