"""TensorRT build/run helpers shared by the AGX sender and the RTX receiver.

Depends only on numpy, the TensorRT Python bindings and libcudart (through
ctypes), so it runs on the Jetson, which has no torch. The PyPI wheel names
its module ``tensorrt_bindings``; JetPack's apt package names it ``tensorrt``.
"""

from __future__ import annotations

import ctypes
import ctypes.util
import os
from pathlib import Path
import time

import numpy as np

try:
    import tensorrt as trt
except ImportError:  # PyPI wheel without the meta package
    import tensorrt_bindings as trt

LOGGER = trt.Logger(trt.Logger.WARNING)

NP_DTYPES = {
    trt.DataType.FLOAT: np.float32,
    trt.DataType.HALF: np.float16,
    trt.DataType.INT32: np.int32,
    trt.DataType.INT64: np.int64,
    trt.DataType.UINT8: np.uint8,
    trt.DataType.INT8: np.int8,
    trt.DataType.BOOL: np.bool_,
}


# ---------------------------------------------------------------- building

def build_engine(onnx_path: Path, engine_path: Path, *, precision: str,
                 profiles: list[dict[str, tuple[tuple[int, ...], ...]]] | None = None,
                 opt_level: int = 3, timing_cache: Path | None = None,
                 fp32_norms: bool = False, workspace_gb: float | None = None) -> dict:
    """Parse an ONNX file and serialize an engine.

    ``precision`` is ``fp32``, ``fp16`` or ``bf16`` (weakly typed: TensorRT
    picks per-layer precisions among fp32 and the enabled reduced type).
    ``profiles`` maps every dynamic input to (min, opt, max) shapes, one dict
    per optimization profile.
    """
    builder = trt.Builder(LOGGER)
    network = builder.create_network(0)
    parser = trt.OnnxParser(network, LOGGER)
    if not parser.parse_from_file(str(onnx_path)):
        errors = "\n".join(str(parser.get_error(i)) for i in range(parser.num_errors))
        raise RuntimeError(f"ONNX parse failed for {onnx_path}:\n{errors}")
    config = builder.create_builder_config()
    if precision == "fp16":
        config.set_flag(trt.BuilderFlag.FP16)
    elif precision == "bf16":
        config.set_flag(trt.BuilderFlag.BF16)
    elif precision != "fp32":
        raise ValueError(f"Unknown precision {precision!r}.")
    if workspace_gb:
        config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, int(workspace_gb * 2**30))
    config.builder_optimization_level = opt_level
    if fp32_norms:
        # Keep normalization layers (and the reductions feeding them) in fp32.
        config.set_flag(trt.BuilderFlag.OBEY_PRECISION_CONSTRAINTS)
        for index in range(network.num_layers):
            layer = network.get_layer(index)
            if layer.type in (trt.LayerType.NORMALIZATION, trt.LayerType.REDUCE):
                layer.precision = trt.DataType.FLOAT
    for shapes in profiles or []:
        profile = builder.create_optimization_profile()
        for name, (low, opt, high) in shapes.items():
            profile.set_shape(name, low, opt, high)
        config.add_optimization_profile(profile)

    cache = None
    if timing_cache is not None:
        data = timing_cache.read_bytes() if timing_cache.exists() else b""
        cache = config.create_timing_cache(data)
        config.set_timing_cache(cache, ignore_mismatch=False)
    start = time.perf_counter()
    serialized = builder.build_serialized_network(network, config)
    if serialized is None:
        raise RuntimeError(f"Engine build failed for {onnx_path}.")
    seconds = time.perf_counter() - start
    engine_path.parent.mkdir(parents=True, exist_ok=True)
    engine_path.write_bytes(memoryview(serialized))
    if cache is not None:
        timing_cache.write_bytes(memoryview(config.get_timing_cache().serialize()))
    return {"engine": str(engine_path), "build_seconds": seconds,
            "bytes": engine_path.stat().st_size, "precision": precision,
            "opt_level": opt_level, "fp32_norms": fp32_norms,
            "tensorrt": trt.__version__}


# ----------------------------------------------------------------- running

class Engine:
    """A deserialized engine with one execution context per optimization profile."""

    def __init__(self, path: Path | str, profiles: tuple[int, ...] | None = None) -> None:
        self.path = Path(path)
        self.runtime = trt.Runtime(LOGGER)
        self.engine = self.runtime.deserialize_cuda_engine(self.path.read_bytes())
        if self.engine is None:
            raise RuntimeError(f"Could not deserialize {self.path}.")
        self.names = [self.engine.get_tensor_name(i) for i in range(self.engine.num_io_tensors)]
        self.inputs = [n for n in self.names
                       if self.engine.get_tensor_mode(n) == trt.TensorIOMode.INPUT]
        self.outputs = [n for n in self.names if n not in self.inputs]
        count = self.engine.num_optimization_profiles
        self.contexts = {}
        for index in (profiles if profiles is not None else range(count)):
            context = self.engine.create_execution_context()
            if count > 1:
                context.set_optimization_profile_async(index, 0)
            self.contexts[index] = context

    def dtype(self, name: str):
        return NP_DTYPES[self.engine.get_tensor_dtype(name)]

    def profile_shapes(self, name: str, profile: int = 0):
        return self.engine.get_tensor_profile_shape(name, profile)

    def set_inputs(self, shapes: dict[str, tuple[int, ...]], profile: int = 0) -> dict:
        """Set input shapes; return the resulting output shapes."""
        context = self.contexts[profile]
        for name, shape in shapes.items():
            if not context.set_input_shape(name, tuple(shape)):
                raise ValueError(f"{self.path.name}: bad shape {shape} for {name}.")
        return {name: tuple(context.get_tensor_shape(name)) for name in self.outputs}

    def bind(self, addresses: dict[str, int], profile: int = 0) -> None:
        context = self.contexts[profile]
        for name, address in addresses.items():
            context.set_tensor_address(name, int(address))

    def enqueue(self, stream: int, profile: int = 0) -> None:
        if not self.contexts[profile].execute_async_v3(int(stream)):
            raise RuntimeError(f"{self.path.name}: enqueue failed.")


# ------------------------------------------------------- ctypes CUDA runtime

class Cudart:
    """The few libcudart calls the Jetson sender needs."""

    H2D, D2H, D2D = 1, 2, 3

    def __init__(self) -> None:
        candidates = [os.environ.get("CUDART_PATH"), "/usr/local/cuda/lib64/libcudart.so",
                      ctypes.util.find_library("cudart")]
        for candidate in filter(None, candidates):
            try:
                self.lib = ctypes.CDLL(candidate)
                break
            except OSError:
                continue
        else:
            raise OSError("libcudart not found; set CUDART_PATH.")
        lib = self.lib
        lib.cudaGetErrorString.restype = ctypes.c_char_p
        lib.cudaMemcpyAsync.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_size_t,
                                        ctypes.c_int, ctypes.c_void_p]
        lib.cudaGraphInstantiate.argtypes = [ctypes.POINTER(ctypes.c_void_p), ctypes.c_void_p,
                                             ctypes.c_ulonglong]
        lib.cudaEventElapsedTime.argtypes = [ctypes.POINTER(ctypes.c_float), ctypes.c_void_p,
                                             ctypes.c_void_p]

    def check(self, status: int) -> None:
        if status != 0:
            raise RuntimeError(f"CUDA error {status}: {self.lib.cudaGetErrorString(status).decode()}")

    def malloc(self, nbytes: int) -> int:
        pointer = ctypes.c_void_p()
        self.check(self.lib.cudaMalloc(ctypes.byref(pointer), ctypes.c_size_t(nbytes)))
        return pointer.value

    def host_alloc(self, nbytes: int) -> int:
        pointer = ctypes.c_void_p()
        self.check(self.lib.cudaHostAlloc(ctypes.byref(pointer), ctypes.c_size_t(nbytes), 0))
        return pointer.value

    def stream(self) -> int:
        stream = ctypes.c_void_p()
        self.check(self.lib.cudaStreamCreateWithFlags(ctypes.byref(stream), 1))
        return stream.value

    def memcpy_async(self, dst: int, src: int, nbytes: int, kind: int, stream: int) -> None:
        self.check(self.lib.cudaMemcpyAsync(dst, src, nbytes, kind, stream))

    def sync(self, stream: int) -> None:
        self.check(self.lib.cudaStreamSynchronize(ctypes.c_void_p(stream)))

    def event(self) -> int:
        event = ctypes.c_void_p()
        self.check(self.lib.cudaEventCreate(ctypes.byref(event)))
        return event.value

    def record(self, event: int, stream: int) -> None:
        self.check(self.lib.cudaEventRecord(ctypes.c_void_p(event), ctypes.c_void_p(stream)))

    def elapsed_ms(self, start: int, end: int) -> float:
        value = ctypes.c_float()
        self.check(self.lib.cudaEventSynchronize(ctypes.c_void_p(end)))
        self.check(self.lib.cudaEventElapsedTime(ctypes.byref(value), start, end))
        return value.value

    def capture(self, stream: int, work) -> int:
        """Capture ``work()`` on ``stream`` into an instantiated CUDA graph."""
        graph, executable = ctypes.c_void_p(), ctypes.c_void_p()
        self.check(self.lib.cudaStreamBeginCapture(ctypes.c_void_p(stream), 1))  # thread-local
        try:
            work()
        finally:
            status = self.lib.cudaStreamEndCapture(ctypes.c_void_p(stream), ctypes.byref(graph))
        self.check(status)
        self.check(self.lib.cudaGraphInstantiate(ctypes.byref(executable), graph, 0))
        return executable.value

    def launch(self, executable: int, stream: int) -> None:
        self.check(self.lib.cudaGraphLaunch(ctypes.c_void_p(executable), ctypes.c_void_p(stream)))


def host_array(address: int, shape, dtype) -> np.ndarray:
    """numpy view of pinned host memory."""
    dtype = np.dtype(dtype)
    count = int(np.prod(shape)) * dtype.itemsize
    buffer = (ctypes.c_uint8 * count).from_address(address)
    return np.frombuffer(buffer, dtype=dtype).reshape(shape)
