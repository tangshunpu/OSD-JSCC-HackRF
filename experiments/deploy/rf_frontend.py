"""GPU baseband and TensorRT helpers for the live receiver."""

from pathlib import Path
import numpy as np
import torch
from scipy import signal
import ofdm_phy as phy


class TorchBaseband:
    """GPU version of ``ofdm_phy.to_baseband``: the same Kaiser filter and alignment as
    ``scipy.signal.resample_poly(z, 1, UP)`` (the band is centred on the LO: no shift)."""

    def __init__(self, device: str = "cuda") -> None:
        if phy.OFFSET:
            raise ValueError("TorchBaseband expects the LO at the band centre (ofdm_phy.OFFSET = 0).")
        half = 10 * phy.UP
        taps = signal.firwin(2 * half + 1, 1 / phy.UP, window=("kaiser", 5.0))
        self.half, self.device = half, device
        self.weight = torch.tensor(taps, dtype=torch.float32, device=device).view(1, 1, -1)

    @torch.inference_mode()
    def __call__(self, iq: np.ndarray) -> np.ndarray:
        x = torch.from_numpy(np.ascontiguousarray(iq)).to(self.device).float()      # [n, 2]
        re, im = x[:, 0], x[:, 1]
        y = torch.nn.functional.conv1d(torch.stack([re, im])[:, None], self.weight,
                                       stride=phy.UP, padding=self.half)[:, 0]
        y = y.double().cpu().numpy()
        return y[0] + 1j * y[1]


class TRTGraph:
    """One static (batch-1) TensorRT engine with device buffers: host inputs -> host outputs."""

    def __init__(self, path: Path) -> None:
        from receiver import TORCH_DTYPES
        from trt_runtime import Engine

        self.engine = Engine(path, profiles=(0,))
        self.stream = torch.cuda.Stream()
        self.buffers = {name: torch.empty(tuple(self.engine.engine.get_tensor_shape(name)),
                                          dtype=TORCH_DTYPES[np.dtype(self.engine.dtype(name))], device="cuda")
                        for name in self.engine.names}
        self.engine.bind({name: t.data_ptr() for name, t in self.buffers.items()}, 0)

    def __call__(self, **inputs) -> dict:
        with torch.cuda.stream(self.stream):
            for name, value in inputs.items():
                self.buffers[name].copy_(torch.from_numpy(np.ascontiguousarray(value)).to(self.buffers[name].dtype))
            self.engine.enqueue(self.stream.cuda_stream, 0)
            out = {name: self.buffers[name].cpu() for name in self.engine.outputs}
        self.stream.synchronize()
        return {name: value.numpy() for name, value in out.items()}
