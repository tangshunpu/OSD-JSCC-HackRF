"""Build a TensorRT engine for one OSD-JSCC-P graph.

Runs on both ends (no torch needed). Examples, from the repo root:

    # AGX Orin sender, static 512x768 (HxW), FP16
    python3 experiments/deploy/build_engine.py --graph sender --size 512x768 --precision fp16

    # RTX receiver UNet: a batch-1 profile for latency and a 1..32 profile for throughput
    python experiments/deploy/build_engine.py --graph unet_step --size 512x768 \
        --precision fp16 --batch 1:1:1 --batch 1:16:32

The engine goes to ``<root>/engines/<tag>/<graph>_<H>x<W>_<precision>.plan`` with a
JSON build record next to it. ``--tag`` names the device (``agx_orin``, ``rtx6000``).
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import socket
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent))
from trt_runtime import build_engine, trt  # noqa: E402

VIDEO = ("video_key_sender", "video_p_sender", "video_rx_jscc", "video_predictor", "video_refiner", "video_decode")
GRAPHS = ("sender", "rx_jscc", "unet_step", "vae_decode", "witt_decode", *VIDEO)
# one ONNX per size, batch 1 (witt_decode: SwinJSCC baseline; video_*: experiments/tiny_stream/export_video_deploy.py,
# --root runs/deploy/tiny_video)
STATIC = ("sender", "witt_decode", *VIDEO)


def input_shapes(graph: str, meta: dict, batch: int, height: int, width: int) -> dict:
    h16, w16, h8, w8 = height // 16, width // 16, height // 8, width // 8
    if graph == "rx_jscc":
        return {"received": (batch, meta["c_max"], h16, w16), "snr_db": (batch,), "rate": (batch,)}
    if graph == "unet_step":
        return {"latent": (batch, 4, h8, w8),
                "conditioning": (batch, meta["context_tokens"], meta["context_dim"]),
                "timestep": (batch,), "sigma": (batch,)}
    if graph == "vae_decode":
        return {"vae_latent": (batch, 4, h8, w8)}
    raise ValueError(graph)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--graph", choices=GRAPHS, required=True)
    parser.add_argument("--size", required=True, help="Image HxW, e.g. 512x768.")
    parser.add_argument("--precision", choices=("fp16", "bf16", "fp32"), default="fp16")
    parser.add_argument("--batch", action="append", default=None,
                        help="min:opt:max batch of one profile (repeatable; receiver graphs).")
    parser.add_argument("--root", type=Path, default=Path("runs/deploy/osd_jscc_p"))
    parser.add_argument("--tag", default=socket.gethostname())
    parser.add_argument("--opt-level", type=int, default=3)
    parser.add_argument("--fp32-norms", action="store_true",
                        help="Pin normalization/reduce layers to fp32.")
    parser.add_argument("--suffix", default="", help="Extra engine-name suffix.")
    parser.add_argument("--out", type=Path)
    args = parser.parse_args()

    height, width = map(int, args.size.lower().split("x"))
    meta = json.loads((args.root / "meta.json").read_text())
    if args.graph in STATIC:
        onnx_path = args.root / "onnx" / f"{args.graph}_{height}x{width}" / "model.onnx"
        profiles = None
        if args.batch:
            raise SystemExit(f"The {args.graph} graph is static (batch 1).")
    else:
        onnx_path = args.root / "onnx" / args.graph / "model.onnx"
        profiles = []
        for spec in args.batch or ["1:1:1"]:
            low, opt, high = map(int, spec.split(":"))
            shapes = [input_shapes(args.graph, meta, b, height, width) for b in (low, opt, high)]
            profiles.append({name: tuple(shape[name] for shape in shapes) for name in shapes[0]})
    name = f"{args.graph}_{height}x{width}_{args.precision}{args.suffix}.plan"
    engine_path = args.out or args.root / "engines" / args.tag / name
    record = build_engine(
        onnx_path, engine_path, precision=args.precision, profiles=profiles,
        opt_level=args.opt_level, fp32_norms=args.fp32_norms,
        timing_cache=engine_path.parent / f"timing_{args.precision}.cache",
    )
    record.update(graph=args.graph, size=[height, width], batch_profiles=args.batch,
                  onnx=str(onnx_path), host=socket.gethostname())
    engine_path.with_suffix(".json").write_text(json.dumps(record, indent=2) + "\n")
    print(json.dumps(record), flush=True)


if __name__ == "__main__":
    main()
