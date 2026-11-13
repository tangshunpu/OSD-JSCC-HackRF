"""Export the OSD-JSCC-P sender and receiver graphs to ONNX (fp32 weights).

Run from the repo root (needs the ``onnx`` package, which the project venv
does not declare):

    uv run --with onnx==1.19.1 python experiments/deploy/export_onnx.py

Writes into ``--out-dir`` (default ``runs/deploy/osd_jscc_p``):

- ``onnx/sender_<H>x<W>/model.onnx``: static-shape sender, one per image size
  (the AGX engine is built per size for the lowest single-image latency);
- ``onnx/{rx_jscc,unet_step,vae_decode}/model.onnx``: receiver graphs with a
  dynamic batch and spatial size (engines pick shapes through profiles);
- ``meta.json``: rates, layer channel ranges, routing thresholds and sigmas;
- ``reference/sender_kodak_<snr>dB.npz``: PyTorch fp32 sender outputs for every
  Kodak image, used to check the AGX engines without torch on the device.

Before exporting, the wrappers in ``graphs.py`` are checked against the
model's own ``encode_source``/``transmit``/``receive`` in fp32.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
import time

import numpy as np
import torch
from safetensors import safe_open
from safetensors.torch import load_file

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from osd_jscc_p.models import MultirateOSDV3  # noqa: E402
from graphs import (  # noqa: E402
    ReceiverJSCC, Sender, UNetStep, VAEDecode, layer_channel_groups,
    merge_unet_lora, routing_constants,
)

DEFAULT_CHECKPOINT = "checkpoints/osd_jscc_p_awgn.safetensors"
PARTS = ("sender", "rx_jscc", "unet_step", "vae_decode")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--checkpoint", default=DEFAULT_CHECKPOINT)
    parser.add_argument("--sd-model", default="stabilityai/sd-turbo")
    parser.add_argument("--reference-dir", type=Path,
                        help="Optional Kodak directory for sender reference outputs.")
    parser.add_argument("--out-dir", type=Path, default=Path("runs/deploy/osd_jscc_p"))
    parser.add_argument("--parts", nargs="+", choices=PARTS, default=list(PARTS))
    parser.add_argument("--sender-sizes", nargs="+", default=["512x768", "768x512"],
                        help="HxW of the static sender graphs (Kodak landscape/portrait).")
    parser.add_argument("--reference-snr", type=float, default=12.0)
    parser.add_argument("--opset", type=int, default=17)
    parser.add_argument("--device", default="cuda:0")
    return parser.parse_args()


def load_rgb(path: Path) -> np.ndarray:
    from PIL import Image

    with Image.open(path) as image:
        return np.asarray(image.convert("RGB"), dtype=np.uint8).copy()


def export(module, inputs, path: Path, input_names, output_names, dynamic_axes, opset):
    import onnx

    path.parent.mkdir(parents=True, exist_ok=True)
    for stale in (path, path.with_name(path.name + ".data")):
        stale.unlink(missing_ok=True)  # onnx.save appends to an existing data file
    start = time.perf_counter()
    raw = path.with_name("raw") / path.name
    raw.parent.mkdir(exist_ok=True)
    torch.onnx.export(
        module, inputs, str(raw), input_names=input_names, output_names=output_names,
        dynamic_axes=dynamic_axes or None, opset_version=opset, do_constant_folding=True,
        dynamo=False,
    )
    # The legacy exporter writes one file per initializer for >2 GB graphs;
    # consolidate into a single external-data file.
    model = onnx.load(str(raw))
    onnx.save(model, str(path), save_as_external_data=True, all_tensors_to_one_file=True,
              location=path.name + ".data", size_threshold=1024)
    for item in raw.parent.iterdir():
        item.unlink()
    raw.parent.rmdir()
    size = sum(item.stat().st_size for item in path.parent.iterdir()) / 2**20
    print(f"  {path} ({size:.0f} MiB, {time.perf_counter() - start:.0f} s)", flush=True)


@torch.no_grad()
def check_wrappers(model, device, snr_db: float) -> None:
    """Compare wrapper graphs and model methods on a synthetic image."""
    generator = torch.Generator(device="cpu").manual_seed(42)
    image = torch.randint(0, 256, (1, 128, 128, 3), dtype=torch.uint8, generator=generator).to(device)
    snr = torch.full((1,), snr_db, device=device)
    sender = Sender(model, out_dtype=torch.float32).eval()
    symbols = sender(image, snr)

    _, codeword, _ = model.encode_source(image.permute(0, 3, 1, 2).float() / 127.5 - 1.0, snr_db)
    channel_type = model.channel.channel_type
    model.channel.channel_type = "none"
    try:
        expected = model.transmit(codeword, snr, model.rates[-1])
    finally:
        model.channel.channel_type = channel_type
    print(f"  sender vs transmit(no noise): max |diff| {(symbols - expected).abs().max():.2e}")

    routing = routing_constants(model)
    received = symbols + torch.randn_like(symbols) * (0.5 * 10 ** (-snr_db / 10)) ** 0.5
    rx, unet_step, vae = ReceiverJSCC(model), UNetStep(model), VAEDecode(model)
    for rate in model.rates:
        channels = model.sent_channels(rate)
        prefix = received[:, :channels]
        zero_filled = torch.zeros_like(received)
        zero_filled[:, :channels] = prefix
        rate_t = torch.full((1,), float(rate), device=device)
        latent, conditioning = rx(zero_filled.half(), snr, rate_t)
        route = 249 if snr_db >= routing["rate_thresholds"][str(rate)] else 499
        timestep = torch.full((1,), float(route), device=device)
        sigma = torch.full((1,), routing["sigmas"][str(route)], device=device)
        ours = vae(unet_step(latent, conditioning, timestep, sigma))[0]
        theirs = model.receive(prefix.half().float(), snr, rate)[0]
        theirs = theirs.add(1.0).mul(127.5).round().clamp(0, 255).to(torch.uint8).permute(1, 2, 0)
        diff = (ours.int() - theirs.int()).abs()
        print(f"  receiver CR{rate}: max |diff| {int(diff.max())} levels, "
              f"mean {diff.float().mean():.4f}")


def main() -> None:
    args = parse_args()
    device = torch.device(args.device)
    torch.backends.mha.set_fastpath_enabled(False)  # no ONNX symbolic for the fused MHA op
    out = args.out_dir
    with safe_open(args.checkpoint, framework="pt", device="cpu") as handle:
        metadata = handle.metadata() or {}
    if metadata.get("format") != "osd_jscc_p_weights_v1":
        raise ValueError("Expected an OSD-JSCC-P safetensors checkpoint")
    config = json.loads(metadata["model_config"])
    model = MultirateOSDV3.from_config(config, sd_model_id=args.sd_model, torch_dtype=torch.float32)
    result = model.load_state_dict(load_file(args.checkpoint), strict=False)
    missing = sorted(model.persistent_keys().intersection(result.missing_keys))
    if missing or result.unexpected_keys:
        raise RuntimeError(f"Checkpoint mismatch: missing={missing}, unexpected={result.unexpected_keys}")
    model = model.to(device)
    model.eval().requires_grad_(False)
    model.configure_inference(diffusion_steps=1, timestep_policy="adaptive")

    print("checking wrappers against the model (fp32)")
    check_wrappers(model, device, args.reference_snr)
    merged = merge_unet_lora(model.unet)
    print(f"merged {merged} UNet LoRA layers; re-checking")
    check_wrappers(model, device, args.reference_snr)

    groups = layer_channel_groups(model)
    meta = {
        "checkpoint": str(args.checkpoint),
        "step": metadata.get("step"),
        "model_config": config,
        "c_max": model.c_max,
        "rates": list(model.rates),
        "layer_channels": groups,
        "sent_channels": {str(rate): model.sent_channels(rate) for rate in model.rates},
        "vae_scaling_factor": float(model.vae.config.scaling_factor),
        "context_tokens": model.semantic_tokens + 1,
        "context_dim": int(model.jscc_decoder.context_dim),
        "symbol_downsample": 16,
        "opset": args.opset,
        **routing_constants(model),
    }
    out.mkdir(parents=True, exist_ok=True)
    (out / "meta.json").write_text(json.dumps(meta, indent=2) + "\n")

    if "sender" in args.parts:
        sender = Sender(model).eval()
        for size in args.sender_sizes:
            height, width = map(int, size.split("x"))
            image = torch.zeros(1, height, width, 3, dtype=torch.uint8, device=device)
            snr = torch.full((1,), args.reference_snr, device=device)
            export(sender, (image, snr), out / "onnx" / f"sender_{height}x{width}" / "model.onnx",
                   ["image", "snr_db"], ["symbols"], None, args.opset)

        # fp32 references for the on-device check (no torch on the AGX).
        reference = Sender(model, out_dtype=torch.float32).eval()
        names, arrays = [], {}
        snr = torch.full((1,), args.reference_snr, device=device)
        with torch.no_grad():
            for path in sorted(args.reference_dir.glob("kodim*.png")) if args.reference_dir else []:
                image = torch.from_numpy(load_rgb(path))[None].to(device)
                arrays[path.stem] = reference(image, snr)[0].cpu().numpy()
                names.append(path.stem)
        ref_path = out / "reference" / f"sender_kodak_{args.reference_snr:g}dB.npz"
        ref_path.parent.mkdir(parents=True, exist_ok=True)
        np.savez(ref_path, **arrays)
        print(f"  {ref_path} ({len(names)} images)")

    batch, h, w = 2, 32, 48  # example shapes only; every axis below is dynamic
    if "rx_jscc" in args.parts:
        inputs = (torch.zeros(batch, model.c_max, h, w, dtype=torch.float16, device=device),
                  torch.full((batch,), 12.0, device=device),
                  torch.full((batch,), 96.0, device=device))
        export(ReceiverJSCC(model).eval(), inputs, out / "onnx" / "rx_jscc" / "model.onnx",
               ["received", "snr_db", "rate"], ["latent", "conditioning"],
               {"received": {0: "B", 2: "h", 3: "w"}, "snr_db": {0: "B"}, "rate": {0: "B"},
                "latent": {0: "B", 2: "H8", 3: "W8"}, "conditioning": {0: "B"}}, args.opset)
    if "unet_step" in args.parts:
        inputs = (torch.zeros(batch, 4, 2 * h, 2 * w, device=device),
                  torch.zeros(batch, meta["context_tokens"], meta["context_dim"], device=device),
                  torch.full((batch,), 249.0, device=device),
                  torch.full((batch,), meta["sigmas"]["249"], device=device))
        export(UNetStep(model).eval(), inputs, out / "onnx" / "unet_step" / "model.onnx",
               ["latent", "conditioning", "timestep", "sigma"], ["vae_latent"],
               {"latent": {0: "B", 2: "H8", 3: "W8"}, "conditioning": {0: "B"},
                "timestep": {0: "B"}, "sigma": {0: "B"},
                "vae_latent": {0: "B", 2: "H8", 3: "W8"}}, args.opset)
    if "vae_decode" in args.parts:
        inputs = (torch.zeros(batch, 4, 2 * h, 2 * w, device=device),)
        export(VAEDecode(model).eval(), inputs, out / "onnx" / "vae_decode" / "model.onnx",
               ["vae_latent"], ["image"],
               {"vae_latent": {0: "B", 2: "H8", 3: "W8"}, "image": {0: "B", 1: "H", 2: "W"}},
               args.opset)


if __name__ == "__main__":
    main()
