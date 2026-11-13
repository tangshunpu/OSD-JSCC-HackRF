"""Transmitter and receiver sub-graphs of OSD-JSCC-P for ONNX/TensorRT export.

The graphs re-express ``OSDJSCCP`` with the
host-side glue folded in, so each TensorRT engine starts and ends at a buffer
the application already has:

- ``Sender``: uint8 RGB image [B,H,W,3] + SNR -> unit-power channel symbols of
  all C_max channels [B,C_max,H/16,W/16] (fp16). Layers cover whole, contiguous
  channels, so a transmission at any rate is a prefix of this buffer and every
  layer is already power-normalized (``power_norm='layer'``).
- ``ReceiverJSCC``: received symbols (zero-filled to C_max) + SNR + CR ->
  deterministic latent (clamped to [-8, 8]) and the UNet context tokens.
- ``UNetStep``: one routed SD-Turbo step (Euler, trailing spacing, epsilon
  prediction) -> VAE-scaled x0. Timestep and sigma are per-sample inputs, so a
  batch may mix SNRs/rates that route to different timesteps.
- ``VAEDecode``: VAE decoder -> uint8 RGB image [B,H,W,3].
"""

from __future__ import annotations

import torch
from torch import nn


def layer_channel_groups(model) -> list[tuple[int, int]]:
    """Channel range [start, end) of every layer, in transmission order."""
    groups, start = [], 0
    for end in model.boundaries:
        if start % 2 or end % 2:
            raise ValueError("Deployment needs layers that cover whole channels.")
        groups.append((start // 2, end // 2))
        start = end
    return groups


def merge_unet_lora(unet: nn.Module) -> int:
    """Merge every PEFT LoRA layer into its base layer and drop the wrapper."""
    from peft.tuners.tuners_utils import BaseTunerLayer

    merged = 0
    for name, module in list(unet.named_modules()):
        if isinstance(module, BaseTunerLayer):
            module.merge()
            parent_name, _, child = name.rpartition(".")
            parent = unet.get_submodule(parent_name) if parent_name else unet
            setattr(parent, child, module.get_base_layer())
            merged += 1
    return merged


def _vae_scale_shift(model) -> tuple[float, float]:
    config = model.vae.config
    return (float(getattr(config, "scaling_factor", 0.18215)),
            float(getattr(config, "shift_factor", 0.0) or 0.0))


class Sender(nn.Module):
    def __init__(self, model, out_dtype: torch.dtype = torch.float16) -> None:
        super().__init__()
        if model.power_norm != "layer":
            raise ValueError("The deployed sender assumes per-layer power normalization.")
        self.vae_encoder = model.vae.encoder
        self.quant_conv = model.vae.quant_conv
        self.adapter = model.latent_adapter
        self.snr_embedding = model.snr_embedding
        self.jscc_encoder = model.jscc_encoder
        self.scale, self.shift = _vae_scale_shift(model)
        self.groups = layer_channel_groups(model)
        self.out_dtype = out_dtype

    def forward(self, image: torch.Tensor, snr_db: torch.Tensor) -> torch.Tensor:
        # TensorRT allows uint8 only on network I/O directly next to a Cast.
        x = image.float().permute(0, 3, 1, 2) / 127.5 - 1.0
        moments = self.vae_encoder(x)
        if self.quant_conv is not None:
            moments = self.quant_conv(moments)
        latent = (moments[:, :4] - self.shift) * self.scale  # posterior mode
        codeword = self.jscc_encoder(self.adapter(latent), self.snr_embedding(snr_db))
        codeword = codeword.float()
        layers = []
        for start, end in self.groups:
            part = codeword[:, start:end]
            power = 2.0 * part.square().mean(dim=(1, 2, 3), keepdim=True)
            layers.append(part / power.clamp_min(1e-12).sqrt())
        return torch.cat(layers, dim=1).to(self.out_dtype)


class ReceiverJSCC(nn.Module):
    def __init__(self, model) -> None:
        super().__init__()
        if model.rate_embedding is None or model.snr_context is None:
            raise ValueError("Deployment expects the full receiver (rate embedding + SNR token).")
        self.snr_embedding = model.snr_embedding
        self.rate_embedding = model.rate_embedding
        self.snr_context = model.snr_context
        self.decoder = model.jscc_decoder

    def forward(self, received: torch.Tensor, snr_db: torch.Tensor,
                rate: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        condition = self.snr_embedding(snr_db) + self.rate_embedding(rate)
        dec = self.decoder
        x = dec.stem(received.float())
        for block in dec.low_attention_blocks:
            x = block(x, condition)
        for block in dec.low_residual_blocks:
            x = block(x, condition)
        context = dec.context_head(x.mean(dim=(-2, -1))).reshape(
            x.shape[0], dec.semantic_tokens, dec.context_dim)
        x = dec.upsample(x)
        for block in dec.high_blocks:
            x = block(x, condition)
        latent = dec.mean_head(x).float().clamp(-8.0, 8.0)
        conditioning = torch.cat((context, self.snr_context(condition)[:, None]), dim=1)
        return latent, conditioning


class UNetStep(nn.Module):
    def __init__(self, model) -> None:
        super().__init__()
        self.unet = model.unet
        self.scale, self.shift = _vae_scale_shift(model)

    def forward(self, latent: torch.Tensor, conditioning: torch.Tensor,
                timestep: torch.Tensor, sigma: torch.Tensor) -> torch.Tensor:
        sigma = sigma[:, None, None, None]
        eps = self.unet(latent / torch.sqrt(sigma * sigma + 1.0), timestep,
                        encoder_hidden_states=conditioning, return_dict=False)[0]
        return (latent - sigma * eps.float()) / self.scale + self.shift


class VAEDecode(nn.Module):
    def __init__(self, model) -> None:
        super().__init__()
        self.post_quant_conv = model.vae.post_quant_conv
        self.decoder = model.vae.decoder

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        if self.post_quant_conv is not None:
            z = self.post_quant_conv(z)
        x = self.decoder(z).float().clamp(-1.0, 1.0)
        x = ((x + 1.0) * 127.5).round().clamp(0.0, 255.0)
        return x.permute(0, 2, 3, 1).to(torch.uint8)


def routing_constants(model) -> dict:
    """Per-rate SNR thresholds and the (timestep, sigma) of both routes."""
    scheduler = model.scheduler
    sigmas = {}
    for timestep in (249, 499):
        scheduler.set_timesteps(timesteps=[timestep], device="cpu")
        if float(scheduler.timesteps[0]) != timestep:
            raise RuntimeError("Unexpected scheduler timestep.")
        sigmas[str(timestep)] = float(scheduler.sigmas[0])
    if scheduler.config.prediction_type != "epsilon":
        raise RuntimeError("UNetStep assumes epsilon prediction.")
    return {
        "rate_thresholds": {str(rate): float(model.rate_thresholds[rate]) for rate in model.rates},
        "high_timestep": 249, "low_timestep": 499, "sigmas": sigmas,
    }
