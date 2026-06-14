"""
SANA flow-matching training loss.

Timestep sampling is handled upstream by get_multirank_stratified_random_timesteps()
(core/loss.py) + TrainFlowMatchEulerDiscreteScheduler.get_shifted_timesteps()
(core/flow_match_model.py), matching the SD2/SDXL flow-matching training path.

Theory
------
Flow matching forward process:

    z_t = (1 - σ) · z₀  +  σ · ε       ε ~ N(0, I),  σ ∈ [0, 1]

Transformer is trained to predict the velocity  v = ε − z₀:

    loss = MSE(transformer(z_t, timestep=t), ε − z₀)

where t is the float timestep value (σ · T or shift-adjusted equivalent) returned
by TrainFlowMatchEulerDiscreteScheduler.get_shifted_timesteps().
"""
from __future__ import annotations

import logging

import torch
import torch.nn.functional as F


def _log_tensor_stats(name: str, t: torch.Tensor) -> None:
    """Logs a one-line summary of a tensor's value range; warns on NaN/inf."""
    has_nan = t.isnan().any().item()
    has_inf = t.isinf().any().item()
    if has_nan or has_inf:
        logging.warning(
            f"compute_sana_loss: {name} contains "
            f"{'NaN ' if has_nan else ''}{'inf' if has_inf else ''}  "
            f"shape={tuple(t.shape)} dtype={t.dtype}"
        )
    else:
        logging.debug(
            f"compute_sana_loss: {name}  "
            f"min={t.float().min().item():.4f}  "
            f"max={t.float().max().item():.4f}  "
            f"shape={tuple(t.shape)} dtype={t.dtype}"
        )


def compute_sana_loss(
    transformer: torch.nn.Module,
    noise_scheduler,
    z: torch.Tensor,
    y: torch.Tensor,
    y_mask: torch.Tensor,
    timesteps: torch.Tensor,
    slice_size: int=None
) -> torch.Tensor:
    """
    One flow-matching forward pass through the SANA transformer.

    Args:
        transformer     : SanaTransformer2DModel (the trained component).
        noise_scheduler : TrainFlowMatchEulerDiscreteScheduler — used for noising.
        z               : Clean VAE latents,  shape (B, C, H, W).
        y               : Text embeddings,    shape (B, N, C_text).
        y_mask          : Attention mask,     shape (B, N).
        timesteps       : Float timestep values, shape (B,) — produced by
                          TrainFlowMatchEulerDiscreteScheduler.get_shifted_timesteps().
        slice_size      : Max size of internal processing batch slices (saves VRAM)

    Returns:
        Scalar loss Tensor with grad attached.
    """

    if slice_size is not None:
        results = []
        for slice_start in range(0, z.shape[0], slice_size):
            slice_end = slice_start + slice_size
            results.append(compute_sana_loss(
                transformer=transformer,
                noise_scheduler=noise_scheduler,
                z=z[slice_start:slice_end],
                y=y[slice_start:slice_end],
                y_mask=y_mask[slice_start:slice_end],
                timesteps=timesteps[slice_start:slice_end],
                slice_size=None
            ))
        return torch.cat(results, dim=0)

    noise = torch.randn_like(z)

    # Noise via the shared training scheduler path:
    # TrainFlowMatchEulerDiscreteScheduler.add_noise → scale_noise
    #   z_t = (1 - σ) · z₀  +  σ · ε
    noisy_z = noise_scheduler.add_noise(z, noise, timesteps)

    # Probe inputs before entering the transformer
    _log_tensor_stats("z", z)
    _log_tensor_stats("noisy_z", noisy_z)
    _log_tensor_stats("y", y)

    with torch.autocast('cuda', dtype=torch.bfloat16):
        model_pred = transformer(
            hidden_states=noisy_z,
            encoder_hidden_states=y,
            timestep=timesteps,
            encoder_attention_mask=y_mask,
        ).sample

    _log_tensor_stats("model_pred", model_pred)

    # Flow-matching velocity target: v = ε − z₀
    target = noise - z

    # return 1D loss per sample (caller sums/averages)
    mean_dims = list(range(1, len(target.shape)))
    return F.mse_loss(model_pred.float(), target.float(), reduction='none').mean(dim=mean_dims)
