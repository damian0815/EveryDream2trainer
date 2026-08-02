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

from typing import Tuple

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
    noise: torch.Tensor = None,
    slice_size: int = None,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    One flow-matching forward pass through the SANA transformer.

    Supports both 4D latents (B, C, H, W) for images and 5D latents
    (B, C, F, H, W) for video.

    Args:
        transformer     : SanaTransformer2DModel or SanaVideoTransformer3DModel
        noise_scheduler : TrainFlowMatchEulerDiscreteScheduler — used for noising.
        z               : Clean VAE latents,  shape (B, C, H, W) or (B, C, F, H, W).
        y               : Text embeddings,    shape (B, N, C_text).
        y_mask          : Attention mask,     shape (B, N).
        timesteps       : Float timestep values, shape (B,) — produced by
                          TrainFlowMatchEulerDiscreteScheduler.get_shifted_timesteps().
        slice_size      : Max size of internal processing batch slices (saves VRAM)

    Returns:
        Tuple of (loss_1d, model_pred, target).
        loss  : full loss, shape (B, C, H, W) or (B, C, F, H, W).
        model_pred: model prediction, same shape as z.
        target   : flow-matching target (noise - z), same shape as z.
    """
    assert z.ndim in (4, 5), f"Expected 4D (image) or 5D (video) latents, got shape {z.shape}"

    if slice_size is not None:
        results_loss = []
        results_pred = []
        results_target = []
        for slice_start in range(0, z.shape[0], slice_size):
            slice_end = slice_start + slice_size
            loss_s, pred_s, target_s = compute_sana_loss(
                transformer=transformer,
                noise_scheduler=noise_scheduler,
                z=z[slice_start:slice_end],
                y=y[slice_start:slice_end],
                y_mask=y_mask[slice_start:slice_end],
                timesteps=timesteps[slice_start:slice_end],
                noise=noise[slice_start:slice_end] if noise is not None else None,
                slice_size=None
            )
            results_loss.append(loss_s)
            results_pred.append(pred_s)
            results_target.append(target_s)
        return torch.cat(results_loss, dim=0), torch.cat(results_pred, dim=0), torch.cat(results_target, dim=0)

    if noise is None:
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

    # return full loss
    loss_full = F.mse_loss(model_pred.float(), target.float(), reduction='none')
    return loss_full, model_pred, target


def compute_sana_dpo_loss(
    policy_transformer: torch.nn.Module,
    reference_transformer: torch.nn.Module,
    noise_scheduler,
    z_good: torch.Tensor,
    z_bad: torch.Tensor,
    y: torch.Tensor,
    y_mask: torch.Tensor,
    timesteps: torch.Tensor,
    noise: torch.Tensor,
    beta: float = 0.1,
    slice_size: int = None,
    model_pred_good: torch.Tensor = None,
    target_good: torch.Tensor = None,
) -> Tuple[torch.Tensor, dict]:
    """
    Diffusion-DPO loss for SANA flow-matching.

    Trains the policy model to prefer good images over bad images using paired
    preference data. The reference model (frozen EMA) provides a baseline.

    Args:
        policy_transformer   : trainable transformer
        reference_transformer: frozen EMA transformer (no grad)
        noise_scheduler      : TrainFlowMatchEulerDiscreteScheduler
        z_good               : clean VAE latents for good images, (n, C, H, W)
        z_bad                : clean VAE latents for bad images,  (n, C, H, W)
        y                    : text embeddings,     (n, N, C_text)
        y_mask               : attention mask,      (n, N)
        timesteps            : float timesteps,     (n,)
        noise                : shared noise tensor, (n, C, H, W)
        beta                 : DPO temperature/scale factor
        slice_size           : max samples per forward slice (saves VRAM)
        model_pred_good      : precomputed policy-good prediction (avoids duplicate forward)
        target_good          : precomputed velocity target for good images

    Returns:
        (loss_1d, info) where loss_1d is shape (n,) and info is a dict of diagnostics.
    """
    if slice_size is not None:
        results_loss = []
        results_info = {k: [] for k in ("dpo_signal", "err_policy_good", "err_policy_bad",
                                         "err_ref_good", "err_ref_bad")}
        for start in range(0, z_good.shape[0], slice_size):
            end = start + slice_size
            loss_s, info_s = compute_sana_dpo_loss(
                policy_transformer, reference_transformer, noise_scheduler,
                z_good[start:end], z_bad[start:end],
                y[start:end], y_mask[start:end],
                timesteps[start:end], noise[start:end],
                beta=beta, slice_size=None,
                model_pred_good=model_pred_good[start:end] if model_pred_good is not None else None,
                target_good=target_good[start:end] if target_good is not None else None,
            )
            results_loss.append(loss_s)
            for k in results_info:
                results_info[k].append(info_s[k])
        return (torch.cat(results_loss, dim=0),
                {k: torch.cat(v, dim=0) for k, v in results_info.items()})

    # Policy forward on good images — reuse if caller already computed it
    if model_pred_good is not None:
        pred_policy_good = model_pred_good
    else:
        _, pred_policy_good, target_good = compute_sana_loss(
            policy_transformer, noise_scheduler,
            z_good, y, y_mask, timesteps, noise=noise,
        )

    # Policy forward on bad images
    _, pred_policy_bad, target_bad = compute_sana_loss(
        policy_transformer, noise_scheduler,
        z_bad, y, y_mask, timesteps, noise=noise,
    )

    # Reference forward on good + bad images (frozen, no grad)
    with torch.no_grad():
        _, pred_ref_good, _ = compute_sana_loss(
            reference_transformer, noise_scheduler,
            z_good, y, y_mask, timesteps, noise=noise,
        )
        _, pred_ref_bad, _ = compute_sana_loss(
            reference_transformer, noise_scheduler,
            z_bad, y, y_mask, timesteps, noise=noise,
        )

    # Per-sample MSE errors (reduce over spatial dims)
    spatial_dims = list(range(1, z_good.ndim))

    err_policy_good = F.mse_loss(pred_policy_good.float(), target_good.float(),
                                 reduction='none').mean(dim=spatial_dims)
    err_policy_bad = F.mse_loss(pred_policy_bad.float(), target_bad.float(),
                                reduction='none').mean(dim=spatial_dims)
    err_ref_good = F.mse_loss(pred_ref_good.float(), target_good.float(),
                              reduction='none').mean(dim=spatial_dims)
    err_ref_bad = F.mse_loss(pred_ref_bad.float(), target_bad.float(),
                             reduction='none').mean(dim=spatial_dims)

    # DPO signal and loss
    dpo_signal = -beta * ((err_policy_good - err_ref_good) - (err_policy_bad - err_ref_bad))
    loss_1d = -F.logsigmoid(dpo_signal)

    info = {
        "dpo_signal": dpo_signal.detach(),
        "err_policy_good": err_policy_good.detach(),
        "err_policy_bad": err_policy_bad.detach(),
        "err_ref_good": err_ref_good.detach(),
        "err_ref_bad": err_ref_bad.detach(),
    }
    return loss_1d, info
