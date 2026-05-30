"""
SANA loss utilities: timestep sampling and forward-pass loss computation.
"""
from __future__ import annotations

import torch


def sample_sana_timesteps(
    batch_size: int,
    train_sampling_steps: int,
    weighting_scheme: str,
    logit_mean: float,
    logit_std: float,
    device: torch.device,
) -> torch.Tensor:
    """
    Draws `batch_size` timestep indices.

    If weighting_scheme is "logit_normal": uses logit-normal distribution
    (via compute_density_for_timestep_sampling).
    Otherwise: uniform random integers in [0, train_sampling_steps).

    Returns a 1D LongTensor of shape (batch_size,).
    """
    if weighting_scheme == "logit_normal":
        from diffusion.model.respace import compute_density_for_timestep_sampling
        u = compute_density_for_timestep_sampling(
            weighting_scheme=weighting_scheme,
            batch_size=batch_size,
            logit_mean=logit_mean,
            logit_std=logit_std,
            mode_scale=None,
        )
        return (u * train_sampling_steps).long().clamp(0, train_sampling_steps - 1).to(device)
    else:
        return torch.randint(0, train_sampling_steps, (batch_size,), device=device)


def compute_sana_loss(
    model,
    z: torch.Tensor,
    y: torch.Tensor,
    y_mask: torch.Tensor,
    timesteps: torch.Tensor,
    data_info: dict,
) -> torch.Tensor:
    """
    Runs one forward pass through the SANA transformer and returns the mean loss.

    model    : SanaTrainingModel
    z        : VAE-encoded latents, shape (B, C, H, W)
    y        : text embeddings from encode_sana_text, shape (B, 1, N, C)
    y_mask   : attention mask from encode_sana_text, shape (B, 1, 1, N)
    timesteps: integer timestep indices, shape (B,)
    data_info: dict with keys "img_hw" and "aspect_ratio" (required by SanaMS forward)

    Returns a scalar Tensor with grad attached.
    """
    loss_dict = model.train_diffusion.training_losses(
        model.transformer,
        z,
        timesteps,
        model_kwargs=dict(y=y, mask=y_mask, data_info=data_info),
    )
    return loss_dict["loss"].mean()

