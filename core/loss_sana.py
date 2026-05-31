"""
SANA flow-matching training loss using 🤗 diffusers.

No SANA repo clone required — all scheduler maths is self-contained here.

Theory
------
Flow matching uses the forward (noising) process:

    z_t = (1 - σ) · z₀  +  σ · ε       ε ~ N(0, I),  σ ∈ [0, 1]

The transformer is trained to predict the velocity  v = ε − z₀:

    loss = MSE(transformer(z_t, timestep=σ·T), ε − z₀)

where T = num_train_timesteps (default 1000).
"""
from __future__ import annotations

import torch
import torch.nn.functional as F


def sample_flow_sigmas(
    batch_size: int,
    weighting_scheme: str,
    logit_mean: float,
    logit_std: float,
    device: torch.device,
    num_train_timesteps: int = 1000,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Samples a σ ∈ (0, 1) for each image and converts it to a continuous timestep.

    Args:
        batch_size          : Number of samples.
        weighting_scheme    : "uniform" or "logit_normal".
        logit_mean          : Mean for logit-normal sampling (ignored for uniform).
        logit_std           : Std  for logit-normal sampling (ignored for uniform).
        device              : Target device.
        num_train_timesteps : T — scales σ to the transformer's timestep range.

    Returns:
        sigma      (B,) float — noise fraction ∈ (0, 1)
        timestep_t (B,) float — σ · T, the value passed to transformer.forward
    """
    if weighting_scheme == "logit_normal":
        u = torch.sigmoid(
            torch.randn(batch_size, device=device) * logit_std + logit_mean
        )
    else:
        u = torch.rand(batch_size, device=device)

    timestep_t = u * num_train_timesteps
    return u, timestep_t


def compute_sana_loss(
    transformer: torch.nn.Module,
    z: torch.Tensor,
    y: torch.Tensor,
    y_mask: torch.Tensor,
    sigma: torch.Tensor,
    timestep_t: torch.Tensor,
) -> torch.Tensor:
    """
    One flow-matching forward pass through the SANA transformer.

    Args:
        transformer : SanaTransformer2DModel (the trained component).
        z           : Clean VAE latents,  shape (B, C, H, W).
        y           : Text embeddings,    shape (B, N, C_text).
        y_mask      : Attention mask,     shape (B, N).
        sigma       : Noise fraction,     shape (B,)   — sampled by sample_flow_sigmas.
        timestep_t  : Continuous timestep shape (B,)   — sampled by sample_flow_sigmas.

    Returns:
        Scalar loss Tensor with grad attached.
    """
    noise = torch.randn_like(z)

    # Forward process: z_t = (1 - σ) · z₀ + σ · ε
    sigma_view = sigma.view(-1, 1, 1, 1).to(z.dtype)
    noisy_z = (1.0 - sigma_view) * z + sigma_view * noise

    # Velocity prediction
    model_pred = transformer(
        hidden_states=noisy_z,
        encoder_hidden_states=y,
        timestep=timestep_t,
        encoder_attention_mask=y_mask,
    ).sample

    # Flow-matching velocity target: v = ε − z₀
    target = noise - z

    return F.mse_loss(model_pred.float(), target.float())
