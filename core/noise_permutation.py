"""
core/noise_permutation.py — Hungarian-matched noise assignment for SANA.

For each batch of clean latents z and freshly-sampled Gaussian noise eps, find the
permutation pi minimizing sum_i ||z_i - eps_{pi(i)}||^2, then diffuse z_i toward
eps_{pi(i)} instead of eps_i.  Because the noise multiset is unchanged (only its
labeling to images shifts), the noise marginal stays exactly Gaussian and inference
sampling is unaffected.

The assignment runs on spatially-pooled, int8-quantized features so the O(B^3)
Hungarian cost stays cheap; the diffusion loss itself still uses full precision.
"""
from scipy.optimize import linear_sum_assignment

import torch
import torch.nn.functional as F


def _pool_for_assignment(latents: torch.Tensor, pool_size: int) -> torch.Tensor:
    """Spatially pool latents to pool_size x pool_size (x pool_size for 5D)."""
    if latents.ndim == 4:
        pooled = F.adaptive_avg_pool2d(latents, (pool_size, pool_size))
    elif latents.ndim == 5:
        pooled = F.adaptive_avg_pool3d(latents, (pool_size, pool_size, pool_size))
    else:
        raise ValueError(
            f"Expected 4D (B, C, H, W) or 5D (B, C, F, H, W) latents, got shape {tuple(latents.shape)}"
        )
    return pooled.reshape(latents.shape[0], -1)


def quantize_int8(tensor: torch.Tensor, scale: float) -> torch.Tensor:
    """Quantize to int8 in [-127, 127] with a shared scale."""
    return torch.clamp(torch.round(tensor * scale), -127, 127).to(torch.int8)


def _build_cost_matrix(
    z_q: torch.Tensor,
    n_q: torch.Tensor,
    max_cost_bytes: int = 200 * 1024 * 1024,
) -> torch.Tensor:
    """
    int32 cost matrix from ||a - b||^2 = ||a||^2 + ||b||^2 - 2 a.b.

    The dot products use int32 elementwise multiply + sum (CUDA has no integer
    matmul: addmm is float-only), which keeps the accumulation exact for the
    pooled/quantized range.  Chunks over noise columns to bound the peak
    B x B x D intermediate.
    """
    b, d = z_q.shape
    z_q = z_q.to(torch.int32)
    n_q = n_q.to(torch.int32)
    z_sq = (z_q ** 2).sum(dim=1)
    n_sq = (n_q ** 2).sum(dim=1)

    max_cols = max(1, max_cost_bytes // (b * d * 4))
    cost = torch.zeros((b, b), dtype=torch.int32, device=z_q.device)
    for s in range(0, b, max_cols):
        e = min(b, s + max_cols)
        dot = (z_q[:, None, :] * n_q[s:e][None, :, :]).sum(dim=-1)
        cost[:, s:e] = z_sq[:, None] + n_sq[None, s:e] - 2 * dot
    return cost


def do_hungarian_noise_permutation(
    clean_latents: torch.Tensor,
    noise: torch.Tensor,
    pool_size: int = 8,
) -> torch.Tensor:
    """
    Returns perm (int64, CPU, shape (B,)) where noise[perm] pairs noise to images,
    minimizing sum_i ||clean_latents[i] - noise[perm[i]]||^2 on pooled quantized features.
    """
    b = clean_latents.shape[0]
    if b <= 1:
        return torch.arange(b, dtype=torch.int64)

    with torch.no_grad():
        z_flat = _pool_for_assignment(clean_latents, pool_size)
        n_flat = _pool_for_assignment(noise, pool_size)

        denom = max(z_flat.abs().max().item(), n_flat.abs().max().item())
        if denom == 0:
            return torch.arange(b, dtype=torch.int64)

        scale = 127.0 / denom
        z_q = quantize_int8(z_flat, scale)
        n_q = quantize_int8(n_flat, scale)

        cost = _build_cost_matrix(z_q, n_q).cpu().numpy()
        _, col_ind = linear_sum_assignment(cost)
    return torch.as_tensor(col_ind, dtype=torch.int64)


def matched_noise_errors(
    clean_latents: torch.Tensor,
    noise: torch.Tensor,
    perm: torch.Tensor,
) -> tuple[float, float]:
    """
    Mean squared per-element distance for the identity and permuted pairings.

    Returns (raw, opt): raw is mean ||z_i - eps_i||^2, opt is mean
    ||z_i - eps_{perm[i]}||^2 on full-precision inputs.  opt <= raw whenever the
    permutation was actually optimal.
    """
    raw = F.mse_loss(clean_latents.float(), noise.float()).item()
    opt = F.mse_loss(clean_latents.float(), noise[perm].float()).item()
    return raw, opt
