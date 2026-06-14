"""
Teacher distillation λ rolloff
==============================

Lightweight module (no heavy training-stack imports) providing the two-sided
cosine teacher-lambda rolloff used during cross-VAE distillation training.

Public symbols imported by ``core.step`` and testable in isolation:

    TEACHER_LAMBDA_SNR_ZERO_LO   – SNR ≤ this → lambda = 0  (pure-noise outer edge)
    TEACHER_LAMBDA_SNR_FULL_LO   – SNR ≥ this → lambda = base (high-noise inner edge)
    TEACHER_LAMBDA_SNR_FULL_HI   – SNR ≤ this → lambda = base (low-noise inner edge)
    TEACHER_LAMBDA_SNR_ZERO_HI   – SNR ≥ this → lambda = 0  (fine-detail outer edge)
    get_teacher_lambda(timesteps, args, noise_scheduler=None) -> torch.Tensor

Physics
-------
Specifying cutoffs in SNR (rather than raw timestep integers) keeps the
boundaries scheduler-agnostic: the same SNR corresponds to the same mixture
of signal vs. noise regardless of whether the student is trained with FM
(shift=1), shifted FM (shift=2), or DDPM.

For reference, the SNR boundaries map to the following integer timesteps:

    FM shift=1 (σ = t/1000):
        SNR_ZERO_HI = 16.0  →  t ≈ 200   (pure detail, interposer adds only blur)
        SNR_FULL_HI =  2.25 →  t ≈ 400   (fine-detail boundary, inner)
        SNR_FULL_LO =  0.065→  t ≈ 797   (composition boundary, inner)
        SNR_ZERO_LO =  0.003→  t ≈ 948   (pure noise, teacher is uninformative)

    Linear-β DDPM (betas linspace(1e-4, 0.02)):
        SNR_ZERO_HI = 16.0  →  t ≈  30   (very clean; DDPM teacher more reliable, kept active longer)
        SNR_FULL_HI =  2.25 →  t ≈ 230   (DDPM has lower SNR per integer step)
        SNR_FULL_LO =  0.065→  t ≈ 680
        SNR_ZERO_LO =  0.003→  t ≈ 860

Both transitions use a smooth half-cosine (Hann) window applied in
**log-SNR space** so the rolloff is symmetric in the natural scale of
diffusion model noise levels.

Convention: higher SNR = cleaner latent = lower noise timestep.
"""

from __future__ import annotations

import math

import torch


# ---------------------------------------------------------------------------
# Configuration constants  (SNR-space)
# ---------------------------------------------------------------------------

#: SNR at or below which teacher lambda falls to zero (pure-noise outer edge).
#: Below this, v ≈ ε for any architecture; teacher signal is redundant with GT.
TEACHER_LAMBDA_SNR_ZERO_LO: float = 1e-9

#: SNR at or above which teacher lambda is at full weight (high-noise inner edge).
#: Mode-selection regime begins; teacher's structural prior is high-value here.
TEACHER_LAMBDA_SNR_FULL_LO: float = 1e-8

#: SNR at or below which teacher lambda is at full weight (low-noise inner edge).
#: Above this, x_1_hat is near-clean and HF-smoothing in the VAE roundtrip dominates the target.
TEACHER_LAMBDA_SNR_FULL_HI: float = 1.0

#: SNR at or above which teacher lambda falls to zero (fine-detail outer edge).
#: At SNR > 2, the target is essentially a smoothed clean latent and the student
#: should learn HF from GT, not from teacher.
TEACHER_LAMBDA_SNR_ZERO_HI: float = 2.0


# ---------------------------------------------------------------------------
# Internal SNR helper
# ---------------------------------------------------------------------------

def _snr_from_scheduler(
    timesteps: torch.Tensor,
    noise_scheduler,
) -> torch.Tensor:
    """
    Compute per-sample SNR for *timesteps* using *noise_scheduler*.

    Supported scheduler types (duck-typed; no diffusers import required):

    * **Flow-matching** scheduler with ``.get_sigmas_for_timesteps()``:
      ``SNR = ((1 − σ) / σ)²``
    * **DDPM-style** scheduler with ``.alphas_cumprod`` table:
      ``SNR = ᾱ / (1 − ᾱ)``
    * **None** (fallback): assumes a linear FM schedule ``σ = t / 1000``.

    Returns a float32 tensor of shape ``[B]`` on the same device as *timesteps*.
    """
    if noise_scheduler is None:
        # Fallback: linear flow-matching schedule σ = t / num_train_timesteps
        sigma = (timesteps.float() / 1000.0).clamp(min=1e-8, max=1.0 - 1e-8)
        return ((1.0 - sigma) / sigma) ** 2

    if hasattr(noise_scheduler, 'get_sigmas_for_timesteps'):
        # TrainFlowMatchEulerDiscreteScheduler — exact lookup by value.
        dev = noise_scheduler.timesteps.device
        sigmas = noise_scheduler.get_sigmas_for_timesteps(
            timesteps.to(dev)
        ).float().to(timesteps.device)
        return ((1.0 - sigmas) / sigmas.clamp(min=1e-8)) ** 2

    if hasattr(noise_scheduler, 'sigmas') and hasattr(noise_scheduler, 'timesteps'):
        # Base FlowMatchEulerDiscreteScheduler (e.g. SDPipelineInference…).
        # sigmas[:-1] corresponds to timesteps[i] in descending order.
        # We look up by nearest-match in timestep-value space.
        ts_vals = noise_scheduler.timesteps.float().cpu()   # [N] descending values
        sigs    = noise_scheduler.sigmas[:-1].float().cpu() # [N], drop trailing 0
        ts_q    = timesteps.float().cpu()                   # [B] query values
        dist    = (ts_vals.unsqueeze(0) - ts_q.unsqueeze(1)).abs()  # [B, N]
        idx     = dist.argmin(dim=1)                        # [B]
        sigmas  = sigs[idx].to(timesteps.device)
        return ((1.0 - sigmas) / sigmas.clamp(min=1e-8)) ** 2

    if hasattr(noise_scheduler, 'alphas_cumprod'):
        # DDPM / v-prediction family
        ac = noise_scheduler.alphas_cumprod.to(timesteps.device)
        alpha_bar = ac[timesteps.long()].float()
        return alpha_bar / (1.0 - alpha_bar).clamp(min=1e-8)

    raise ValueError(
        f"Cannot compute SNR for scheduler type {type(noise_scheduler).__name__!r}: "
        "expected a flow-matching scheduler (has .get_sigmas_for_timesteps) or a "
        "DDPM-style scheduler (has .alphas_cumprod)."
    )


# ---------------------------------------------------------------------------
# Public function
# ---------------------------------------------------------------------------

def get_teacher_lambda(
    timesteps: torch.Tensor,
    args,
    noise_scheduler=None,
) -> torch.Tensor:
    """
    Return a per-sample teacher-distillation lambda tensor, shape ``[B]``.

    Applies a two-sided cosine (half-Hann) rolloff in **log-SNR space** to
    ``args.teacher_lambda``.  The active (full-weight) SNR zone is:

        ``[TEACHER_LAMBDA_SNR_FULL_LO, TEACHER_LAMBDA_SNR_FULL_HI]``

    i.e. ``[0.065, 2.25]`` by default, which corresponds to approximately
    ``t ∈ [400, 797]`` for an FM shift=1 student scheduler.

    Outside the zone the signal rolls off smoothly via a cosine window to zero
    at the outer edges ``[TEACHER_LAMBDA_SNR_ZERO_LO, TEACHER_LAMBDA_SNR_ZERO_HI]``.

    Because the cutoffs are specified in SNR (not raw timestep integers), the
    boundaries remain consistent across FM (shift=1), FM (shift=2), and DDPM
    student schedulers.

    Parameters
    ----------
    timesteps:
        Integer (or float) timestep tensor, shape ``[B]``.
    args:
        Namespace with at least ``teacher_lambda: float``.  Optionally
        ``teacher_lambda_falloff: bool`` activates the legacy linear ramp
        override for backward compatibility.
    noise_scheduler:
        The *student* noise scheduler, used to convert timesteps → SNR.
        When ``None``, falls back to a linear FM assumption (``σ = t/1000``).

    Returns
    -------
    torch.Tensor
        Per-sample lambda values, shape ``[B]``, on the same device as *timesteps*.
    """
    base = args.teacher_lambda
    t    = timesteps.float()

    if not args.teacher_lambda_falloff:
        return base * torch.ones_like(timesteps, dtype=torch.float32, device=timesteps.device)

    # ── SNR-based two-sided cosine rolloff in log-SNR space ───────────────────
    snr     = _snr_from_scheduler(timesteps, noise_scheduler)
    log_snr = torch.log(snr.clamp(min=1e-12))

    # Pre-compute log boundaries (cached as Python floats — no grad needed)
    log_zero_lo = math.log(TEACHER_LAMBDA_SNR_ZERO_LO)   # lowest SNR → 0
    log_full_lo = math.log(TEACHER_LAMBDA_SNR_FULL_LO)   # SNR above here → full
    log_full_hi = math.log(TEACHER_LAMBDA_SNR_FULL_HI)   # SNR below here → full
    log_zero_hi = math.log(TEACHER_LAMBDA_SNR_ZERO_HI)   # highest SNR → 0

    # Low-SNR rolloff: 0 → 1 as log_snr rises from log_zero_lo → log_full_lo
    lo_alpha = (log_snr - log_zero_lo) / max(1e-8, log_full_lo - log_zero_lo)
    lo_scale = 0.5 * (1.0 - torch.cos(lo_alpha.clamp(0.0, 1.0) * math.pi))

    # High-SNR rolloff: 1 → 0 as log_snr rises from log_full_hi → log_zero_hi
    hi_alpha = (log_zero_hi - log_snr) / max(1e-8, log_zero_hi - log_full_hi)
    hi_scale = 0.5 * (1.0 - torch.cos(hi_alpha.clamp(0.0, 1.0) * math.pi))

    scale = lo_scale * hi_scale
    return (scale * base).to(device=timesteps.device)


