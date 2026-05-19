"""
Teacher distillation debugging utilities.

When --debug_teacher is passed, logs detailed per-tensor statistics and
low-resolution RGB preview images for the first 10 training steps.
Previews are written to:
    <logdir>/debug_teacher/step_NNNNN/<name>.jpg
A text summary is logged via :mod:`logging` and also written to
    <logdir>/debug_teacher/step_NNNNN/stats.txt

No VAE is used; latents are projected to RGB via a fixed linear colour
matrix (technique originally by @erucipe / @keturn / @torridgristle /
@StAlKeR7779).
"""
from __future__ import annotations

import io
import logging
import os
from typing import Optional

import torch
import torch.nn.functional as F
from PIL import Image

# ---------------------------------------------------------------------------
# Latent → RGB preview helpers
# (technique by @erucipe / @keturn / @torridgristle / @StAlKeR7779)
# ---------------------------------------------------------------------------

# SD 1.x per-channel RGB projection factors (updated for v1.5 by @torridgristle)
_SD1_LATENT_RGB_FACTORS = [
    #    R        G        B
    [0.3444, 0.1385, 0.0670],   # L1
    [0.1247, 0.4027, 0.1494],   # L2
    [-0.3192, 0.2513, 0.2103],  # L3
    [-0.1307, -0.1874, -0.7445],  # L4
]

# SDXL per-channel RGB projection factors (by @StAlKeR7779)
_SDXL_LATENT_RGB_FACTORS = [
    #    R        G        B
    [0.3816, 0.4930, 0.5320],
    [-0.3753, 0.1631, 0.1739],
    [0.1770, 0.3588, -0.2048],
    [-0.4350, -0.2644, -0.4289],
]

# Optional spatial-smoothing kernel applied after the SDXL projection
_SDXL_SMOOTH_MATRIX = [
    [0.0358, 0.0964, 0.0358],
    [0.0964, 0.4711, 0.0964],
    [0.0358, 0.0964, 0.0358],
]


def _sample_to_lowres_estimated_image(
    samples: torch.Tensor,
    latent_rgb_factors: torch.Tensor,
    smooth_matrix: Optional[torch.Tensor] = None,
) -> Image.Image:
    """
    Project a latent sample to a low-resolution RGB preview via a linear
    colour-factor matrix, with an optional smoothing convolution pass.

    Args:
        samples:            shape (1, C, H, W) — single latent from step callback
        latent_rgb_factors: (C, 3) projection matrix
        smooth_matrix:      optional (3, 3) spatial smoothing kernel

    Returns:
        PIL Image (RGB).
    """
    latent_image = samples[0].permute(1, 2, 0) @ latent_rgb_factors  # H×W×3

    if smooth_matrix is not None:
        latent_image = latent_image.unsqueeze(0).permute(3, 0, 1, 2)  # 3×1×H×W
        latent_image = F.conv2d(
            latent_image,
            smooth_matrix.reshape((1, 1, 3, 3)),
            padding=1,
        )
        latent_image = latent_image.permute(1, 2, 3, 0).squeeze(0)  # H×W×3

    latents_ubyte = (
        ((latent_image + 1) / 2)  # scale -1..1 → 0..1
        .clamp(0, 1)
        .mul(0xFF)
        .byte()
    ).cpu()

    return Image.fromarray(latents_ubyte.numpy())


def latents_to_preview_image(latents: torch.Tensor, *, is_sdxl: bool = False) -> bytes:
    """
    Convert a latent tensor to a low-quality preview JPEG without using the VAE.
    Uses a per-model RGB factor matrix for a coloured, recognisable approximation.

    Args:
        latents:  shape (1, C, H, W) — single-item latent from a step callback
        is_sdxl:  use SDXL projection factors + smoothing when True,
                  otherwise use SD 1.x factors.

    Returns:
        JPEG bytes.
    """
    with torch.no_grad():
        if is_sdxl:
            rgb_factors = torch.tensor(
                _SDXL_LATENT_RGB_FACTORS,
                dtype=latents.dtype,
                device=latents.device,
            )
            smooth = torch.tensor(
                _SDXL_SMOOTH_MATRIX,
                dtype=latents.dtype,
                device=latents.device,
            )
            img = _sample_to_lowres_estimated_image(latents, rgb_factors, smooth)
        else:
            rgb_factors = torch.tensor(
                _SD1_LATENT_RGB_FACTORS,
                dtype=latents.dtype,
                device=latents.device,
            )
            img = _sample_to_lowres_estimated_image(latents, rgb_factors)

    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=60)
    return buf.getvalue()


# ---------------------------------------------------------------------------
# Tensor-recovery helpers
# ---------------------------------------------------------------------------

def _get_fm_sigma(timesteps: torch.Tensor, scheduler) -> torch.Tensor:
    """
    Return flow-match sigma ∈ (0, 1) for each timestep in *timesteps*.
    Falls back to ``timestep / 1000`` if the scheduler doesn't expose
    ``get_sigmas_for_timesteps``.
    """
    try:
        return scheduler.get_sigmas_for_timesteps(
            timesteps.to(scheduler.timesteps.device)
        ).to(timesteps.device).float()
    except (AttributeError, RuntimeError):
        return (timesteps.float() / 1000.0).clamp(1e-6, 1.0 - 1e-6)


def _recover_x0(
    pred: torch.Tensor,
    noisy: torch.Tensor,
    timesteps: torch.Tensor,
    scheduler,
    pred_type: str,
) -> torch.Tensor:
    """
    Recover the predicted clean-image latent x₀ from a raw UNet prediction.

    Works for FM velocity, epsilon, and v-prediction heads.
    """
    pred_type = pred_type.lower().replace("-", "_")
    p = pred.float()
    x_t = noisy.float()

    if pred_type in ("flow_prediction", "flow_match", "flow_matching"):
        # x_t = (1-σ)·x₀ + σ·ε  →  x₁ = x_t − σ·v  (v = ε − x₁)
        sigma = _get_fm_sigma(timesteps, scheduler).view(-1, 1, 1, 1)
        return x_t - sigma * p

    if pred_type == "epsilon":
        ac = scheduler.alphas_cumprod.to(timesteps.device)
        ab = ac[timesteps].float().view(-1, 1, 1, 1)
        return (x_t - (1.0 - ab).sqrt() * p) / ab.sqrt().clamp(min=1e-8)

    if pred_type in ("v_prediction", "v_pred"):
        ac = scheduler.alphas_cumprod.to(timesteps.device)
        ab = ac[timesteps].float().view(-1, 1, 1, 1)
        return ab.sqrt() * x_t - (1.0 - ab).sqrt() * p

    # Unknown type — return raw prediction unchanged
    logging.debug(f"[teacher_debug] Unknown pred_type '{pred_type}', skipping x0 recovery.")
    return p


def _recover_fm_velocity(
    pred: torch.Tensor,
    noisy: torch.Tensor,
    timesteps: torch.Tensor,
    scheduler,
    pred_type: str,
    noise: torch.Tensor,
) -> torch.Tensor:
    """
    Compute the FM-convention velocity (ε − x₀) from any raw prediction.
    """
    x0 = _recover_x0(pred, noisy, timesteps, scheduler, pred_type)
    return noise.float() - x0


# ---------------------------------------------------------------------------
# Pretty-print helpers
# ---------------------------------------------------------------------------

def _stats(t: torch.Tensor) -> str:
    t = t.detach().float()
    return (
        f"min={t.min():.4f}  max={t.max():.4f}  "
        f"mean={t.mean():.4f}  std={t.std():.4f}  shape={list(t.shape)}"
    )


# ---------------------------------------------------------------------------
# Step deduplication — only log each (output_dir, global_step) once,
# even if multiple slices are processed within the same step.
# ---------------------------------------------------------------------------
_logged_steps: set[tuple[str, int]] = set()


# ---------------------------------------------------------------------------
# Main debug-logging entry point
# ---------------------------------------------------------------------------

def log_teacher_debug(
    *,
    global_step: int,
    output_dir: str,
    is_sdxl: bool,
    # Scheduler / prediction-type info
    student_pred_type: str,
    student_scheduler,
    teacher_pred_type: Optional[str],
    teacher_scheduler,
    # Student-side tensors
    noise: torch.Tensor,                  # [B, C, H, W]  shared noise ε
    noisy_latents: torch.Tensor,          # [B, C, H, W]  student x_t
    model_pred: torch.Tensor,             # [B, C, H, W]  raw student UNet output
    target: torch.Tensor,                 # [B, C, H, W]  student optimisation target
    clean_latents: torch.Tensor,          # [B, C, H, W]  clean image latents x₀
    student_timesteps: torch.Tensor,      # [B]
    # Teacher-side tensors (all optional — None when no teacher is active)
    teacher_noisy: Optional[torch.Tensor] = None,       # [B, C, H, W]
    teacher_output: Optional[torch.Tensor] = None,      # [B, C, H, W]  raw teacher UNet output
    teacher_timesteps: Optional[torch.Tensor] = None,   # [B]
    teacher_target: Optional[torch.Tensor] = None,      # [B, C, H, W]  teacher→student target
) -> None:
    """
    Log teacher-distillation debug information for *global_step* (no-op after
    step 9).  Writes a ``stats.txt`` and JPEG previews under::

        <output_dir>/debug_teacher/step_NNNNN/

    Each step is logged at most once (deduplication against multiple slices).
    """
    if global_step >= 10:
        return

    dedup_key = (output_dir, global_step)
    if dedup_key in _logged_steps:
        return
    _logged_steps.add(dedup_key)

    step_dir = os.path.join(output_dir, "debug_teacher", f"step_{global_step:05d}")
    os.makedirs(step_dir, exist_ok=True)

    with torch.no_grad():
        # ── Derive student predicted-clean and predicted-velocity ─────────
        s_x0 = _recover_x0(model_pred, noisy_latents, student_timesteps,
                            student_scheduler, student_pred_type)
        s_vel = _recover_fm_velocity(model_pred, noisy_latents, student_timesteps,
                                     student_scheduler, student_pred_type, noise)

        # ── Derive teacher predicted-clean and predicted-velocity ──────────
        if (teacher_output is not None
                and teacher_scheduler is not None
                and teacher_noisy is not None
                and teacher_timesteps is not None
                and teacher_pred_type is not None):
            t_x0 = _recover_x0(teacher_output, teacher_noisy, teacher_timesteps,
                                teacher_scheduler, teacher_pred_type)
            t_vel = _recover_fm_velocity(teacher_output, teacher_noisy, teacher_timesteps,
                                         teacher_scheduler, teacher_pred_type, noise)
        else:
            t_x0 = None
            t_vel = None

        # ── Text summary ───────────────────────────────────────────────────
        def _ts_list(ts):
            if ts is None:
                return "N/A"
            return str(ts.detach().cpu().tolist())

        lines = [
            f"╔══ Teacher Debug  step={global_step} ══╗",
            f"  student_pred_type  : {student_pred_type}",
            f"  teacher_pred_type  : {teacher_pred_type or 'N/A'}",
            f"  student_timesteps  : {_ts_list(student_timesteps)}",
            f"  teacher_timesteps  : {_ts_list(teacher_timesteps)}",
            "",
            "─── STUDENT ───────────────────────────────────────────────────",
            f"  noise (ε)                : {_stats(noise)}",
            f"  noisy_latents (x_t)      : {_stats(noisy_latents)}",
            f"  model_pred (raw output)  : {_stats(model_pred)}",
            f"  pred_clean_latents (x₀)  : {_stats(s_x0)}",
            f"  pred_velocity (ε−x₀)     : {_stats(s_vel)}",
            f"  target (actual)          : {_stats(target)}",
            f"  clean_latents (actual x₀): {_stats(clean_latents)}",
            "",
            "─── TEACHER ───────────────────────────────────────────────────",
            f"  teacher_noise (shared ε) : {_stats(noise) if teacher_output is not None else 'N/A'}",
            f"  teacher_noisy_latents    : {_stats(teacher_noisy) if teacher_noisy is not None else 'N/A'}",
            f"  teacher_output (raw)     : {_stats(teacher_output) if teacher_output is not None else 'N/A'}",
            f"  teacher_pred_clean (x₀)  : {_stats(t_x0) if t_x0 is not None else 'N/A'}",
            f"  teacher_pred_velocity    : {_stats(t_vel) if t_vel is not None else 'N/A'}",
            f"  teacher_target (→student): {_stats(teacher_target) if teacher_target is not None else 'N/A'}",
            "╚══════════════════════════════════════════════════════════════╝",
        ]
        summary = "\n".join(lines)
        logging.info(summary)
        with open(os.path.join(step_dir, "stats.txt"), "w", encoding="utf-8") as f:
            f.write(summary + "\n")

        # ── Preview images ─────────────────────────────────────────────────
        previews: dict[str, Optional[torch.Tensor]] = {
            "01_student_noise":          noise,
            "02_student_noisy_latents":  noisy_latents,
            "03_student_model_pred_raw": model_pred,
            "04_student_pred_clean_x0":  s_x0,
            "05_student_pred_velocity":  s_vel,
            "06_student_target_actual":  target,
            "07_actual_image_latents":   clean_latents,
            "08_teacher_noisy_latents":  teacher_noisy,
            "09_teacher_output_raw":     teacher_output,
            "10_teacher_pred_clean_x0":  t_x0,
            "11_teacher_pred_velocity":  t_vel,
            "12_teacher_target_student": teacher_target,
        }

        for name, tensor in previews.items():
            if tensor is None:
                continue
            path = os.path.join(step_dir, f"{name}.jpg")
            try:
                t = tensor.detach().float().cpu()
                if t.dim() == 3:
                    t = t.unsqueeze(0)
                elif t.dim() == 4 and t.shape[0] > 1:
                    t = t[:1]           # first item in batch only
                img_bytes = latents_to_preview_image(t, is_sdxl=is_sdxl)
                with open(path, "wb") as f:
                    f.write(img_bytes)
            except Exception as exc:
                logging.warning(f"[teacher_debug] Could not save preview '{name}': {exc}")

