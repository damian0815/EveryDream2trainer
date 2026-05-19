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

import numpy as np
import torch
import torch.nn.functional as F
import torchvision.utils as vutils
from PIL import Image, ImageDraw, ImageFont

from core.prediction_bridge import (
    _normalise_pred_type,
    _recover_x0_from_epsilon,
    _recover_x0_from_vpred,
    _recover_x0_from_fm_velocity,
    _x0_to_fm_velocity,
)

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


def _latent_to_chw_float(latent_1chw: torch.Tensor, *, is_sdxl: bool) -> torch.Tensor:
    """
    Project a single (1, C, H, W) latent to a (3, H, W) float32 tensor in [0, 1].
    Used for TensorBoard grid assembly.
    """
    with torch.no_grad():
        if is_sdxl:
            rgb_factors = torch.tensor(
                _SDXL_LATENT_RGB_FACTORS,
                dtype=latent_1chw.dtype,
                device=latent_1chw.device,
            )
            smooth = torch.tensor(
                _SDXL_SMOOTH_MATRIX,
                dtype=latent_1chw.dtype,
                device=latent_1chw.device,
            )
            img = _sample_to_lowres_estimated_image(latent_1chw, rgb_factors, smooth)
        else:
            rgb_factors = torch.tensor(
                _SD1_LATENT_RGB_FACTORS,
                dtype=latent_1chw.dtype,
                device=latent_1chw.device,
            )
            img = _sample_to_lowres_estimated_image(latent_1chw, rgb_factors)
    # img is uint8 (H, W, 3) numpy → (3, H, W) float [0,1]
    arr = np.array(img).astype("float32") / 255.0          # H×W×3
    return torch.from_numpy(arr).permute(2, 0, 1)           # 3×H×W


# ---------------------------------------------------------------------------
# Tensor-recovery helpers
# Delegates to the canonical implementations in core.prediction_bridge to
# avoid duplicating maths.
# ---------------------------------------------------------------------------

def _get_fm_sigma(timesteps: torch.Tensor, scheduler, *, device: torch.device) -> torch.Tensor:
    """
    Return flow-match sigma ∈ (0, 1) for each timestep in *timesteps*,
    always on *device*.
    Falls back to ``timestep / 1000`` if the scheduler doesn't expose
    ``get_sigmas_for_timesteps``.
    """
    try:
        return scheduler.get_sigmas_for_timesteps(
            timesteps.to(scheduler.timesteps.device)
        ).to(device=device, dtype=torch.float32)
    except (AttributeError, RuntimeError):
        return (timesteps.float().to(device) / 1000.0).clamp(1e-6, 1.0 - 1e-6)


def _recover_x0(
    pred: torch.Tensor,
    noisy: torch.Tensor,
    timesteps: torch.Tensor,
    scheduler,
    pred_type: str,
) -> torch.Tensor:
    """
    Recover the predicted clean-image latent x₀ from a raw UNet prediction.
    Delegates to the canonical helpers in :mod:`core.prediction_bridge`.
    All intermediate tensors are moved to the same device as *pred*.
    """
    try:
        pred_type = _normalise_pred_type(pred_type)
    except ValueError:
        logging.debug(f"[teacher_debug] Unknown pred_type '{pred_type}', returning raw prediction.")
        return pred.float()

    device = pred.device
    p = pred.float()
    x_t = noisy.float().to(device)
    ts = timesteps.to(device)

    if pred_type == "flow_prediction":
        sigma = _get_fm_sigma(ts, scheduler, device=device)
        return _recover_x0_from_fm_velocity(x_t, p, sigma)

    if pred_type == "epsilon":
        ab = scheduler.alphas_cumprod[ts.cpu()].float().to(device)
        return _recover_x0_from_epsilon(x_t, p, ab)

    if pred_type == "v_prediction":
        ab = scheduler.alphas_cumprod[ts.cpu()].float().to(device)
        return _recover_x0_from_vpred(x_t, p, ab)

    logging.debug(f"[teacher_debug] Unhandled normalised pred_type '{pred_type}', returning raw prediction.")
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
    Compute FM-convention velocity (ε − x₀) from any raw prediction.
    Delegates to :func:`core.prediction_bridge._x0_to_fm_velocity`.
    """
    x0 = _recover_x0(pred, noisy, timesteps, scheduler, pred_type)
    return _x0_to_fm_velocity(noise.float(), x0)


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
    log_writer=None,
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
    step 9).

    Produces a single TensorBoard image ``debug_teacher/latents`` at
    *global_step* where:
      - **columns** (X-axis) = one labelled tensor type per column
      - **rows**    (Y-axis) = one batch sample per row

    A text stats summary is logged via :mod:`logging`.  No disk image files
    are written.

    Each step is logged at most once (deduplication against multiple slices).
    """
    if global_step >= 10:
        return

    dedup_key = (output_dir, global_step)
    if dedup_key in _logged_steps:
        return
    _logged_steps.add(dedup_key)

    with torch.no_grad():
        # ── Derive student predicted-clean and predicted-velocity ─────────
        s_x0 = _recover_x0(model_pred, noisy_latents, student_timesteps,
                            student_scheduler, student_pred_type)
        s_vel = _recover_fm_velocity(model_pred, noisy_latents, student_timesteps,
                                     student_scheduler, student_pred_type, noise)

        # ── Derive teacher predicted-clean and predicted-velocity ──────────
        have_teacher = (
            teacher_output is not None
            and teacher_scheduler is not None
            and teacher_noisy is not None
            and teacher_timesteps is not None
            and teacher_pred_type is not None
        )
        if have_teacher:
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
            f"  teacher_noise (shared ε) : {_stats(noise) if have_teacher else 'N/A'}",
            f"  teacher_noisy_latents    : {_stats(teacher_noisy) if teacher_noisy is not None else 'N/A'}",
            f"  teacher_output (raw)     : {_stats(teacher_output) if teacher_output is not None else 'N/A'}",
            f"  teacher_pred_clean (x₀)  : {_stats(t_x0) if t_x0 is not None else 'N/A'}",
            f"  teacher_pred_velocity    : {_stats(t_vel) if t_vel is not None else 'N/A'}",
            f"  teacher_target (→student): {_stats(teacher_target) if teacher_target is not None else 'N/A'}",
            "╚══════════════════════════════════════════════════════════════╝",
        ]
        summary = "\n".join(lines)
        logging.info(summary)

        # ── Build TensorBoard grid ─────────────────────────────────────────
        if log_writer is None:
            return

        # Ordered columns: (short_label, tensor_or_None)
        columns: list[tuple[str, Optional[torch.Tensor]]] = [
            ("student\nnoise",       noise),
            ("student\nnoisy-lat",   noisy_latents),
            ("student\npred-raw",    model_pred),
            ("student\npred-x0",     s_x0),
            ("student\npred-vel",    s_vel),
            ("student\ntarget",      target),
            ("actual\nclean-lat",    clean_latents),
            ("teacher\nnoise",       noise if have_teacher else None),
            ("teacher\nnoisy-lat",   teacher_noisy),
            ("teacher\npred-raw",    teacher_output),
            ("teacher\npred-x0",     t_x0),
            ("teacher\npred-vel",    t_vel),
            ("teacher\ntarget",      teacher_target),
        ]

        batch_size = noise.shape[0]
        # Infer cell dimensions from the first available tensor
        ref = noise  # always present
        _, _, lH, lW = ref.shape
        cell_h, cell_w = lH, lW   # preview at native latent resolution

        LABEL_H = 28          # pixels reserved above each column for the text label
        PAD     = 2           # padding between cells (make_grid padding)
        num_cols = len(columns)

        # ── Render every cell: shape (3, cell_h, cell_w) float [0,1] ──────
        # Grid is row-major: [col0_row0, col1_row0, ..., col(N-1)_row0, col0_row1, ...]
        cells: list[torch.Tensor] = []
        blank = torch.zeros(3, cell_h, cell_w)  # placeholder for missing tensors

        for b in range(batch_size):
            for _label, tensor in columns:
                if tensor is None:
                    cells.append(blank)
                else:
                    try:
                        sample = tensor[b].detach().float().cpu()
                        if sample.dim() == 3:
                            sample = sample.unsqueeze(0)   # → (1, C, H, W)
                        cells.append(_latent_to_chw_float(sample, is_sdxl=is_sdxl))
                    except Exception as exc:
                        logging.warning(f"[teacher_debug] cell render failed: {exc}")
                        cells.append(blank)

        # make_grid with nrow=num_cols → each row in the grid = one batch item
        grid = vutils.make_grid(
            cells,
            nrow=num_cols,
            padding=PAD,
            normalize=False,
            pad_value=0.15,     # grey separator
        )
        # grid: (3, H_grid, W_grid) float [0,1]

        # ── Prepend a label row at the top ─────────────────────────────────
        grid_w = grid.shape[2]
        label_strip = Image.new("RGB", (grid_w, LABEL_H), color=(30, 30, 30))
        draw = ImageDraw.Draw(label_strip)

        # Try to get a small bitmap font; fall back to default if unavailable
        try:
            font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 9)
        except (IOError, OSError):
            font = ImageFont.load_default()

        # Column centres in the grid image
        # make_grid pads PAD on left before each image block
        col_x_start = PAD  # first cell starts after PAD border
        cell_stride = cell_w + PAD
        for col_idx, (label, _t) in enumerate(columns):
            cx = col_x_start + col_idx * cell_stride + cell_w // 2
            # Multi-line label: split on \n
            parts = label.split("\n")
            y = 1
            for part in parts:
                try:
                    bbox = draw.textbbox((0, 0), part, font=font)
                    tw = bbox[2] - bbox[0]
                except AttributeError:
                    tw = len(part) * 6  # rough fallback
                draw.text((cx - tw // 2, y), part, fill=(220, 220, 220), font=font)
                y += 10

        # Convert label strip to tensor (3, LABEL_H, grid_w)
        label_tensor = torch.from_numpy(
            np.array(label_strip).astype("float32") / 255.0
        ).permute(2, 0, 1)

        # Stack: label row on top, then the cell grid
        final_grid = torch.cat([label_tensor, grid], dim=1)  # (3, LABEL_H+H_grid, W_grid)

        log_writer.add_image(
            "debug_teacher/latents",
            final_grid,
            global_step=global_step,
        )



