"""
Teacher distillation debugging utilities.

When --debug_teacher is passed, logs detailed per-tensor statistics and
RGB preview images for the first 10 training steps via TensorBoard.

Rendering strategy
------------------
* **True-latent** tensors (noisy x_t, predicted x₀, clean latents) are decoded
  with TAESD (from the ``taesd/`` submodule) to give proper pixel previews.
* **Velocity / noise** tensors (raw UNet output, FM velocity, noise ε, targets)
  cannot be passed through TAESD meaningfully; they are shown using the classic
  per-channel linear colour-projection matrix (@erucipe / @keturn / @torridgristle
  / @StAlKeR7779) with the *per-column* correct latent-space factors.
* In cross-VAE mode the teacher velocity ``t_vel`` is suppressed because it
  would mix tensors from two incompatible latent spaces.
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

from core.latent_interposer import get_shared_interposer
from core.prediction_bridge import (
    _recover_x0_from_epsilon,
    _recover_x0_from_vpred,
    _recover_x0_from_fm_velocity,
    _x0_to_fm_velocity,
)

# ---------------------------------------------------------------------------
# Latent → RGB preview helpers
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
    Uses the correct SD1 or SDXL colour-projection factors.
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
# TAESD-based latent decoder for debug previews
# ---------------------------------------------------------------------------


def _decode_scaled_latent_taesd(latent_1chw: torch.Tensor,
                        latent_type: str,
                        out_h: int,
                        out_w: int) -> torch.Tensor:
    pass


def _decode_latent_taesd_broken(
    latent_1chw: torch.Tensor,
    latent_type: str,
    out_h: int,
    out_w: int,
) -> Optional[torch.Tensor]:
    """
    Decode a single (1, C, H, W) latent to a (3, out_h, out_w) float32 [0,1]
    tensor using the TAESD tiny autoencoder for *latent_type*.

    Returns ``None`` if TAESD is unavailable or decoding fails.
    """
    try:
        from core.latent_interposer import TaesdLatentConverter, get_shared_interposer
        if latent_type not in TaesdLatentConverter.SUPPORTED_SPACES:
            return None
        converter = get_shared_interposer()
        # Run on the same device as the input; .to() is a no-op if already there.
        device = latent_1chw.device
        decoder = converter._get_decoder(latent_type).to(device)
        with torch.no_grad():
            px = decoder(latent_1chw.float()).clamp(0.0, 1.0)  # [1, 3, H*8, W*8]
            px = px.squeeze(0)                                   # [3, H*8, W*8]
            if px.shape[-2] != out_h or px.shape[-1] != out_w:
                px = F.interpolate(
                    px.unsqueeze(0),
                    size=(out_h, out_w),
                    mode="bilinear",
                    align_corners=False,
                ).squeeze(0)
        return px.cpu()  # grid assembly requires CPU tensors
    except Exception as exc:
        logging.debug(f"[teacher_debug] TAESD decode failed ({latent_type}): {exc}")
        return None


def _render_cell(
    sample_1chw: torch.Tensor,
    *,
    is_latent: bool,
    latent_type: Optional[str],
    cell_h: int,
    cell_w: int,
) -> torch.Tensor:
    """
    Render a single (1, C, H, W) tensor to a (3, cell_h, cell_w) float32 [0,1]
    display tile.
    """
    converter = get_shared_interposer()
    # Run on the same device as the input; .to() is a no-op if already there.
    device = sample_1chw.device
    decoder = converter._get_decoder(latent_type).to(device)

    px = decoder(sample_1chw.float()).clamp(0.0, 1.0)  # [1, 3, H*8, W*8]
    px = px.squeeze(0)  # [3, H*8, W*8]
    if px.shape[-2] != cell_h or px.shape[-1] != cell_w:
        px = F.interpolate(
            px.unsqueeze(0),
            size=(cell_h, cell_w),
            mode="bilinear",
            align_corners=False,
        ).squeeze(0)

    return px



# ---------------------------------------------------------------------------
# Generic grid builder
# ---------------------------------------------------------------------------

def _build_labelled_grid(
    columns: list,   # list of (label: str, tensor: Tensor|None, latent_type: str|None, is_true_latent: bool)
    batch_size: int,
    cell_h: int,
    cell_w: int,
    label_h: int = 28,
    pad: int = 2,
) -> "Image.Image":
    """
    Render *columns* into a labelled PIL Image grid.

    Each column is a 4-tuple ``(label, tensor, latent_type, is_true_latent)``:

    * *label*          – column header text (``\\n`` for line-breaks).
    * *tensor*         – ``[B, C, H, W]`` (or ``None`` → blank cell).
    * *latent_type*    – ``"v1"``, ``"xl"``, etc.  Used by :func:`_render_cell`
                         to choose between TAESD decoding and linear projection.
    * *is_true_latent* – ``True`` for noisy/clean latents (TAESD attempted),
                         ``False`` for velocity/noise tensors (linear projection).

    Returns a ``PIL.Image.Image`` with one row per batch element and one column
    per entry in *columns*.
    """
    blank = torch.zeros(3, cell_h, cell_w)
    num_cols = len(columns)

    cells: list[torch.Tensor] = []
    for b in range(batch_size):
        for _label, tensor, col_lt, is_latent in columns:
            if tensor is None:
                cells.append(blank)
            else:
                try:
                    sample = tensor[b].detach().float()   # keep on original device
                    if sample.dim() == 3:
                        sample = sample.unsqueeze(0)       # → (1, C, H, W)
                    cells.append(
                        _render_cell(sample, is_latent=is_latent,
                                     latent_type=col_lt, cell_h=cell_h, cell_w=cell_w)
                    )
                except Exception as exc:
                    logging.warning(f"[teacher_debug] cell render failed: {exc}")
                    cells.append(blank)

    grid = vutils.make_grid([c.cpu() for c in cells], nrow=num_cols, padding=pad,
                            normalize=False, pad_value=0.15)

    # ── label strip ────────────────────────────────────────────────────────
    grid_w = grid.shape[2]
    label_strip = Image.new("RGB", (grid_w, label_h), color=(30, 30, 30))
    draw = ImageDraw.Draw(label_strip)
    try:
        font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 9)
    except (IOError, OSError):
        # Try common macOS font paths before giving up
        _mac_candidates = [
            "/System/Library/Fonts/Supplemental/Arial.ttf",
            "/Library/Fonts/Arial.ttf",
            "/System/Library/Fonts/Geneva.ttf",
        ]
        font = None
        for _p in _mac_candidates:
            try:
                font = ImageFont.truetype(_p, 9)
                break
            except (IOError, OSError):
                pass
        if font is None:
            font = ImageFont.load_default()

    col_x_start = pad
    cell_stride  = cell_w + pad
    for col_idx, (label, _t, _lt, _il) in enumerate(columns):
        cx = col_x_start + col_idx * cell_stride + cell_w // 2
        parts = label.split("\n")
        y = 1
        for part in parts:
            try:
                bbox = draw.textbbox((0, 0), part, font=font)
                tw = bbox[2] - bbox[0]
            except AttributeError:
                tw = len(part) * 6
            draw.text((cx - tw // 2, y), part, fill=(220, 220, 220), font=font)
            y += 10

    label_tensor = torch.from_numpy(
        np.array(label_strip).astype("float32") / 255.0
    ).permute(2, 0, 1)

    final = torch.cat([label_tensor, grid], dim=1)   # (3, label_h + H_grid, W_grid)
    final_np = final.permute(1, 2, 0).clamp(0, 1).mul(255).byte().cpu().numpy()
    return Image.fromarray(final_np)


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




def _normalise_pred_type(t: str) -> str:
    if t in ("v-prediction", "v_prediction"):
        return "v_prediction"
    if t in ("flow-matching", "flow_prediction", "flow_match"):
        return "flow_prediction"
    if t == "epsilon":
        return "epsilon"
    raise ValueError(f"Unrecognised prediction type: {t!r}")


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
    teacher_debug_capture: Optional[dict[str, torch.Tensor]] = None,
    # Latent-space type codes ("v1", "xl", "v3", "fx") for TAESD decoding
    # and per-column colour-projection selection.
    teacher_target: Optional[torch.Tensor] = None,
    student_latent_type: Optional[str] = None,
    teacher_latent_type: Optional[str] = None,
) -> None:
    """
    Log teacher-distillation debug information for *global_step* (no-op after
    step 9).

    Produces a single TensorBoard image ``debug_teacher/latents`` at
    *global_step* where:
      - **columns** (X-axis) = one labelled tensor type per column
      - **rows**    (Y-axis) = one batch sample per row

    True-latent columns (noisy x_t, predicted x₀, clean x₁) are decoded via
    TAESD for a proper pixel preview when *student_latent_type* /
    *teacher_latent_type* are provided.  Velocity / noise tensors fall back to
    the classic per-channel linear colour-projection.

    In cross-VAE mode (``student_latent_type != teacher_latent_type``) the
    teacher velocity column (``t_vel``) is suppressed because it would mix
    tensors from two incompatible latent spaces.

    A text stats summary is logged via :mod:`logging`.
    Each step is logged at most once (deduplication against multiple slices).
    """
    if global_step >= 10:
        return

    dedup_key = (output_dir, global_step)
    if dedup_key in _logged_steps:
        return
    _logged_steps.add(dedup_key)

    # Infer latent types from is_sdxl when explicit types are not supplied.
    if student_latent_type is None:
        student_latent_type = "xl" if is_sdxl else "v1"

    cross_vae = (
        student_latent_type is not None
        and teacher_latent_type is not None
        and student_latent_type != teacher_latent_type
    )

    with torch.no_grad():
        # ── Derive student predicted-clean and predicted-velocity ─────────
        s_x0 = _recover_x0(model_pred, noisy_latents, student_timesteps,
                            student_scheduler, student_pred_type)
        s_vel = _recover_fm_velocity(model_pred, noisy_latents, student_timesteps,
                                     student_scheduler, student_pred_type, noise)

        teacher_noise = teacher_debug_capture.get("teacher_noise")
        teacher_noisy_latents = teacher_debug_capture.get("teacher_noisy_latents")
        teacher_output = teacher_debug_capture.get("teacher_output")
        teacher_timesteps = teacher_debug_capture.get("teacher_timesteps")

        # ── Derive teacher predicted-clean and predicted-velocity ──────────
        have_teacher = (
            teacher_output is not None
            and teacher_scheduler is not None
            and teacher_noisy_latents is not None
            and teacher_timesteps is not None
            and teacher_pred_type is not None
        )
        if have_teacher:
            t_x0 = _recover_x0(teacher_output, teacher_noisy_latents, teacher_timesteps,
                                teacher_scheduler, teacher_pred_type)
            # t_vel requires noise and x0 in the same latent space.
            # In cross-VAE mode the student noise is in student space while
            # t_x0 is in teacher space — the subtraction is meaningless.
            if cross_vae:
                t_vel = None   # suppressed; would mix incompatible spaces
            else:
                t_vel = _recover_fm_velocity(teacher_output, teacher_noisy_latents, teacher_timesteps,
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
            f"  student_latent_type: {student_latent_type or 'N/A (inferred from is_sdxl)'}",
            f"  teacher_latent_type: {teacher_latent_type or 'N/A (same as student)'}",
            f"  cross_vae          : {cross_vae}",
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
            f"  noise (ε)                : {_stats(teacher_noise) if have_teacher else 'N/A'}",
            f"  noisy_latents (x_t)      : {_stats(teacher_noisy_latents) if teacher_noisy_latents is not None else 'N/A'}",
            f"  model_pred (raw output)  : {_stats(teacher_output) if teacher_output is not None else 'N/A'}",
            f"  pred_clean_latents (x₀)  : {_stats(t_x0) if t_x0 is not None else 'N/A'}",
            f"  pred_velocity            : {_stats(t_vel) if t_vel is not None else ('N/A (cross-VAE)' if cross_vae else 'N/A')}",
            f"  teacher_target (→student): {_stats(teacher_target) if teacher_target is not None else 'N/A'}",
            "╚══════════════════════════════════════════════════════════════╝",
        ]
        summary = "\n".join(lines)
        logging.info(summary)

        # ── Build TensorBoard grid ─────────────────────────────────────────
        if log_writer is None:
            return

        # Column definitions:
        #   (short_label, tensor_or_None, latent_type_for_this_col, is_true_latent)
        #
        # is_true_latent=True  → TAESD decoder attempted (noisy, x0, clean)
        # is_true_latent=False → linear colour-projection (velocity, noise, target)
        #
        # teacher_target is in *student* space (already converted back by the
        # interposer), so it uses student_latent_type.
        # t_vel is None in cross-VAE mode (suppressed above).
        slt = student_latent_type
        tlt = teacher_latent_type or student_latent_type  # same space when no cross-VAE
        columns: list[tuple[str, Optional[torch.Tensor], Optional[str], bool]] = [
            ("student\nnoise",       noise,                              slt,  False),
            ("student\nnoisy-lat",   noisy_latents,                      slt,  True),
            ("student\npred-raw",    model_pred,                         slt,  False),
            ("student\npred-x0",     s_x0,                               slt,  True),
            ("student\npred-vel",    s_vel,                              slt,  False),
            ("student\ntarget",      target,                             slt,  False),
            ("actual\nclean-lat",    clean_latents,                      slt,  True),
            ("teacher\nnoise",       noise if have_teacher else None,    slt,  False),
            ("teacher\nnoisy-lat",   teacher_noisy_latents,                      tlt,  True),
            ("teacher\npred-raw",    teacher_output,                     tlt,  False),
            ("teacher\npred-x0",     t_x0,                               tlt,  True),
            ("teacher\npred-vel",    t_vel,                              tlt,  False),
            ("teacher\ntarget",      teacher_target,                     slt,  False),
        ]

        batch_size = noise.shape[0]
        # Infer cell dimensions from the first available tensor
        ref = noise  # always present
        _, _, lH, lW = ref.shape
        # Use 8× the latent size so TAESD-decoded cells are not downsampled heavily.
        # Cap at 128 px per side to keep the TensorBoard image manageable.
        cell_h = min(lH * 8, 128)
        cell_w = min(lW * 8, 128)

        pil_img = _build_labelled_grid(columns, batch_size, cell_h, cell_w)

        # TensorBoard wants a (3, H, W) float32 tensor in [0, 1].
        final_grid = torch.from_numpy(
            np.array(pil_img).astype("float32") / 255.0
        ).permute(2, 0, 1)


        log_writer.add_image(
            "debug_teacher/latents",
            final_grid,
            global_step=global_step,
        )


# ---------------------------------------------------------------------------
# Private: infer prediction-type without importing from core.loss
# (avoids a circular import — core.loss already imports log_teacher_debug)
# ---------------------------------------------------------------------------

def _get_pred_type(model) -> str:
    """Return ``'flow_prediction'``, ``'v_prediction'``, or ``'epsilon'``."""
    try:
        from core.flow_match_model import TrainFlowMatchEulerDiscreteScheduler
        if isinstance(model.noise_scheduler, TrainFlowMatchEulerDiscreteScheduler):
            return "flow_prediction"
    except ImportError:
        pass
    return model.noise_scheduler.config.get("prediction_type", "epsilon")


# ---------------------------------------------------------------------------
# Notebook-friendly helpers
# ---------------------------------------------------------------------------

def prepare_teacher_target_inputs(
    image_path: str,
    seed: int,
    teacher_model: "TrainingModel",
    caption: str,
    *,
    student_model: "Optional[TrainingModel]" = None,
    resolution: Optional[int] = None,
    timestep_index: int = 500,
) -> tuple:
    """
    Build the arguments required by :func:`core.loss.get_teacher_target` from
    raw materials, for use in Jupyter notebooks and interactive debugging.

    Parameters
    ----------
    image_path:
        Path to any image file (JPEG, PNG, …).  It is centre-cropped to a
        square and resized to *resolution*.
    seed:
        RNG seed for reproducible noise.
    teacher_model:
        Loaded teacher :class:`~model.training_model.TrainingModel`.  Its text
        encoder(s) are used to produce *teacher_conditioning*.
    caption:
        Text prompt to condition the teacher on.
    student_model:
        Loaded student :class:`~model.training_model.TrainingModel`.  Its VAE
        encodes the image into student latent space and its scheduler provides
        *student_timesteps*.  When ``None`` the teacher model is used as a
        stand-in (convenient for same-VAE debugging).
    resolution:
        Square pixel resolution.  Defaults to ``1024`` for SDXL students and
        ``512`` for SD1/2.
    timestep_index:
        Index into ``student_model.noise_scheduler.timesteps``
        (``0`` = highest noise, ``-1`` = nearly clean).  Default ``500``
        (midpoint).

    Returns
    -------
    ``(teacher_conditioning, clean_image_latents, noise, student_timesteps)``
        Exactly the four positional inputs needed by ``get_teacher_target()``.

    Typical Jupyter usage::

        from utils.teacher_debug import prepare_teacher_target_inputs, get_teacher_debug_grid_image
        from core.loss import get_teacher_target

        t_cond, clean_lat, noise, ts = prepare_teacher_target_inputs(
            "photo.jpg", seed=42,
            teacher_model=teacher_model, caption="a cat",
            student_model=student_model,
        )
        debug_capture = {}
        teacher_target = get_teacher_target(
            teacher_model=teacher_model,
            teacher_conditioning=t_cond,
            student_model=student_model,
            student_timesteps=ts,
            clean_image_latents=clean_lat,
            noise=noise,
            _debug_capture=debug_capture,
        )
        debug_capture.update(clean_image_latents=clean_lat, noise=noise,
                             teacher_target=teacher_target)
        img = get_teacher_debug_grid_image(debug_capture,
                                           teacher_model=teacher_model,
                                           student_model=student_model)
        display(img)   # inline in Jupyter
    """
    from PIL import Image as _PIL
    import torchvision.transforms.functional as _TF
    from model.training_model import get_text_conditioning, Conditioning

    vae_model = student_model if student_model is not None else teacher_model

    # ── 1. Resolve target resolution ─────────────────────────────────────────
    if resolution is None:
        resolution = 1024 if vae_model.is_sdxl else 512

    # ── 2. Load and centre-crop image to (resolution × resolution) ───────────
    img = _PIL.open(image_path).convert("RGB")
    w, h = img.size
    short = min(w, h)
    img = img.crop(((w - short) // 2, (h - short) // 2,
                    (w + short) // 2, (h + short) // 2))
    img = img.resize((resolution, resolution), _PIL.LANCZOS)

    # ── 3. Encode image with student (or teacher) VAE → scaled latents ────────
    device = vae_model.unet.device
    # Training dataloader uses [-1, 1] pixel values
    pixel_values = (_TF.to_tensor(img) * 2.0 - 1.0).unsqueeze(0)
    pixel_values = pixel_values.to(device=device, dtype=vae_model.vae.dtype)

    vae_scale = float(
        getattr(vae_model.vae.config, "scaling_factor",
                0.13025 if vae_model.is_sdxl else 0.18215)
    )
    with torch.no_grad():
        dist = vae_model.vae.encode(pixel_values, return_dict=False)[0]
        clean_image_latents = dist.sample() * vae_scale    # [1, C, H, W]

    # ── 4. Generate reproducible noise ───────────────────────────────────────
    rng = torch.Generator(device=device)
    rng.manual_seed(seed)
    noise = torch.randn(clean_image_latents.shape, generator=rng,
                        device=device, dtype=clean_image_latents.dtype)

    # ── 5. Pick a student timestep ────────────────────────────────────────────
    sched = vae_model.noise_scheduler
    idx = max(0, min(timestep_index, len(sched.timesteps) - 1))
    student_timesteps = sched.timesteps[idx: idx + 1].to(device)   # [1]

    # ── 6. Build teacher conditioning ─
    t_device = teacher_model.unet.device
    teacher_model.load_textenc_to_device(t_device)

    with torch.no_grad():
        enc_hs, enc_pe, enc_2_hs, enc_2_pe = get_text_conditioning(
            [caption], teacher_model, args=None
        )

    if teacher_model.is_sdxl:
        # Standard SDXL add_time_ids: (orig_h, orig_w, crop_y, crop_x, tgt_h, tgt_w)
        add_time_ids = torch.tensor(
            [[resolution, resolution, 0, 0, resolution, resolution]],
            dtype=torch.float32, device=t_device,
        )
        teacher_conditioning = Conditioning.sdxl_conditioning(
            text_encoder_hidden_states=enc_hs,
            text_encoder_pooled_embeds=enc_pe,
            text_encoder_2_hidden_states=enc_2_hs,
            text_encoder_2_pooled_embeds=enc_2_pe,
            add_time_ids=add_time_ids,
        )
    else:
        teacher_conditioning = Conditioning.sd12_conditioning(
            text_encoder_hidden_states=enc_hs,
            text_encoder_pooled_embeds=enc_pe,
        )

    return teacher_conditioning, clean_image_latents, noise, student_timesteps


def prepare_conditioning(
    caption: str,
    model: "TrainingModel",
    resolution: Optional[int] = None,
) -> "Conditioning":
    """
    Build a :class:`~model.training_model.Conditioning` object for *model*
    from a text *caption*.  Useful in notebooks when you need student
    conditioning to run the student UNet forward pass.

    Parameters
    ----------
    caption:
        Text prompt.
    model:
        Any loaded :class:`~model.training_model.TrainingModel`.
    resolution:
        Square pixel resolution used to build SDXL ``add_time_ids``.
        Defaults to ``1024`` for SDXL, ``512`` otherwise.

    Returns
    -------
    :class:`~model.training_model.Conditioning`

    Example::

        from utils.teacher_debug import prepare_conditioning
        s_cond = prepare_conditioning("a cat", student_model)
        img = get_teacher_debug_grid_image(
            debug_capture,
            teacher_model=teacher_model,
            student_model=student_model,
            student_conditioning=s_cond,
        )
        display(img)
    """
    from model.training_model import get_text_conditioning, Conditioning

    if resolution is None:
        resolution = 1024 if model.is_sdxl else 512

    device = model.unet.device
    model.load_textenc_to_device(device)

    with torch.no_grad():
        enc_hs, enc_pe, enc_2_hs, enc_2_pe = get_text_conditioning(
            [caption], model, args=None
        )

    if model.is_sdxl:
        add_time_ids = torch.tensor(
            [[resolution, resolution, 0, 0, resolution, resolution]],
            dtype=torch.float32, device=device,
        )
        return Conditioning.sdxl_conditioning(
            text_encoder_hidden_states=enc_hs,
            text_encoder_pooled_embeds=enc_pe,
            text_encoder_2_hidden_states=enc_2_hs,
            text_encoder_2_pooled_embeds=enc_2_pe,
            add_time_ids=add_time_ids,
        )
    else:
        return Conditioning.sd12_conditioning(
            text_encoder_hidden_states=enc_hs,
            text_encoder_pooled_embeds=enc_pe,
        )


def get_teacher_debug_grid_image(
    debug_capture: dict,
    *,
    teacher_model: "Optional[TrainingModel]" = None,
    student_model: "Optional[TrainingModel]" = None,
    student_conditioning: "Optional[Conditioning]" = None,
) -> "Image.Image":
    """
    Build a ``PIL.Image`` debug grid from a *debug_capture* dict.

    Build a debug grid showing all student and teacher intermediate tensors.

    Parameters
    ----------
    debug_capture:
        Dict populated by :func:`core.loss.get_teacher_target` (via
        ``_debug_capture``) plus manual additions.  All keys are optional —
        missing tensors render as blank grey cells.

        Keys read:

        ``"clean_image_latents"``  – clean student-space latents  ``[B,C,H,W]``
        ``"noise"``                – shared noise ε               ``[B,C,H,W]``
        ``"student_timesteps"``    – student timesteps            ``[B]``
        ``"teacher_noisy_latents"``        – teacher noisy input          ``[B,C,H,W]``
        ``"teacher_output"``       – raw teacher UNet output      ``[B,C,H,W]``
        ``"teacher_timesteps"``    – teacher timesteps            ``[B]``
        ``"teacher_target"``       – distillation target (student space) ``[B,C,H,W]``
        ``"model_pred"``           – raw student UNet output      ``[B,C,H,W]``
                                     (auto-computed when *student_conditioning*
                                     is supplied and this key is missing)

    teacher_model:
        Loaded teacher :class:`~model.training_model.TrainingModel`.
    student_model:
        Loaded student :class:`~model.training_model.TrainingModel`.
    student_conditioning:
        :class:`~model.training_model.Conditioning` for the student UNet.
        When provided *and* ``"model_pred"`` is absent from *debug_capture*,
        the student UNet is run automatically so that the ``student pred-raw``
        and ``student pred-x0`` columns are populated.
        Build it with :func:`prepare_conditioning`::

            s_cond = prepare_conditioning("a cat", student_model)
            img = get_teacher_debug_grid_image(
                debug_capture,
                teacher_model=teacher_model,
                student_model=student_model,
                student_conditioning=s_cond,
            )
            display(img)

    :func:`prepare_teacher_target_inputs` builds all ``get_teacher_target``
    inputs and pre-populates ``clean_image_latents`` and ``noise``::

        debug_capture = {}
        teacher_target = get_teacher_target(..., _debug_capture=debug_capture)
        debug_capture.update(
            clean_image_latents=clean_latents,
            noise=noise,
            teacher_target=teacher_target,
            student_timesteps=ts,
        )
        s_cond = prepare_conditioning("a cat", student_model)
        img = get_teacher_debug_grid_image(
            debug_capture,
            teacher_model=teacher_model,
            student_model=student_model,
            student_conditioning=s_cond,
        )
        display(img)   # inline in Jupyter
    """
    dc = debug_capture

    # ── Infer latent-space types ──────────────────────────────────────────────
    def _safe_infer(model):
        try:
            from core.latent_interposer import infer_latent_space_type
            return infer_latent_space_type(model)
        except Exception:
            return None

    slt = _safe_infer(student_model) or ("xl" if (student_model is not None and student_model.is_sdxl) else "v1")
    tlt = _safe_infer(teacher_model) or ("xl" if (teacher_model is not None and teacher_model.is_sdxl) else slt)

    # ── Derive teacher predicted clean-latent (t_x0) ─────────────────────────
    t_x0: Optional[torch.Tensor] = None
    t_out = dc.get("teacher_output")
    t_nsy = dc.get("teacher_noisy_latents")
    t_ts  = dc.get("teacher_timesteps")
    if t_out is not None and t_nsy is not None and t_ts is not None and teacher_model is not None:
        try:
            with torch.no_grad():
                t_x0 = _recover_x0(t_out, t_nsy, t_ts,
                                    teacher_model.noise_scheduler,
                                    _get_pred_type(teacher_model))
        except Exception as exc:
            logging.debug(f"[teacher_debug] t_x0 derivation failed: {exc}")

    # ── Compute student noisy latent (x_t) if not already in capture ─────────
    # Requires "clean_image_latents", "noise", and "student_timesteps" in dc
    # plus student_model for its scheduler.
    student_noisy: Optional[torch.Tensor] = dc.get("student_noisy")
    if (student_noisy is None
            and student_model is not None
            and all(k in dc for k in ("clean_image_latents", "noise", "student_timesteps"))):
        try:
            with torch.no_grad():
                sn = student_model.noise_scheduler.add_noise(
                    dc["clean_image_latents"].float(),
                    dc["noise"].float(),
                    dc["student_timesteps"],
                )
                student_noisy = sn.to(dc["clean_image_latents"].dtype)
        except Exception as exc:
            logging.debug(f"[teacher_debug] student_noisy computation failed: {exc}")

    # ── Auto-run student UNet when conditioning is provided ───────────────────
    # Populates dc["model_pred"] so the student pred-raw / pred-x0 columns show.
    model_pred: Optional[torch.Tensor] = dc.get("model_pred")
    if (model_pred is None
            and student_model is not None
            and student_conditioning is not None
            and student_noisy is not None
            and "student_timesteps" in dc):
        try:
            with torch.no_grad():
                unet = student_model.unet
                unet_dtype = unet.dtype
                noisy_in  = student_noisy.to(dtype=unet_dtype)
                ts_in     = dc["student_timesteps"].to(device=unet.device, dtype=unet_dtype)
                hs        = student_conditioning.prompt_embeds.to(device=unet.device, dtype=unet_dtype)
                added_kw  = (
                    student_conditioning.get_added_cond_kwargs(dtype=unet_dtype)
                    if student_model.is_sdxl else None
                )
                model_pred = unet(
                    noisy_in, ts_in,
                    encoder_hidden_states=hs,
                    added_cond_kwargs=added_kw,
                ).sample.float()
        except Exception as exc:
            logging.warning(f"[teacher_debug] student UNet forward failed: {exc}")

    # ── Derive student predicted-clean latent (s_x0) ────────────────────────
    s_x0: Optional[torch.Tensor] = None
    if (model_pred is not None
            and student_noisy is not None
            and "student_timesteps" in dc
            and student_model is not None):
        try:
            with torch.no_grad():
                s_x0 = _recover_x0(
                    model_pred, student_noisy, dc["student_timesteps"],
                    student_model.noise_scheduler,
                    _get_pred_type(student_model),
                )
        except Exception as exc:
            logging.debug(f"[teacher_debug] s_x0 derivation failed: {exc}")

    # ── Find a reference tensor for shape/batch-size ──────────────────────────
    _ref = next(
        (v for v in [dc.get("clean_image_latents"), dc.get("noise"),
                     t_nsy, t_out] if v is not None),
        None,
    )
    if _ref is None:
        raise ValueError(
            "debug_capture contains no tensors to visualise.  "
            "Add at least one of: clean_image_latents, noise, "
            "teacher_noisy_latents, teacher_output."
        )
    batch_size = _ref.shape[0]
    lH, lW = _ref.shape[-2], _ref.shape[-1]
    cell_h  = min(lH * 8, 128)
    cell_w  = min(lW * 8, 128)

    # ── Column definitions ────────────────────────────────────────────────────
    # (label, tensor, latent_type, is_true_latent)
    # is_true_latent=True  → TAESD decoder attempted (noisy / x0 / clean)
    # is_true_latent=False → linear colour-projection (velocity / noise / target)
    #
    # Student side first, then teacher side.
    columns: list = [
        # ── Student ────────────────────────────────────────────────────────
        ("student\nclean",      dc.get("clean_image_latents"), slt, True),
        ("student\nnoisy",      student_noisy,                  slt, True),
        ("student\nnoise/eps",  dc.get("noise"),                slt, False),
        ("student\npred-raw",   model_pred,                     slt, False),
        ("student\npred-x0",    s_x0,                           slt, True),
        ("student\ntarget",     dc.get("teacher_target"),       slt, False),
        # ── Teacher ────────────────────────────────────────────────────────
        ("teacher\nnoisy",      t_nsy,                          tlt, True),
        ("teacher\npred-raw",   t_out,                          tlt, False),
        ("teacher\npred-x0",    t_x0,                           tlt, True),
    ]

    return _build_labelled_grid(columns, batch_size, cell_h, cell_w)
