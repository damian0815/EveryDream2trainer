"""
Shared offload/reload helpers for pause/resume training.

Both train.py (SD/SDXL via TrainingModel) and train_sana.py (SANA via
SanaTrainingModel) use these to move model components off the GPU during a
training pause and back on when resuming.
"""
import gc
import logging

import torch
import torch.nn as nn


def _unwrap(m: nn.Module) -> nn.Module:
    """Strip DDP wrapper if present."""
    return m.module if isinstance(m, torch.nn.parallel.DistributedDataParallel) else m


def _is_dispatched(module: nn.Module) -> bool:
    """Check if *module* is under accelerate dispatch (device_map or BitsAndBytes).

    When a module is dispatched, calling .to() raises:
        "You shouldn't move a model that is dispatched using accelerate hooks"
    """
    return hasattr(module, '_hf_hook') or getattr(module, 'hf_quantizer', None) is not None


def unload_model_for_pause(model) -> None:
    """
    Move every GPU-resident model component to CPU.

    Uses hasattr/getattr so the same function works for TrainingModel (SD/SDXL)
    and SanaTrainingModel (SANA).  Safe to call when a component is already on
    CPU or is None.
    .to() calls are guarded against accelerate-dispatched modules.
    """
    # ── Primary NN ────────────────────────────────────────────────────────
    if hasattr(model, 'unet') and model.unet is not None and not _is_dispatched(_unwrap(model.unet)):
        _unwrap(model.unet).to('cpu')
    if hasattr(model, 'transformer') and model.transformer is not None and not _is_dispatched(model.transformer):
        model.transformer.to('cpu')

    # ── Text encoders ─────────────────────────────────────────────────────
    if hasattr(model, 'load_textenc_to_device'):
        model.load_textenc_to_device('cpu')
    # text_encoder_2 is handled by load_textenc_to_device on TrainingModel

    # ── VAE ───────────────────────────────────────────────────────────────
    if hasattr(model, 'load_vae_to_device'):
        model.load_vae_to_device('cpu')

    # ── EMA (in-memory) — TrainingModel only ──────────────────────────────
    for ema_attr in ('unet_ema', 'text_encoder_ema', 'text_encoder_2_ema'):
        ema = getattr(model, ema_attr, None)
        if ema is not None:
            ema.to('cpu')

    # ── Self-flow projection head only (teacher IS the main EMA) ─────────
    sf_head = getattr(model, 'self_flow_proj_head', None)
    if sf_head is not None:
        sf_head.to('cpu')

    # ── CLIP model — TrainingModel only ───────────────────────────────────
    clip = getattr(model, 'clip_model', None)
    if clip is not None:
        clip.to('cpu')
    clip_proc = getattr(model, 'clip_processor', None)
    if clip_proc is not None and hasattr(clip_proc, 'to'):
        clip_proc.to('cpu')

    gc.collect()
    torch.cuda.empty_cache()
    logging.info("Model components offloaded to CPU")


def reload_model_for_resume(model, device: torch.device,
                            train_dtype: torch.dtype | None = None,
                            vae_dtype: torch.dtype | None = None,
                            args=None) -> None:
    """
    Move all model components back to *device*.

    *train_dtype* / *vae_dtype* — dtypes used when moving (SD/SDXL only;
    SANA components already have the correct dtype on CPU).
    *args* — used for gradient_checkpointing re-apply and DDP re-wrap flags.
    """
    # ── Primary NN ────────────────────────────────────────────────────────
    if hasattr(model, 'unet') and model.unet is not None and not _is_dispatched(_unwrap(model.unet)):
        _unwrap(model.unet).to(device, dtype=train_dtype)
    if hasattr(model, 'transformer') and model.transformer is not None and not _is_dispatched(model.transformer):
        model.transformer.to(device)

    # ── VAE (load before text encoders, matching train.py order) ──────────
    if hasattr(model, 'vae') and model.vae is not None and not _is_dispatched(model.vae):
        model.vae.to(device, dtype=vae_dtype)

    # ── Text encoders ─────────────────────────────────────────────────────
    if hasattr(model, 'text_encoder') and model.text_encoder is not None:
        if not _is_dispatched(model.text_encoder):
            is_sana = hasattr(model, 'transformer')  # heuristic: SANA uses transformer, not unet
            if is_sana:
                model.text_encoder.to(device)
            else:
                # SD/SDXL: follow the same dtype logic as train.py setup
                if args is not None and getattr(args, 'disable_textenc_training', False) and getattr(args, 'amp', False):
                    _unwrap(model.text_encoder).to(device, dtype=torch.float16)
                else:
                    _unwrap(model.text_encoder).to(device, dtype=train_dtype)
        if hasattr(model, 'text_encoder_2') and model.text_encoder_2 is not None and not _is_dispatched(_unwrap(model.text_encoder_2)):
            if args is not None and getattr(args, 'disable_textenc_training', False) and getattr(args, 'amp', False):
                _unwrap(model.text_encoder_2).to(device, dtype=torch.float16)
            else:
                _unwrap(model.text_encoder_2).to(device, dtype=train_dtype)

    # ── EMA (in-memory) — TrainingModel only ──────────────────────────────
    for ema_attr in ('unet_ema', 'text_encoder_ema', 'text_encoder_2_ema'):
        ema = getattr(model, ema_attr, None)
        if ema is not None:
            ema.to(device)

    # ── Self-flow projection head only (teacher IS the main EMA) ─────────
    sf_head = getattr(model, 'self_flow_proj_head', None)
    if sf_head is not None:
        sf_head.to(device)

    # ── CLIP model — TrainingModel only ───────────────────────────────────
    clip = getattr(model, 'clip_model', None)
    if clip is not None:
        clip.to(device)
    clip_proc = getattr(model, 'clip_processor', None)
    if clip_proc is not None and hasattr(clip_proc, 'to'):
        clip_proc.to(device)

    # ── Re-apply gradient checkpointing if it was enabled ─────────────────
    if args is not None and getattr(args, 'gradient_checkpointing', False):
        if hasattr(model, 'transformer') and model.transformer is not None:
            model.transformer.enable_gradient_checkpointing()
        if hasattr(model, 'unet') and model.unet is not None:
            _unwrap(model.unet).enable_gradient_checkpointing()
            if hasattr(_unwrap(model.text_encoder), 'gradient_checkpointing_enable'):
                _unwrap(model.text_encoder).gradient_checkpointing_enable()
            if hasattr(model, 'text_encoder_2') and model.text_encoder_2 is not None:
                if hasattr(_unwrap(model.text_encoder_2), 'gradient_checkpointing_enable'):
                    _unwrap(model.text_encoder_2).gradient_checkpointing_enable()

    gc.collect()
    torch.cuda.empty_cache()
    logging.info("Model components reloaded to %s", device)
