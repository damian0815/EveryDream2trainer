"""
train_sana.py — SANA model training entry point for EveryDream2trainer.

Uses 🤗 diffusers for all model components (SanaPipeline, SanaTransformer2DModel,
AutoencoderDC, FlowMatchEulerDiscreteScheduler).  No SANA repo clone required.

Reuses:
  - run_accumulation_loop() from core/step.py  (nibble/accumulation/backward/optimizer step)
  - EveryDreamBatch / DataLoaderMultiAspect    (data loading, aspect-ratio bucketing)
  - EveryDreamOptimizer                        (AdamW, CAME, Prodigy, ...)
  - utils/train_args.py                        (shared CLI / JSON config parser)
  - utils/inference_context.py                 (eval/train guard during sample generation)
  - SampleGenerator                            (sample generation, TensorBoard logging)
"""
import argparse
import contextlib
import gc
import copy
import hashlib
import logging
import os
import random
import re
import time
from argparse import Namespace

import numpy as np
import safetensors
from colorama import Fore, Style
from diffusers import AutoencoderKLWan
from tqdm.auto import tqdm

import torch
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter
from accelerate.utils import set_seed

import data.aspects as aspects_module
from data.resolve_items import resolve_image_source_items
from core.semaphore_files import (check_semaphore_file_and_unlink, WANT_SAMPLES_SEMAPHORE_FILE,
                                  SAVE_FULL_SEMAPHORE_FILE, SAVE_FULL_AND_STOP_SEMAPHORE_FILE, STOP_SEMAPHORE_FILE,
                                  SAVE_FULL_WITH_OPTIMIZER_SEMAPHORE_FILE,
                                  SAVE_FULL_WITH_OPTIMIZER_AND_STOP_SEMAPHORE_FILE,
                                  PAUSE_TRAINING_SEMAPHORE_FILE, RESUME_TRAINING_SEMAPHORE_FILE,
                                  WANT_SAMPLES_OTHEREMA_SEMAPHORE_FILE)
from utils.distributed import barrier as dist_barrier
from utils.model_offload import unload_model_for_pause, reload_model_for_resume
from core.step import (run_accumulation_loop, repeat_with_oom_handling, _dump_memory_snapshot,
                       pause_memory_history, compute_train_process_01)
from core.step import get_best_match_resolution, choose_effective_batch_size
from core.loss import get_multirank_stratified_random_timesteps, build_self_flow_latents, apply_negative_loss_hinge
from core.loss_sana import compute_sana_loss, compute_sana_dpo_loss
from core.self_flow import get_self_flow_modules, compute_self_flow_loss
from core.teacher_lambda import get_teacher_lambda
from core.flow_match_model import TrainFlowMatchEulerDiscreteScheduler
from core.log import setup_local_logger, log_args, do_log_step, do_log_step_optimizer, append_epoch_log, LogData
from data.dataset import select_caption_variants
from model.sana_training_model import SanaTrainingModel, _is_dispatched, load_sana_model, save_sana_model
from model.sana_text_encoder import encode_prompts, encode_null_prompt
from model.training_model import TrainingVariables, find_last_checkpoint, get_use_ema_decay_training
from model.ema import update_ema, update_ema_disk
from optimizer.optimizers import EveryDreamOptimizer
from utils.sample_generator import SampleGenerator
from core.sample_generation import generate_samples
from utils.train_args import parse_train_args

import json
from plugins.plugins import PluginRunner, load_plugin

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _trim_to_actual_length(embeds: torch.Tensor, mask: torch.Tensor):
    """Trim embeds (B, N, C) and mask (B, N) to the longest actual
    sequence length in the batch, discarding trailing padding.

    Finds the index of the last non-pad token rather than just counting,
    to be safe against hypothetical mask gaps.
    """
    indices = torch.arange(mask.shape[1], device=mask.device).unsqueeze(0)
    max_actual = (indices * mask).max().item() + 1
    if max_actual < embeds.shape[1]:
        embeds = embeds[:, :max_actual, :]
        mask = mask[:, :max_actual]
    return embeds, mask


# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------

def _add_sana_args(parser) -> None:
    """Adds SANA-specific arguments on top of the shared EveryDream2 arg parser."""
    # Note: required=False here because the value may be supplied via --config JSON.
    # _setup_sana_args() validates that model_id is present.
    parser.add_argument("--model_id", type=str, default=None,
                        help="HuggingFace hub model ID, e.g. "
                             "Efficient-Large-Model/SANA1.5_1.6B_1024px_diffusers")
    parser.add_argument("--resume_from", type=str, default=None,
                        help="Path to a saved transformer .safetensors checkpoint to resume from")
    parser.add_argument("--save_every", type=int, default=1000,
                        help="Save transformer weights every N optimizer steps")
    parser.add_argument("--max_sequence_length", type=int, default=300,
                        help="Gemma token budget (default: 300)")
    parser.add_argument("--te_quantization", type=str, default='none', choices=['none', 'int4', 'int8'],
                        help="Quantization for the gemma text encoder")

    # Video training arguments
    parser.add_argument("--is_video", action="store_true",
                        help="Enable video training mode using SanaVideoPipeline")
    parser.add_argument("--video_frames", type=int, default=81,
                        help="Number of frames to extract per video")
    parser.add_argument("--video_fps", type=int, default=16,
                        help="Target FPS for the video")
    parser.add_argument("--default_motion_score", type=int, default=30,
                        help="Default motion score appended to captions")

    # DPO arguments
    parser.add_argument("--dpo_beta", type=float, default=0.1,
                        help="DPO temperature/scale factor")


def _setup_sana_args(args: Namespace) -> Namespace:
    """
    Fills in SANA-specific derived fields and overrides shared defaults that
    don't make sense for SANA (e.g. text encoder is always frozen).
    """
    # SANA text encoder is Gemma — always frozen
    args.disable_textenc_training = True

    # --model_id is required; may come from CLI or from --config JSON
    if not getattr(args, 'model_id', None):
        raise ValueError("--model_id is required (pass on CLI or via --config JSON)")

    # Map the shared --resume_ckpt to SANA's resume_from when the user passes
    # it via a JSON config (resume_ckpt is what EveryDreamOptimizer.load() uses)
    if args.resume_from is None:
        args.resume_from = args.resume_ckpt
    if args.resume_ckpt is None:
        args.resume_ckpt = args.resume_from

    if args.resume_from == 'findlast':
        args.resume_from = find_last_checkpoint(args.logdir, resolve_to_transformer=True)

    # Map sample_steps → used by SampleGenerator as default_sample_steps
    # (shared arg is sample_steps; train_sana historically called it sample_every)
    if not hasattr(args, 'sample_every'):
        args.sample_every = args.sample_steps

    # Ensure single resolution for video mode
    if args.is_video:
        if len(args.resolution) > 1:
            raise ValueError("Video training requires a single --resolution value")
        if args.resolution_multiplier and len(args.resolution_multiplier) > 1:
            raise ValueError("Video training requires a single resolution_multiplier value")

    # Derive gradient-accumulation multiplier expected by choose_effective_batch_size()
    batch_size = args.batch_size
    optimizer_batch_size = args.optimizer_batch_size
    if args.initial_batch_size is None:
        initial_bs = optimizer_batch_size if optimizer_batch_size is not None else batch_size
        args.initial_batch_size = initial_bs
    if args.final_batch_size is None:
        final_bs = optimizer_batch_size if optimizer_batch_size is not None else batch_size
        args.final_batch_size = final_bs

    # Resolution list validation (shared setup_args does this for SD/SDXL)
    if not isinstance(args.resolution, list):
        args.resolution = [args.resolution]
    if args.resolution_multiplier and len(args.resolution_multiplier) != len(args.resolution):
        raise ValueError(
            f"--resolution_multiplier: pass one multiplier per --resolution entry "
            f"(got {len(args.resolution_multiplier)} for {len(args.resolution)} resolutions)"
        )

    # Expand per-resolution slice-size lists
    args.forward_slice_size = _expand_to_per_resolution(
        args.forward_slice_size, args.resolution, "forward_slice_size"
    )
    args.max_backward_slice_size = _expand_to_per_resolution(
        args.max_backward_slice_size, args.resolution, "max_backward_slice_size"
    )

    if args.caption_cross_concatenation_p > 0:
        raise ValueError("--caption_cross_concatenation_p > 0 is not supported for SANA training")

    if getattr(args, 'dpo_p', 0) > 0:
        if not getattr(args, 'ema_device', None) or args.ema_device == 'cpu':
            logging.info("DPO enabled: force-enabling EMA as reference model")
            args.ema_device = "cuda"
        if not getattr(args, 'ema_decay_rate', None):
            args.ema_decay_rate = 0.9999

    return args


def _expand_to_per_resolution(values: list, resolutions: list, name: str) -> list:
    """Broadcast a 1-element list to one entry per resolution, or validate length."""
    if not values:
        return values
    if len(values) == 1:
        return values * len(resolutions)
    if len(values) != len(resolutions):
        raise ValueError(
            f"--{name}: pass either one value (applied to all resolutions) or "
            f"one per resolution. Got {len(values)} values for {len(resolutions)} resolutions."
        )
    return values


def _init_sana_ema(model: SanaTrainingModel, args: Namespace, device: torch.device, log_folder: str) -> None:
    """Initialise (or resume) main EMA shadow weights for the SANA transformer.

    Supports three ``ema_device`` modes:
      * ``"disk"``  — EMA kept as ``transformer_ema.safetensors`` in an
                       ``ema_working/`` subdirectory.  Sets ``model.ema_working_dir``.
      * ``"cpu"`` / ``"cuda"`` — in-memory ``copy.deepcopy(model.transformer)``.
      * ``None`` (disabled) — no-op (handled by caller).

    When resuming (``args.resume_from`` directory contains a
    ``transformer_ema.safetensors`` sidecar), that sidecar is loaded
    automatically so accumulated EMA momentum is preserved.
    """
    import shutil

    resume_dir = os.path.dirname(args.resume_from) if args.resume_from else ""

    # ── Disk-offload mode ──────────────────────────────────────────────────
    if args.ema_device == "disk":
        ema_wd = os.path.join(log_folder, "ema_working")
        os.makedirs(ema_wd, exist_ok=True)
        model.ema_working_dir = ema_wd

        working_file = os.path.join(ema_wd, "transformer_ema.safetensors")
        resume_file = os.path.join(resume_dir, "transformer_ema.safetensors")
        if not os.path.isfile(working_file):
            if os.path.isfile(resume_file):
                logging.info(f"EMA (disk): resuming transformer EMA from sidecar {resume_file}")
                shutil.copy2(resume_file, working_file)
            else:
                logging.info(f"EMA (disk): initialising transformer EMA from current weights → {working_file}")
                state = {k: v.detach().cpu().contiguous()
                         for k, v in model.transformer.state_dict().items()}
                safetensors.torch.save_file(state, working_file)
        else:
            logging.info(f"EMA (disk): found existing working file {working_file}")

        logging.info(
            f"EMA enabled (disk-offload): decay={args.ema_decay_rate}, "
            f"interval={args.ema_update_interval}, working_dir={ema_wd}"
        )
        return

    # ── In-memory mode (cpu / cuda) ───────────────────────────────────────
    if model.transformer_ema is None:
        ema = copy.deepcopy(model.transformer).to(args.ema_device, dtype=model.transformer.dtype)
        ema.requires_grad_(False)
        model.transformer_ema = ema
    else:
        model.transformer_ema = model.transformer_ema.to(args.ema_device, dtype=model.transformer.dtype)
        model.transformer_ema.requires_grad_(False)

    # Load sidecar if resuming
    if resume_dir:
        resume_file = os.path.join(resume_dir, "transformer_ema.safetensors")
        if os.path.isfile(resume_file):
            logging.info(f"EMA: loading transformer EMA sidecar from {resume_file}")
            sd = safetensors.torch.load_file(resume_file, device=args.ema_device)
            missing, unexpected = model.transformer_ema.load_state_dict(sd, strict=False)
            if missing:
                logging.warning(f"EMA resume: transformer missing keys (first 5): {missing[:5]}")
            if unexpected:
                logging.warning(f"EMA resume: transformer unexpected keys (first 5): {unexpected[:5]}")
        else:
            logging.info(
                f"EMA: no sidecar for transformer at {resume_file!r} — "
                f"initialising from current weights"
            )

    logging.info(
        f"EMA enabled (in-memory, device={args.ema_device}): "
        f"decay={args.ema_decay_rate}, interval={args.ema_update_interval}"
    )


def parse_sana_args() -> Namespace:
    """
    Parses CLI / JSON config args for SANA training.

    Uses the shared EveryDream2 arg parser so that all optimizer, data, logging,
    and timestep-sampling flags are identical between train.py and train_sana.py,
    making JSON config files interchangeable between the two entry points.

    SANA-specific flags (--model_id, --resume_from, --save_every,
    --max_sequence_length) are added on top via _add_sana_args().
    """
    args = parse_train_args(
        description="Train SANA model with EveryDream2 infrastructure",
        extra_args_fn=_add_sana_args,
        require_resume_ckpt=False,
    )
    return _setup_sana_args(args)



# ---------------------------------------------------------------------------
# Training-variables setup
# ---------------------------------------------------------------------------

def setup_sana_training_variables(args: Namespace) -> TrainingVariables:
    """
    Creates and initialises a TrainingVariables instance from the unified args
    namespace. Correctly builds per-resolution forward/backward slice-size maps.
    """
    tv = TrainingVariables()
    tv.setup_default_slice_sizes(args)
    return tv


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------


def _encoder_short_name(text_encoder: torch.nn.Module) -> str:
    """Short, filesystem-safe identifier for the text encoder model."""
    name = getattr(text_encoder, 'name_or_path', '')
    name = name or getattr(text_encoder.config, 'name_or_path', '') if hasattr(text_encoder, 'config') else name
    name = name or 'text_encoder'
    # sanitise for use in filenames
    return re.sub(r'[^a-zA-Z0-9_-]', '_', os.path.basename(name))


def build_sana_data_loader(args: Namespace, seed: int, plugin_runner: PluginRunner):
    """
    Resolves ImageSourceItems for every resolution in args.resolution, merges
    them, and returns a (EveryDreamBatch, torch DataLoader) pair.

    Images are returned in [-1, 1] range — the correct input range for
    diffusers AutoencoderDC.
    """
    from data.data_loader import DataLoaderMultiAspect
    from data.every_dream import EveryDreamBatch, build_torch_dataloader

    aspects_per_resolution = {
        r: aspects_module.get_aspect_buckets(r)
        for r in args.resolution
    }
    global_resolution_weights = {
        r: (args.resolution_multiplier[i] if args.resolution_multiplier else 1.0)
        for i, r in enumerate(args.resolution)
    }
    if any(w != 1.0 for w in global_resolution_weights.values()):
        logging.info(f"SANA data: resolution weights: {global_resolution_weights}")

    image_source_items = resolve_image_source_items(args, aspects_per_resolution, divisible_by=32)

    data_loader_multi_aspect = DataLoaderMultiAspect(
        image_train_items=image_source_items,
        seed=seed,
        batch_size=args.batch_size,
        caption_variants=args.caption_variants,
        expand_caption_variants=args.expand_caption_variants,
        global_resolution_weights=global_resolution_weights,
    )

    dataset = EveryDreamBatch(
        data_loader=data_loader_multi_aspect,
        tokenizer=None,
        seed=seed,
        plugin_runner=plugin_runner,
        default_motion_score=args.default_motion_score if args.is_video else 30,
        load_dpo_bad=getattr(args, 'load_dpo_bad', False),
        rotation_degrees=getattr(args, 'rotation_degrees', 0.0),
    )

    data_loader = build_torch_dataloader(
        dataset,
        batch_size=args.batch_size,
        num_workers=args.num_dataloader_workers if args.num_dataloader_workers is not None else 4,
    )

    if args.is_video:
        from data.video_train_item import set_debug_video_path
        set_debug_video_path(os.path.join(args.logdir, 'debug_video_frames'))

    return dataset, data_loader


# ---------------------------------------------------------------------------
# VAE encoding helper
# ---------------------------------------------------------------------------

def _encode_latents(
    model: SanaTrainingModel,
    images: torch.Tensor,
    device: torch.device,
    slice_size: int=None
) -> torch.Tensor:
    """
    Encodes a batch of images via the model's VAE.
    Returns scaled latents * vae.config.scaling_factor.
    AutoencoderDC output has ``.latent``; AutoencoderKL output has ``.sample``.
    Casts images to the VAE's dtype, not the transformer's.
    """
    if slice_size is not None:
        results = []
        for slice_start in range(0, images.shape[0], slice_size):
            results.append(_encode_latents(model, images[slice_start:slice_start + slice_size], device))
        return torch.cat(results, dim=0)

    """ original: 
    # Video data processing (original code)
                        z = vae_encode(
                            config.vae.vae_type,
                            vae,
                            batch[0].permute(0, 2, 1, 3, 4).to(vae_dtype),
                            device=accelerator.device,
                            cache_key=data_info["cache_key"],
                            if_cache=config.vae.if_cache,
                            data_info=data_info,
                        )  # B,F,C,H,W -> B,C,F,H,W
    """

    vae_dtype = next(model.vae.parameters()).dtype
    with torch.no_grad():
        encoded = model.vae.encode(images.to(device, dtype=vae_dtype))
        latents = encoded.latent if hasattr(encoded, "latent") else encoded.latent_dist.sample()
    if type(model.vae) is AutoencoderKLWan:
        scaling_factor = 1
    else:
        scaling_factor = model.vae.config.scaling_factor
    return latents.to(model.dtype) * scaling_factor


# ---------------------------------------------------------------------------
# Per-batch resolution bookkeeping
# ---------------------------------------------------------------------------

def _update_tv_for_batch(tv: TrainingVariables, full_batch: dict, args: Namespace) -> None:
    """
    Sets tv.batch_resolution, tv.forward_slice_size, and tv.max_backward_slice_size
    from the actual pixel count of the current batch.  Must be called once per batch
    before train_sana_step().

    For video tensors (B, C, F, H, W), the spatial dims are at index -2, -1.
    """
    if full_batch["image"].ndim == 5:
        image_pixel_count = full_batch["image"].shape[-2] * full_batch["image"].shape[-1]
    else:
        image_pixel_count = full_batch["image"].shape[2] * full_batch["image"].shape[3]
    tv.batch_resolution = get_best_match_resolution(args.resolution, image_pixel_count)
    tv.forward_slice_size = tv.default_forward_slice_size[tv.batch_resolution]
    tv.max_backward_slice_size = tv.default_max_backward_slice_size[tv.batch_resolution]


# ---------------------------------------------------------------------------
# Per-step training
# ---------------------------------------------------------------------------

def train_sana_step(
    full_batch: dict,
    model: SanaTrainingModel,
    tv: TrainingVariables,
    ed_optimizer: EveryDreamOptimizer,
    log_data: LogData,
    steps_pbar,
    device: torch.device,
    args: Namespace,
    train_progress_01,
    plugin_runner=None,
    log_writer: SummaryWriter = None,
) -> None:
    """
    Handles per-batch SANA training: text encoding, VAE encoding, timestep sampling,
    then delegates nibbling/accumulation/backward/step to run_accumulation_loop().
    """
    # 1. Text-encode the full batch once (encoder is frozen), with OOM retry
    caption_variants = select_caption_variants(
        full_batch["captions"],
        requested_variants=args.caption_variants
    )
    assert len(caption_variants) == 1
    caption_variant = caption_variants[0]
    del caption_variants

    if args.debug_save_memory_snapshots:
        _dump_memory_snapshot(
            os.path.join(getattr(args, '_snapshot_dir', '.'), f"gs{tv.global_step:06d}.pickle")
        )

    encoder_id = _encoder_short_name(model.text_encoder)

    # Pre-encode null conditioning once for this step
    model.load_textenc_to_device(device)
    null_y, null_y_mask = encode_null_prompt(
        model.tokenizer,
        model.text_encoder,
        device,
        max_sequence_length=model.max_sequence_length,
        complex_human_instruction=model.complex_human_instruction or None,
        dtype=model.dtype,
        cache_path=None,
    )
    if args.offload_text_encoder:
        model.load_textenc_to_device('cpu')

    def _save_embedding_to_cache(
        pathname: str, variant: str, embeds: torch.Tensor, mask: torch.Tensor,
        caption_text: str,
    ):
        """Write a single cache entry to disk (no-op if already exists)."""
        import safetensors.torch as _st
        sha = hashlib.sha256(caption_text.encode("utf-8")).hexdigest()[:8]
        stem, _ = os.path.splitext(pathname)
        cache_path = f"{stem}.{encoder_id}.{variant}.{sha}.embeddings.safetensors"
        if not os.path.exists(cache_path):
            _st.save_file(
                {"embeds": embeds.contiguous(), "mask": mask.contiguous()},
                cache_path,
            )

    def _pad_to_max_n(embeds: list, masks: list):
        """Pad all entries in-place to the same N (second axis)."""
        max_n = max(e.shape[1] for e in embeds)
        for i in range(len(embeds)):
            n = embeds[i].shape[1]
            if n < max_n:
                embeds[i] = torch.nn.functional.pad(embeds[i], (0, 0, 0, max_n - n), value=0.0)
                masks[i] = torch.nn.functional.pad(masks[i], (0, max_n - n), value=False)

    def nibble_loss_fn(nibble: dict) -> torch.Tensor:
        n = nibble["image"].shape[0]

        # — try reading from the JIT embedding cache first —
        precomputed_variant = nibble.get("precomputed_embeddings", {}).get(caption_variant, [])
        all_cached = bool(precomputed_variant) and all(p is not None for p in precomputed_variant)

        if all_cached:
            embeds_list = [p["embeds"].to(device) for p in precomputed_variant]
            mask_list = [p["mask"].to(device) for p in precomputed_variant]
            for i in range(len(embeds_list)):
                embeds_list[i], mask_list[i] = _trim_to_actual_length(embeds_list[i], mask_list[i])
            _pad_to_max_n(embeds_list, mask_list)
            y = torch.cat(embeds_list, dim=0).to(device, dtype=model.dtype)
            y_mask = torch.cat(mask_list, dim=0).to(device)
        else:
            ti_active = plugin_runner is not None and any(
                type(p).__name__ == 'TextualInversionPlugin' for p in plugin_runner.plugins
            )
            ti_grad_ctx = torch.enable_grad() if ti_active else contextlib.nullcontext()
            with torch.no_grad(), pause_memory_history(), ti_grad_ctx:
                model.load_textenc_to_device(device)

                if precomputed_variant and any(p is not None for p in precomputed_variant):
                    # — partially cached: encode only the missing ones —
                    need_encode: list[tuple[int, str]] = [
                        (i, c) for i, (c, p) in
                        enumerate(zip(nibble["captions"][caption_variant], precomputed_variant))
                        if p is None
                    ]
                    missing_captions = [c for _, c in need_encode]
                    new_embeds, new_mask = repeat_with_oom_handling(
                        initial_slice_size=tv.forward_slice_size,
                        callback=lambda sz: encode_prompts(
                            model.tokenizer, model.text_encoder, missing_captions, device,
                            max_sequence_length=model.max_sequence_length,
                            complex_human_instruction=model.complex_human_instruction or None,
                            dtype=model.dtype,
                            slice_size=sz,
                        ),
                        oom_log_info=f"OOM gs:{tv.global_step}/l:{tv.accumulated_loss_images_count} SANA text encoder encode",
                    )
                    # — write newly encoded entries to disk —
                    if args.cache_text_embeddings:
                        for j, (orig_idx, _) in enumerate(need_encode):
                            _save_embedding_to_cache(
                                pathname=nibble["pathnames"][orig_idx],
                                variant=caption_variant,
                                embeds=new_embeds[j:j+1],
                                mask=new_mask[j:j+1],
                                caption_text=nibble["captions"][caption_variant][orig_idx],
                            )

                    # — merge cached + newly computed —
                    embeds_list = [None] * len(nibble["pathnames"])
                    mask_list   = [None] * len(nibble["pathnames"])
                    for i, p in enumerate(precomputed_variant):
                        if p is not None:
                            embeds, mask = p["embeds"].to(device), p["mask"].to(device)
                            embeds, mask = _trim_to_actual_length(embeds, mask)
                            embeds_list[i] = embeds
                            mask_list[i]   = mask
                    for j, (orig_idx, _) in enumerate(need_encode):
                        embeds_list[orig_idx] = new_embeds[j:j+1].to(device)
                        mask_list[orig_idx]   = new_mask[j:j+1].to(device)

                    _pad_to_max_n(embeds_list, mask_list)
                    y = torch.cat(embeds_list, dim=0).to(device, dtype=model.dtype)
                    y_mask = torch.cat(mask_list, dim=0).to(device)
                else:
                    # — nothing cached: encode everything —
                    y, y_mask = repeat_with_oom_handling(
                        initial_slice_size=tv.forward_slice_size,
                        callback=lambda sz: encode_prompts(
                            model.tokenizer, model.text_encoder,
                            nibble["captions"][caption_variant], device,
                            max_sequence_length=model.max_sequence_length,
                            complex_human_instruction=model.complex_human_instruction or None,
                            dtype=model.dtype,
                            slice_size=sz,
                        ),
                        oom_log_info=f"OOM gs:{tv.global_step}/l:{tv.accumulated_loss_images_count} SANA text encoder encode",
                    )
                    # — write all newly encoded entries to disk —
                    if args.cache_text_embeddings:
                        for i in range(y.shape[0]):
                            _save_embedding_to_cache(
                                pathname=nibble["pathnames"][i],
                                variant=caption_variant,
                                embeds=y[i:i+1],
                                mask=y_mask[i:i+1],
                                caption_text=nibble["captions"][caption_variant][i],
                            )

                if args.offload_text_encoder:
                    model.load_textenc_to_device('cpu')

        # Build conditional dropout mask
        n = y.shape[0]
        cd_mask = torch.zeros(n, dtype=torch.bool, device=y.device)
        for i in range(n):
            p = nibble.get("cond_dropout", [None] * n)[i]
            if p is None:
                p = args.cond_dropout
            cd_mask[i] = random.random() < p

        if cd_mask.any():
            n_dropped = cd_mask.sum().item()
            _null_y = null_y
            _null_y_mask = null_y_mask
            if _null_y.shape[1] < y.shape[1]:
                _null_y = F.pad(_null_y, (0, 0, 0, y.shape[1] - _null_y.shape[1]), value=0.0)
                _null_y_mask = F.pad(_null_y_mask, (0, y.shape[1] - _null_y_mask.shape[1]), value=False)
            y[cd_mask] = _null_y.expand(n_dropped, -1, -1)
            y_mask[cd_mask] = _null_y_mask.expand(n_dropped, -1)

        if args.debug_save_memory_snapshots:
            _dump_memory_snapshot(
                os.path.join(getattr(args, '_snapshot_dir', '.'), f"gs{tv.global_step:06d}.pickle")
            )

        # 2. VAE-encode the full batch once, with OOM retry
        model.load_vae_to_device(device)
        z = repeat_with_oom_handling(
            initial_slice_size=tv.forward_slice_size,
            callback=lambda sz: _encode_latents(model, nibble["image"], device, slice_size=sz),
            oom_log_info=f"OOM gs:{tv.global_step}/l:{tv.accumulated_loss_images_count} SANA VAE encode",
        )
        if args.offload_vae:
            model.load_vae_to_device('cpu')

        if args.debug_save_memory_snapshots:
            _dump_memory_snapshot(
                os.path.join(getattr(args, '_snapshot_dir', '.'), f"gs{tv.global_step:06d}.pickle")
            )

        # 3. Sample stratified flow-matching timesteps for the full batch
        timesteps = _draw_stratified_timesteps(n, tv, model, args, device)

        # Generate noise once, shared by main loss and self-flow
        noise = torch.randn_like(z)

        loss_full, model_pred, target = repeat_with_oom_handling(
            initial_slice_size=tv.forward_slice_size,
            callback=lambda slice_size: compute_sana_loss(
                model.transformer,
                model.noise_scheduler,
                z,
                y,
                y_mask,
                timesteps,
                noise=noise,
                slice_size=slice_size
            ),
            oom_log_info=f"OOM gs:{tv.global_step}/l:{tv.accumulated_loss_images_count} SANA transformer forward",
        )
        mean_dims = list(range(1, len(target.shape)))
        loss_1d = loss_full.mean(dim=mean_dims)

        # Hinge + loss scale (matches SD/SDXL path in core/step.py)
        loss_scale = nibble.get("loss_scale", torch.ones(n, device=device))
        if loss_scale.ndim == 0:
            loss_scale = loss_scale.expand(n)
        loss_1d = apply_negative_loss_hinge(
            loss_1d, (loss_scale < 0).to(loss_1d.device), margin=args.negative_loss_margin
        )
        loss_1d = loss_1d * loss_scale.abs().to(loss_1d.device)

        # Teacher distillation (frozen snapshot, full-latent MSE)
        if (model.teacher_transformer is not None
                and args.teacher_lambda > 0
                and random.random() < args.teacher_p):

            # Optional timestep cap
            do_teacher = torch.ones(n, dtype=torch.bool, device=device)
            if args.teacher_timestep_max is not None:
                do_teacher &= (timesteps < args.teacher_timestep_max)

            if do_teacher.any():
                with torch.no_grad():
                    _, teacher_pred, _ = compute_sana_loss(
                        model.teacher_transformer, model.noise_scheduler,
                        z, y, y_mask, timesteps,
                        noise=noise,
                        slice_size=tv.forward_slice_size,
                    )

                l_teacher = F.mse_loss(
                    model_pred.float(), teacher_pred.float(), reduction='none'
                ).mean(dim=list(range(1, model_pred.ndim)))

                # Per-sample lambda with optional SNR falloff
                teacher_lambda_val = get_teacher_lambda(
                    timesteps, args, noise_scheduler=model.noise_scheduler
                )
                l_teacher[~do_teacher] = 0
                loss_1d = loss_1d + args.teacher_lambda * teacher_lambda_val * l_teacher

                if log_writer is not None:
                    log_writer.add_scalar("loss/teacher", l_teacher.mean().item(), global_step=tv.global_step)
                    log_writer.add_scalar("loss/teacher_lambda", teacher_lambda_val.mean().item(), global_step=tv.global_step)

        # loss_preview_image — concatenate model_pred, target, loss along spatial dim
        log_data.loss_preview_image = torch.cat(
            [model_pred, target, loss_full],
            dim=-2
        ).detach().clone().cpu()
        del model_pred, target, loss_full

        # Self-Flow representation loss
        has_ema_teacher = model.transformer_ema is not None
        do_self_flow = (
            has_ema_teacher
            and random.random() < args.self_flow_p
        )
        if do_self_flow:
            num_train_timesteps = model.noise_scheduler.config.num_train_timesteps
            s_timesteps = torch.randint(0, num_train_timesteps, (n,), device=device)
            s_timesteps = TrainFlowMatchEulerDiscreteScheduler.get_shifted_timesteps(
                s_timesteps, model.noise_scheduler.timesteps
            )
            patch_size = getattr(model.transformer.config, 'patch_size', 1)

            x_tau, x_tau_min, tau_min_ts, tau_1d, tau_mask_1d = build_self_flow_latents(
                latents=z,
                noise=noise,
                noise_scheduler=model.noise_scheduler,
                t=timesteps,
                s=s_timesteps,
                mask_ratio=args.self_flow_mask_ratio,
                patch_size=patch_size,
            )

            # Resolve hook modules: student block in live model, teacher block in EMA
            sf_student_mod, sf_teacher_mod = get_self_flow_modules(
                model.transformer, model.transformer_ema, args.self_flow_mode
            )

            # Student forward (grad-enabled) — hook on block to capture intermediate features
            student_storage = {}
            def _sf_student_hook(module, inp, output):
                out = output[0] if isinstance(output, tuple) else output
                student_storage['h'] = out
            student_handle = sf_student_mod.register_forward_hook(_sf_student_hook)
            try:
                model.transformer(
                    hidden_states=x_tau.to(dtype=y.dtype),
                    encoder_hidden_states=y,
                    timestep=tau_1d.to(dtype=y.dtype),
                    encoder_attention_mask=y_mask,
                )
            finally:
                student_handle.remove()

            # Teacher forward (no grad) — teacher IS the main EMA
            # Must call the full model (not the block) so time_embed processes the timestep
            teacher_storage = {}
            def _sf_teacher_hook(module, inp, output):
                out = output[0] if isinstance(output, tuple) else output
                teacher_storage['h'] = out

            teacher_handle = sf_teacher_mod.register_forward_hook(_sf_teacher_hook)
            try:
                with torch.no_grad():
                    model.transformer_ema(
                        hidden_states=x_tau_min.to(dtype=y.dtype),
                        encoder_hidden_states=y,
                        timestep=tau_min_ts.to(dtype=y.dtype),
                        encoder_attention_mask=y_mask,
                    )
            finally:
                teacher_handle.remove()

            l_rep_1d = compute_self_flow_loss(
                student_features=student_storage['h'],
                teacher_features=teacher_storage['h'],
                proj_head=model.self_flow_proj_head,
                debug_mask=tau_mask_1d
            )
            if log_writer is not None and (tv.global_step + 1) % args.log_step == 0:
                log_writer.add_scalar("loss/self_flow", l_rep_1d.mean().item(), global_step=tv.global_step)
            loss_1d = loss_1d + args.self_flow_gamma * l_rep_1d

        # Diffusion-DPO RLHF loss
        has_dpo = (nibble.get("dpo_bad") is not None and random.random() < getattr(args, 'dpo_p', 0))
        if has_dpo:
            dpo_bad_images = nibble["dpo_bad"]
            dpo_mask = dpo_bad_images.sum(dim=[1, 2, 3]) != 0

            if dpo_mask.any():
                loss_1d = loss_1d * (~dpo_mask).float()

                z_bad = repeat_with_oom_handling(
                    initial_slice_size=tv.forward_slice_size,
                    callback=lambda sz: _encode_latents(
                        model, dpo_bad_images[dpo_mask], device, slice_size=sz),
                    oom_log_info=f"OOM gs:{tv.global_step} DPO VAE encode bad",
                )

                dpo_loss_1d, dpo_info = compute_sana_dpo_loss(
                    policy_transformer=model.transformer,
                    reference_transformer=model.transformer_ema,
                    noise_scheduler=model.noise_scheduler,
                    z_good=z[dpo_mask],
                    z_bad=z_bad,
                    y=y[dpo_mask],
                    y_mask=y_mask[dpo_mask],
                    timesteps=timesteps[dpo_mask],
                    noise=noise[dpo_mask],
                    beta=getattr(args, 'dpo_beta', 0.1),
                    slice_size=tv.forward_slice_size,
                    model_pred_good=model_pred[dpo_mask],
                    target_good=target[dpo_mask],
                )

                loss_1d[dpo_mask] += dpo_loss_1d

                if log_writer is not None:
                    log_writer.add_scalar("loss/dpo", dpo_loss_1d.mean().item(),
                                          global_step=tv.global_step)
                    log_writer.add_scalar("loss/dpo_signal", dpo_info["dpo_signal"].mean().item(),
                                          global_step=tv.global_step)

        # Log CD vs non-CD loss
        log_data.loss_log_step_cd.append(
            loss_1d[cd_mask].mean().detach().item() if cd_mask.any() else 0
        )
        log_data.loss_log_step_non_cd.append(
            loss_1d[~cd_mask].mean().detach().item() if (~cd_mask).any() else 0
        )

        # Per-timestep coverage (feeds histograms in do_log_step)
        for t in timesteps:
            log_data.timestep_coverage[int(t.item())] += 1
            log_data.cumulative_timestep_coverage[int(t.item())] += 1

        del y, y_mask, z

        return loss_1d

    if args.debug_save_memory_snapshots:
        _dump_memory_snapshot(
            os.path.join(getattr(args, '_snapshot_dir', '.'), f"gs{tv.global_step:06d}.pickle")
        )

    def did_step_optimizer_cb():
        _do_post_optimizer_ema_update(model, tv, args=args, device=device)

    # 5. Generic accumulation loop (handles nibbling, OOM, backward, optimizer.step)
    run_accumulation_loop(
        full_batch=full_batch,
        tv=tv,
        ed_optimizer=ed_optimizer,
        model=model,
        nibble_loss_fn=nibble_loss_fn,
        plugin_runner=plugin_runner,
        log_data=log_data,
        steps_pbar=steps_pbar,
        did_step_optimizer_cb=did_step_optimizer_cb,
        args=args,
        train_progress_01=train_progress_01,
        log_writer=log_writer,
    )


def _draw_stratified_timesteps(
    batch_size: int,
    tv: TrainingVariables,
    model: SanaTrainingModel,
    args: Namespace,
    device: torch.device,
) -> torch.Tensor:
    """
    Returns a (batch_size,) float tensor of flow-matching timestep values.

    Uses multirank stratified sampling when args.timesteps_multirank_stratified is
    True (mirrors the path in core/step.py).  Falls back to uniform random integer
    indices otherwise.

    The integer indices are converted to float timestep values via
    TrainFlowMatchEulerDiscreteScheduler.get_shifted_timesteps(), which incorporates
    any configured frequency shift.
    """
    if args.timesteps_multirank_stratified:
        while (
            tv.remaining_stratified_timesteps is None
            or tv.remaining_stratified_timesteps.shape[0] < max(batch_size, tv.desired_effective_batch_size)
        ):
            chunk = get_multirank_stratified_random_timesteps(
                batch_size=tv.desired_effective_batch_size,
                device=device,
                distribution=args.timesteps_multirank_stratified_distribution,
                alpha=args.timesteps_multirank_stratified_alpha,
                beta=args.timesteps_multirank_stratified_beta,
                mode_scale=args.timesteps_multirank_stratified_mode_scale,
                stratify=args.timesteps_multirank_stratified_stratify,
            )
            tv.remaining_stratified_timesteps = (
                chunk if tv.remaining_stratified_timesteps is None
                else torch.cat([tv.remaining_stratified_timesteps, chunk])
            )
        timestep_indices = tv.remaining_stratified_timesteps[:batch_size]
        tv.remaining_stratified_timesteps = tv.remaining_stratified_timesteps[batch_size:]
    else:
        num_train_timesteps = model.noise_scheduler.config.num_train_timesteps
        timestep_indices = torch.randint(0, num_train_timesteps, (batch_size,))

    return TrainFlowMatchEulerDiscreteScheduler.get_shifted_timesteps(
        timestep_indices, model.noise_scheduler.timesteps
    ).to(device)



# ---------------------------------------------------------------------------
# Main training loop
# ---------------------------------------------------------------------------

def train_sana_loop(
    model: SanaTrainingModel,
    ed_optimizer: EveryDreamOptimizer,
    tv: TrainingVariables,
    dataset,
    data_loader,
    log_writer: SummaryWriter,
    device: torch.device,
    args: Namespace,
    sample_generator: SampleGenerator,
    logdir: str,
    log_time: str | None = None,
    plugin_runner=None,
) -> None:
    """
    Outer training loop: epochs → batches, with periodic save and sample generation.
    """

    global_step = 0

    try:
        from utils.gpu import GPU
        gpu_device = device if device.index is not None else torch.device(device.type, 0)
        gpu = GPU(gpu_device) if device.type == 'cuda' else None
        if device.type == 'cuda':
            torch.cuda.memory._set_allocator_settings("expandable_segments:True")
    except Exception:
        gpu = None

    # Initialise the effective batch size before the first step.
    # run_accumulation_loop() re-evaluates this after each optimizer step.
    tv.desired_effective_batch_size = choose_effective_batch_size(args, 0)

    logging.info(
        f"Effective optimizer batch size: {tv.desired_effective_batch_size} images "
        f"(data batch_size={args.batch_size})"
    )

    training_start_time = time.time()
    epoch_times = []

    epoch_pbar = tqdm(range(args.max_epochs), position=0, leave=True, dynamic_ncols=True)
    epoch_pbar.set_description(f"{Fore.LIGHTCYAN_EX}Epochs{Style.RESET_ALL}")

    append_epoch_log(global_step=tv.global_step, epoch_pbar=epoch_pbar, gpu=gpu, log_writer=log_writer)

    log_data = LogData()
    should_stop = False
    wants_pause = False

    if args.debug_save_memory_snapshots and torch.cuda.is_available():
        snapshot_dir = os.path.join(logdir, "memory_snapshots")
        os.makedirs(snapshot_dir, exist_ok=True)
        args._snapshot_dir = snapshot_dir
        torch.cuda.memory._record_memory_history(
            max_entries=100_000,
            #stacks='python',  # skip native symbolization — usually the biggest win
        )
        torch.cuda.memory._dump_snapshot(os.path.join(snapshot_dir, "after_model_load.pickle"))
        logging.info(f" Memory snapshotting enabled → {snapshot_dir}")

    plugin_runner.run_on_training_start(
        log_folder=logdir,
        project_name=args.project_name,
        max_epochs=args.max_epochs,
    )

    epoch = 0
    for epoch in range(args.max_epochs):
        epoch_start_time = time.time()
        # Reset stratified timestep buffer at each epoch boundary (matches train.py)
        tv.remaining_stratified_timesteps = None

        plugin_runner.run_on_epoch_start(
            epoch=epoch,
            global_step=tv.global_step,
            epoch_length=len(data_loader),
            project_name=args.project_name,
            log_folder=logdir,
            data_root=args.data_root,
        )

        dataset.shuffle(epoch_n=epoch, max_epochs=args.max_epochs)
        sample_generator.on_epoch_start(epoch, global_step, epoch_length = len(data_loader))
        steps_pbar = tqdm(data_loader, desc=f"Step")

        epoch_len = len(data_loader)
        local_step = 0

        for full_batch in steps_pbar:
            tv.global_step = global_step

            _update_tv_for_batch(tv, full_batch, args)

            plugin_runner.run_on_step_start(
                epoch=epoch,
                global_step=global_step,
                local_step=local_step,
                num_samples=tv.total_trained_samples_count,
                project_name=args.project_name,
                log_writer=log_writer,
                log_folder=logdir,
                data_root=args.data_root,
                batch=full_batch,
            )

            step_start_time = time.time()
            train_progress_01 = compute_train_process_01(
                epoch=epoch,
                step=local_step,
                steps_per_epoch=epoch_len,
                max_epochs=args.max_epochs,
                max_global_steps=args.max_steps
            )

            train_sana_step(
                full_batch=full_batch,
                model=model,
                tv=tv,
                ed_optimizer=ed_optimizer,
                log_data=log_data,
                steps_pbar=steps_pbar,
                device=device,
                args=args,
                plugin_runner=plugin_runner,
                log_writer=log_writer,
                train_progress_01=train_progress_01
            )

            ed_optimizer.notify_step()

            log_step_opt = args.log_step_optimizer if args.log_step_optimizer is not None else args.log_step
            if (tv.global_step + 1) % log_step_opt == 0:
                ed_optimizer.flush_optimizer_logs(tv.global_step)
                do_log_step_optimizer(args, ed_optimizer, log_writer, tv)

            images_per_sec = full_batch["image"].shape[0] / (time.time() - step_start_time)
            log_data.images_per_sec_log_step.append(images_per_sec)

            if (tv.global_step + 1) % args.log_step == 0:
                logs = do_log_step(args, ed_optimizer, log_data, logdir, log_writer, model, tv)
                append_epoch_log(global_step=tv.global_step, epoch_pbar=epoch_pbar, gpu=gpu, log_writer=log_writer, **logs)
                torch.cuda.empty_cache()

            if global_step > 0 and global_step % args.save_every == 0:
                save_path = os.path.join(logdir, f"gs{global_step}")
                save_sana_model(save_path, model=model, optimizer=ed_optimizer, global_step=global_step,
                                num_samples=tv.total_trained_samples_count, lora=args.lora)

            user_wants_samples = check_semaphore_file_and_unlink(WANT_SAMPLES_SEMAPHORE_FILE)
            user_wants_otherema_samples = check_semaphore_file_and_unlink(WANT_SAMPLES_OTHEREMA_SEMAPHORE_FILE)
            if user_wants_samples or user_wants_otherema_samples or sample_generator.should_generate_samples(tv.global_step, local_step=local_step):
                generate_samples(model=model, sample_generator=sample_generator,
                                 global_step=global_step, batch=full_batch,
                                 args=args, device=device, log_folder=logdir,
                                 log_time=log_time, train_dtype=model.dtype,
                                 vae_dtype=None, swap_ema_requested=user_wants_otherema_samples)

            def plugin_runner_save_fn(path: str, step: int, num_samples: int) -> None:
                save_sana_model(path, model=model, optimizer=ed_optimizer, global_step=step,
                                num_samples=num_samples, lora=args.lora)

            plugin_runner.run_on_step_end(
                epoch=epoch,
                global_step=global_step,
                local_step=local_step,
                num_samples=tv.total_trained_samples_count,
                project_name=args.project_name,
                log_writer=log_writer,
                log_folder=logdir,
                data_root=args.data_root,
                batch=full_batch,
                save_fn=plugin_runner_save_fn,
            )

            global_step += 1
            local_step += 1

            should_save = False
            should_save_optimizer = args.save_optimizer
            if args.max_steps is not None and global_step >= args.max_steps:
                logging.info(f"Reached max_steps={args.max_steps}, stopping.")
                should_stop = True
                should_save = True

            if check_semaphore_file_and_unlink(STOP_SEMAPHORE_FILE):
                should_stop = True
            if check_semaphore_file_and_unlink(SAVE_FULL_SEMAPHORE_FILE):
                should_save = True
            if check_semaphore_file_and_unlink(SAVE_FULL_AND_STOP_SEMAPHORE_FILE):
                should_save = True
                should_stop = True
            if check_semaphore_file_and_unlink(SAVE_FULL_WITH_OPTIMIZER_AND_STOP_SEMAPHORE_FILE):
                should_save = True
                should_stop = True
                should_save_optimizer = True
            if check_semaphore_file_and_unlink(SAVE_FULL_WITH_OPTIMIZER_SEMAPHORE_FILE):
                should_save = True
                should_save_optimizer = True
            if check_semaphore_file_and_unlink(PAUSE_TRAINING_SEMAPHORE_FILE):
                logging.info("pause_training.semaphore detected — pausing after this step")
                wants_pause = True

            if should_save:
                logging.info("Save requested -> saving")
                ckpt_path = _make_ckpt_path(logdir, args, epoch, tv)
                save_sana_model(ckpt_path, model=model, optimizer=ed_optimizer if should_save_optimizer else None,
                                global_step=global_step, num_samples=tv.total_trained_samples_count, lora=args.lora)
            if should_stop:
                logging.info("Stop requested -> stopping")
                break

            if wants_pause:
                dist_barrier()

                save_path = _make_ckpt_path(logdir, args, epoch, tv)
                save_sana_model(save_path, model=model, optimizer=ed_optimizer,
                                global_step=global_step, num_samples=tv.total_trained_samples_count, lora=args.lora)
                logging.info(f"Checkpoint saved to {save_path}")

                unload_model_for_pause(model)
                logging.info("Training paused — create resume_training.semaphore to continue")

                while not check_semaphore_file_and_unlink(RESUME_TRAINING_SEMAPHORE_FILE):
                    time.sleep(1)

                dist_barrier()

                reload_model_for_resume(model, device, model.dtype, None, args)
                logging.info("Training resumed")

                wants_pause = False


        epoch_pbar.update(1)
        if len(log_data.loss_epoch) > 0:
            loss_epoch = sum(log_data.loss_epoch) / len(log_data.loss_epoch)
            log_writer.add_scalar(tag="loss/epoch", scalar_value=loss_epoch, global_step=tv.global_step)

        plugin_runner.run_on_epoch_end(
            epoch=epoch,
            global_step=tv.global_step,
            project_name=args.project_name,
            log_folder=logdir,
            data_root=args.data_root,
        )

        gc.collect()

        elapsed_epoch_time = (time.time() - epoch_start_time) / 60
        epoch_times.append(dict(epoch=epoch, time=elapsed_epoch_time))
        log_writer.add_scalar(
            "performance/minutes per epoch", elapsed_epoch_time, tv.global_step
        )

        if should_stop:
            break

    plugin_runner.run_on_training_end(
        log_folder=logdir,
        project_name=args.project_name,
        global_step=tv.global_step,
    )

    logging.info("SANA training complete.")

    final_ckpt_path = _make_ckpt_path(logdir, args, epoch, tv)
    save_sana_model(final_ckpt_path, model=model, optimizer=ed_optimizer, global_step=global_step,
                    num_samples=tv.total_trained_samples_count, lora=args.lora)

    logging.info(f" * generating final samples...")
    _, batch = next(enumerate(data_loader))
    generate_samples(model=model, sample_generator=sample_generator,
                     global_step=tv.global_step, batch=batch,
                     args=args, device=device, log_folder=logdir,
                     log_time=log_time, train_dtype=model.dtype,
                     vae_dtype=None)

    total_elapsed_time = time.time() - training_start_time
    logging.info(f"{Fore.CYAN}Training complete{Style.RESET_ALL}")
    logging.info(f"Total training time took {total_elapsed_time / 60:.2f} minutes, total steps: {tv.global_step}")
    logging.info(f"Average epoch time: {np.mean([t['time'] for t in epoch_times]):.2f} minutes")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main_sana() -> None:
    """
    Main SANA training entry point. Wires together argument parsing, model
    loading, data pipeline, optimiser, training variables, logging, and the
    training loop.
    """
    import os
    os.environ["TOKENIZERS_PARALLELISM"] = "false"

    import pprint
    args = parse_sana_args()
    from utils.train_args import validate_self_flow_ema_args
    validate_self_flow_ema_args(args)
    log_time, log_folder = setup_local_logger(args)

    if args.debug_log_on_nan:
        torch.autograd.set_detect_anomaly(True)

    set_seed(args.seed)

    print(" Args:")
    pprint.pprint(vars(args))

    device = torch.device(
        "cuda" if torch.cuda.is_available()
        else "mps" if torch.backends.mps.is_available()
        else "cpu"
    )

    logging.info(f"Loading SANA model from {args.model_id}...")
    if args.debug_no_load_model:
        logging.warning("--debug_no_load_model passed - not loading model!")
        model = SanaTrainingModel(
            noise_scheduler=None,
            text_encoder=None,
            tokenizer=None,
            transformer=None,
            vae=None,
            model_id=args.model_id,
        )
    else:
        model = load_sana_model(args)

        if get_use_ema_decay_training(args):
            _init_sana_ema(model, args, device, log_folder)
        else:
            model.transformer_ema = None
            model.ema_working_dir = None

        if not _is_dispatched(model.transformer):
            model.transformer.to(device)
        if model.self_flow_proj_head is not None:
            model.self_flow_proj_head.to(device)
        if model.teacher_transformer is not None:
            model.teacher_transformer.to(device)
        if model.transformer_ema is not None:
            model.transformer_ema.to(device)
        if not _is_dispatched(model.text_encoder):
            if not args.offload_text_encoder:
                model.text_encoder.to(device)
        if not args.offload_vae and not _is_dispatched(model.vae):
            model.vae.to(device)

        if args.gradient_checkpointing:
            model.transformer.enable_gradient_checkpointing()

    if not os.path.exists(args.optimizer_config):
        raise FileNotFoundError(
            f"Optimizer config not found: {args.optimizer_config}. "
            "Pass --optimizer_config pointing to a valid JSON file (e.g. optimizer.json)."
        )
    with open(args.optimizer_config) as f:
        optimizer_config = json.load(f)


    if args.plugins is not None:
        plugins = [load_plugin(name) for name in args.plugins]
    else:
        logging.info("No plugins specified")
        plugins = []
    plugin_runner = PluginRunner(plugins=plugins)
    plugin_runner.run_on_model_load(unet=model.unet, text_encoder=model.text_encoder, tokenizer=model.tokenizer, optimizer_config=optimizer_config)

    if args.cache_text_embeddings:
        if any(type(p).__name__ == 'TextualInversionPlugin' for p in plugin_runner.plugins):
            logging.warning(" * Disabling text embedding cache: textual_inversion plugin active")
            args.cache_text_embeddings = False

    logging.info(f"Building data loader for resolutions: {args.resolution}")
    dataset, data_loader = build_sana_data_loader(args, seed=args.seed, plugin_runner=plugin_runner)
    dataset.text_encoder_name = _encoder_short_name(model.text_encoder)
    dataset.cache_text_embeddings = args.cache_text_embeddings
    dataset.clean_stale_embeddings = args.clean_stale_embeddings
    epoch_len = len(data_loader)

    # ── EMA strength target → decay rate derivation ───────────────────────
    if get_use_ema_decay_training(args):
        if args.ema_strength_target is not None:
            total_number_of_steps: float = epoch_len * args.max_epochs
            args.ema_decay_rate = args.ema_strength_target ** (1.0 / total_number_of_steps)
            logging.info(
                f"ema_strength_target={args.ema_strength_target} → "
                f"ema_decay_rate={args.ema_decay_rate:.8f} over {total_number_of_steps:.0f} steps)"
            )

    log_writer = SummaryWriter(log_dir=log_folder, flush_secs=20)

    # Dump args + optimizer config next to the TensorBoard event file
    log_args(log_writer, args, optimizer_config, log_folder, log_time)

    ed_optimizer = EveryDreamOptimizer(
        args=args,
        optimizer_config=optimizer_config,
        model=model,
        epoch_len=epoch_len,
        plugin_runner=plugin_runner,
        log_writer=log_writer,
    )

    tv = setup_sana_training_variables(args)

    sample_generator = SampleGenerator(
        log_folder=log_folder,
        log_writer=log_writer,
        default_resolution=args.resolution[0],
        config_file_path=args.sample_prompts,
        batch_size=1,
        default_seed=args.seed,
        default_sample_steps=args.sample_steps,
        is_video=getattr(args, 'is_video', False),
        video_frames=getattr(args, 'video_frames', 81),
        video_fps=getattr(args, 'video_fps', 16),
    )

    logging.info(
        f"Starting SANA training — log_folder={log_folder}, "
        f"model={args.model_id}, "
        f"resolutions={args.resolution}, "
        f"batch_size={args.batch_size}"
    )

    _is_main = True # multiprocess placeholder
    if _is_main and sample_generator.generate_pretrain_samples:
        _, batch = next(enumerate(data_loader))
        generate_samples(model=model, sample_generator=sample_generator,
                         global_step=0, batch=batch,
                         args=args, device=device, log_folder=log_folder,
                         log_time=log_time, train_dtype=model.dtype,
                         vae_dtype=None)

    train_sana_loop(
        model=model,
        ed_optimizer=ed_optimizer,
        tv=tv,
        dataset=dataset,
        data_loader=data_loader,
        log_writer=log_writer,
        device=device,
        args=args,
        sample_generator=sample_generator,
        logdir=log_folder,
        log_time=log_time,
        plugin_runner=plugin_runner,
    )

    log_writer.close()

def _make_ckpt_path(logdir, args, epoch, tv: TrainingVariables):
    return os.path.join(logdir, 'ckpts',
                 f"{args.project_name}-ep{epoch:02}-gs{tv.global_step:05}-n{tv.total_trained_samples_count:06}")


def _do_post_optimizer_ema_update(model: SanaTrainingModel, tv: TrainingVariables, device, args: argparse.Namespace):
    # ── Main EMA update (self-flow teacher IS the main EMA) ──────
    if get_use_ema_decay_training(args):
        samples_since_ema = tv.total_trained_samples_count - tv.last_ema_total_trained_samples_count
        steps_since_ema = samples_since_ema / args.batch_size
        if steps_since_ema > args.ema_update_interval:
            effective_ema_decay_rate = args.ema_decay_rate ** samples_since_ema
            if model.ema_working_dir is not None:
                _f = os.path.join(model.ema_working_dir, "transformer_ema.safetensors")
                update_ema_disk(model.transformer, _f, effective_ema_decay_rate)
            else:
                update_ema(
                    model.transformer, model.transformer_ema,
                    effective_ema_decay_rate,
                    default_device=device, ema_device=args.ema_device,
                )
            tv.last_ema_total_trained_samples_count = tv.total_trained_samples_count


if __name__ == "__main__":
    main_sana()
