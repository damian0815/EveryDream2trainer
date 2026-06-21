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
import gc
import logging
import os
import random
import time
from argparse import Namespace

import numpy as np
from colorama import Fore, Style
from diffusers import AutoencoderKLWan
from tqdm.auto import tqdm

import torch
from torch.utils.tensorboard import SummaryWriter
from accelerate.utils import set_seed

import data.aspects as aspects_module
import data.resolver as resolver_module
from core.semaphore_files import check_semaphore_file_and_unlink, WANT_SAMPLES_SEMAPHORE_FILE, SAVE_FULL_SEMAPHORE_FILE, SAVE_FULL_AND_STOP_SEMAPHORE_FILE
from core.step import run_accumulation_loop, repeat_with_oom_handling, _dump_memory_snapshot, pause_memory_history, \
    compute_train_process_01
from core.step import get_best_match_resolution, choose_effective_batch_size
from core.loss import get_multirank_stratified_random_timesteps, build_self_flow_latents
from core.loss_sana import compute_sana_loss
from core.self_flow import get_self_flow_modules, compute_self_flow_loss
from core.flow_match_model import TrainFlowMatchEulerDiscreteScheduler
from core.log import setup_local_logger, log_args, do_log_step, append_epoch_log, LogData
from data.dataset import select_caption_variants
from model.sana_training_model import SanaTrainingModel, load_sana_model, save_sana_model
from model.sana_text_encoder import encode_prompts
from model.training_model import TrainingVariables, find_last_checkpoint
from train import update_ema
from optimizer.optimizers import EveryDreamOptimizer
from utils.inference_context import inference_guard
from utils.sample_generator import SampleGenerator
from utils.train_args import parse_train_args

import json
from plugins.plugins import PluginRunner, load_plugin

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

    image_source_items = resolver_module.resolve_sources(
        args.data_root, args, aspects_per_resolution
    )

    for item in image_source_items:
        if item.error is not None:
            logging.error(f"Skipping corrupt image {item.pathname}: {item.error}")
    image_source_items = [s for s in image_source_items if s.error is None]

    if args.skip_undersized_images:
        before = len(image_source_items)
        image_source_items = [s for s in image_source_items if s.is_feasible_for_any_resolution()]
        dropped = before - len(image_source_items)
        if dropped:
            logging.info(f"Dropped {dropped} images undersized at all resolutions")

    if not image_source_items:
        raise RuntimeError(
            f"No training images found in '{args.data_root}'. "
            "Check --data_root and that your folder contains supported image files."
        )

    logging.info(
        f"SANA data: {len(image_source_items)} source images across "
        f"resolutions {list(aspects_per_resolution.keys())}"
    )

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
    )

    data_loader = build_torch_dataloader(
        dataset,
        batch_size=args.batch_size,
        num_workers=args.num_dataloader_workers if args.num_dataloader_workers is not None else 4,
    )

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

    def nibble_loss_fn(nibble: dict) -> torch.Tensor:
        n = nibble["image"].shape[0]

        with torch.no_grad(), pause_memory_history():
            model.load_textenc_to_device(device)
            y, y_mask = repeat_with_oom_handling(
                initial_slice_size=tv.forward_slice_size,
                callback=lambda sz: encode_prompts(
                    model.tokenizer,
                    model.text_encoder,
                    nibble["captions"][caption_variant],
                    device,
                    max_sequence_length=model.max_sequence_length,
                    complex_human_instruction=model.complex_human_instruction or None,
                    dtype=model.dtype,
                    slice_size=sz
                ),
                oom_log_info=f"OOM gs:{tv.global_step}/l:{tv.accumulated_loss_images_count} SANA text encoder encode",
            )
            if args.offload_text_encoder:
                model.load_textenc_to_device('cpu')

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

        # loss_preview_image — concatenate model_pred, target, loss along spatial dim
        log_data.loss_preview_image = torch.cat(
            [model_pred, target, loss_full],
            dim=-2
        ).detach().clone().cpu()
        del model_pred, target, loss_full

        # Self-Flow representation loss
        do_self_flow = (
            model.self_flow_teacher_transformer is not None
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

            sf_student_mod, sf_teacher_mod = get_self_flow_modules(
                model.transformer, model.self_flow_teacher_transformer, args.self_flow_mode
            )

            # Student forward (grad-enabled) with hook to capture intermediate features
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

            # Teacher forward (no grad) with hook
            teacher_storage = {}
            def _sf_teacher_hook(module, inp, output):
                out = output[0] if isinstance(output, tuple) else output
                teacher_storage['h'] = out
            teacher_handle = sf_teacher_mod.register_forward_hook(_sf_teacher_hook)
            try:
                with torch.no_grad():
                    model.self_flow_teacher_transformer(
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
            if log_writer is not None:
                log_writer.add_scalar("loss/self_flow", l_rep_1d.mean().item(), global_step=tv.global_step)
            loss_1d = loss_1d + args.self_flow_gamma * l_rep_1d

        del y, y_mask, z

        return loss_1d

    if args.debug_save_memory_snapshots:
        _dump_memory_snapshot(
            os.path.join(getattr(args, '_snapshot_dir', '.'), f"gs{tv.global_step:06d}.pickle")
        )

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
        did_step_optimizer_cb=None,
        args=args,
        train_progress_01=train_progress_01
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



def generate_samples(model: SanaTrainingModel, sample_generator: SampleGenerator, global_step: int, batch: dict) -> None:
    logging.info(f"Generating samples at gs:{global_step}")
    with inference_guard(model.transformer):
        pipe = sample_generator.create_inference_pipe(
            model_being_trained=model,
            diffusers_scheduler_config=model.noise_scheduler.config,
        )
        was_tiling = getattr(pipe.vae, 'use_tiling')
        if was_tiling is None:
            # pipe does not support tiling
            pass
        else:
            pipe.vae.enable_tiling()
        sample_generator.reload_config()
        if batch is not None:
            flattened_captions_dict = [v
                                       for _, l in batch["captions"].items()
                                       for v in l]
            sample_generator.update_random_captions(flattened_captions_dict)
        sample_generator.generate_samples(pipe, global_step)
        if was_tiling is not None:
            if was_tiling:
                pipe.vae.enable_tiling()
            else:
                pipe.vae.disable_tiling()
        del pipe
    gc.collect()
    torch.cuda.empty_cache()


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

    if plugin_runner is not None:
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

        if plugin_runner is not None:
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

            if plugin_runner is not None:
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

            ed_optimizer.step_schedulers(tv.global_step)

            # Self-Flow EMA teacher update (independent interval from main EMA)
            if model.self_flow_teacher_transformer is not None:
                sf_interval = getattr(args, 'self_flow_ema_update_interval', 1)
                if ((tv.global_step + 1) % sf_interval) == 0:
                    update_ema(
                        model.transformer,
                        model.self_flow_teacher_transformer,
                        args.self_flow_ema_decay,
                        default_device=device,
                        ema_device=device,
                    )

            images_per_sec = full_batch["image"].shape[0] / (time.time() - step_start_time)
            log_data.images_per_sec_log_step.append(images_per_sec)

            if (tv.global_step + 1) % args.log_step == 0:
                logs = do_log_step(args, ed_optimizer, log_data, logdir, log_writer, model, tv)
                append_epoch_log(global_step=tv.global_step, epoch_pbar=epoch_pbar, gpu=gpu, log_writer=log_writer, **logs)
                torch.cuda.empty_cache()

            if global_step > 0 and global_step % args.save_every == 0:
                save_path = os.path.join(logdir, f"gs{global_step}")
                logging.info(f"Saving SANA model to {save_path}")
                save_sana_model(save_path, model=model, optimizer=ed_optimizer,  global_step=global_step, num_samples=tv.total_trained_samples_count)

            user_wants_samples = check_semaphore_file_and_unlink(WANT_SAMPLES_SEMAPHORE_FILE)
            if user_wants_samples or sample_generator.should_generate_samples(tv.global_step, local_step=local_step):
                generate_samples(model, sample_generator, global_step=global_step, batch=full_batch)

            if plugin_runner is not None:
                def plugin_runner_save_fn(path: str, step: int, num_samples: int) -> None:
                    save_sana_model(path, model=model, optimizer=ed_optimizer, global_step=step, num_samples=num_samples)

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
            if args.max_steps is not None and global_step >= args.max_steps:
                logging.info(f"Reached max_steps={args.max_steps}, stopping.")
                should_stop = True
                should_save = True

            if check_semaphore_file_and_unlink(SAVE_FULL_SEMAPHORE_FILE):
                should_save = True
            if check_semaphore_file_and_unlink(SAVE_FULL_AND_STOP_SEMAPHORE_FILE):
                should_save = True
                should_stop = True
            if should_save:
                logging.info("Save requested -> saving")
                ckpt_path = _make_ckpt_path(logdir, args, epoch, tv)
                save_sana_model(ckpt_path, model=model, optimizer=ed_optimizer, global_step=global_step, num_samples=tv.total_trained_samples_count)
            if should_stop:
                logging.info("Stop requested -> stopping")
                break


        epoch_pbar.update(1)
        if len(log_data.loss_epoch) > 0:
            loss_epoch = sum(log_data.loss_epoch) / len(log_data.loss_epoch)
            log_writer.add_scalar(tag="loss/epoch", scalar_value=loss_epoch, global_step=tv.global_step)

        if plugin_runner is not None:
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

    if plugin_runner is not None:
        plugin_runner.run_on_training_end(
            log_folder=logdir,
            project_name=args.project_name,
            global_step=tv.global_step,
        )

    logging.info("SANA training complete.")

    final_ckpt_path = _make_ckpt_path(logdir, args, epoch, tv)
    logging.info(f" * saving final model to {final_ckpt_path}...")
    save_sana_model(final_ckpt_path, model=model, optimizer=ed_optimizer, global_step=global_step, num_samples=tv.total_trained_samples_count)

    logging.info(f" * generating final samples...")
    _, batch = next(enumerate(data_loader))
    generate_samples(model, sample_generator, global_step=tv.global_step, batch=batch)

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
    model = load_sana_model(args)
    model.transformer.to(device)
    if model.self_flow_teacher_transformer is not None:
        model.self_flow_teacher_transformer.to(device)
        model.self_flow_proj_head.to(device)
    # if it's not quantized, push the text encoder to device
    if getattr(model.text_encoder, 'hf_quantizer', None) is None:
        if not args.offload_text_encoder:
            model.text_encoder.to(device)
    if not args.offload_vae:
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

    logging.info(f"Building data loader for resolutions: {args.resolution}")
    dataset, data_loader = build_sana_data_loader(args, seed=args.seed, plugin_runner=plugin_runner)
    epoch_len = len(data_loader)

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
        generate_samples(model, sample_generator, global_step=0, batch=batch)

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
        plugin_runner=plugin_runner,
    )

    log_writer.close()

def _make_ckpt_path(logdir, args, epoch, tv: TrainingVariables):
    return os.path.join(logdir, 'ckpts',
                 f"{args.project_name}-ep{epoch:02}-gs{tv.global_step:05}-n{tv.total_trained_samples_count:06}")

if __name__ == "__main__":
    main_sana()
