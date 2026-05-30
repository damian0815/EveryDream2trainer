import argparse
import contextlib
import gc
import logging
import os
import random
import time
from argparse import Namespace
from dataclasses import dataclass
from typing import Callable, Optional

import math
import torch
from diffusers import FlowMatchEulerDiscreteScheduler
from torch.amp import autocast
from torch.utils.tensorboard import SummaryWriter
import line_profiler
from torchvision import transforms

from core.flow_match_model import TrainFlowMatchEulerDiscreteScheduler
from core.log import LogData
from core.loss import get_noise, get_multirank_stratified_random_timesteps, get_model_prediction_and_target, get_loss, \
    get_local_contrastive_flow_loss, apply_negative_loss_hinge, get_contrastive_flow_matching_loss, get_clip_loss, \
    compute_saturation_penalty_loss
from core.self_flow import compute_self_flow_loss
from data.dataset import select_caption_variants
from model.training_model import TrainingVariables, TrainingModel, get_text_conditioning, Conditioning
from optimizer.optimizers import InfOrNanException, EveryDreamOptimizer
from plugins.plugins import PluginRunner
from utils.distributed import get_rank, is_distributed as _is_distributed, get_distributed_state_signal, StateSignal, \
    sync_ddp_gradients
import torch.distributed as dist

# Type alias: any callable that takes a nibble dict and returns a scalar loss tensor with grad.
# nibble_loss_fn is responsible for ALL model-specific work (forward pass, loss computation, logging).
# It must return a scalar Tensor with grad attached.
NibbleLossFn = Callable[
    [dict],         # nibble batch dict (same structure as full_batch but smaller)
    torch.Tensor,   # scalar loss tensor WITH grad attached (mean over nibble images)
]

# Global performance profiling storage
# Structure: {<label>: [duration_per_image, duration_per_image, ...]}
PERFORMANCE_PROFILE = {}


ENABLE_PERFORMANCE_LOGGING = False
def record_performance_timing(label: str, duration_seconds: float, num_images: int):

    """Record performance timing for a phase, normalized per image."""
    if not ENABLE_PERFORMANCE_LOGGING:
        return

    global PERFORMANCE_PROFILE
    if label not in PERFORMANCE_PROFILE:
        PERFORMANCE_PROFILE[label] = []
    duration_per_image = duration_seconds / max(1, num_images)
    PERFORMANCE_PROFILE[label].append(duration_per_image)
    logging.info(f"[PERF] {label}: {duration_seconds:.4f}s total, {duration_per_image:.6f}s per image ({num_images} images)")


def run_accumulation_loop(
    full_batch: dict,
    tv: TrainingVariables,
    ed_optimizer: EveryDreamOptimizer,
    nibble_loss_fn: NibbleLossFn,
    plugin_runner: Optional[PluginRunner],
    log_writer: SummaryWriter,
    steps_pbar,
    did_step_optimizer_cb: Optional[Callable],
    args: argparse.Namespace,
    train_progress_01: float = 0.0,
    model_for_autocast=None,
) -> None:
    """
    Model-agnostic nibble/accumulation/backward/optimizer-step loop.

    Iterates over nibble-sized slices of full_batch. For each nibble:
      - calls nibble_loss_fn(nibble) to get a scalar loss (with grad)
      - accumulates that loss into tv
      - fires a pre-emptive backward if accumulated >= max_backward_slice_size
      - fires backward + optimizer.step() when ready

    nibble_loss_fn is responsible for ALL model-specific work.
    It must return a scalar Tensor with grad attached.
    """
    if did_step_optimizer_cb is None:
        did_step_optimizer_cb = lambda: True

    if plugin_runner is None:
        plugin_runner = PluginRunner()

    remaining_batch = full_batch

    while remaining_batch is not None and remaining_batch['image'].shape[0] > 0:
        batch, remaining_batch = nibble_batch(remaining_batch, get_nibble_size(training_variables=tv))
        assert batch["runt_size"] == 0
        num_images = batch["image"].shape[0]

        use_bfloat16 = (
            model_for_autocast is not None
            and (getattr(model_for_autocast, 'is_sdxl', False) or getattr(args, 'force_bfloat16', False))
        )

        # Pre-emptive backward: fire if adding this nibble would exceed max_backward_slice_size
        if (tv.accumulated_loss_images_count > 0
                and tv.accumulated_loss_images_count + num_images > tv.max_backward_slice_size):
            with torch.amp.autocast('cuda', enabled=args.amp, dtype=torch.bfloat16 if use_bfloat16 else torch.float16):
                optimizer_backward(ed_optimizer, tv, plugin_runner,
                                   f'pre-emptive backward @{tv.accumulated_loss_images_count}/{tv.max_backward_slice_size}: ')

        # Call the model-specific nibble loss function
        loss_mean = nibble_loss_fn(batch)

        # Accumulate loss into tv
        try:
            tv.accumulate_loss(
                loss_mean,
                pathnames=batch.get("pathnames", [])[:num_images],
                captions=(batch.get("captions") or {}).get("default", [])[:num_images],
                timesteps=[],
            )
        except InfOrNanException:
            logging.error("Inf or NaN detected in loss, dropping this loss batch.")

        # Update progress bar
        steps_pbar.set_postfix({
            "_l": tv.accumulated_loss_images_count,
            "b": tv.backwarded_images_count,
            "os": str(tv.optimizer_step),
            "N": str(tv.total_trained_samples_count),
            "gs": str(tv.global_step),
        })

        # Regular backward + optimizer step (if threshold reached)
        should_step_optimizer = (
            (tv.backwarded_images_count + tv.accumulated_loss_images_count)
            >= tv.desired_effective_batch_size
        ) or tv.interleaved_bs1_count is not None

        if ((should_step_optimizer and tv.accumulated_loss_images_count > 0)
                or tv.accumulated_loss_images_count >= tv.max_backward_slice_size):
            with torch.amp.autocast('cuda', enabled=args.amp, dtype=torch.bfloat16 if use_bfloat16 else torch.float16):
                optimizer_backward(ed_optimizer, tv, plugin_runner, 'regular backward: ')

        if should_step_optimizer and tv.backwarded_images_count > 0:
            # Block until everyone wants to step (no-op in single-GPU mode)
            while True:
                global_state_signal = get_distributed_state_signal(StateSignal.WANTS_STEP, full_batch['image'].device)
                if global_state_signal == StateSignal.DONE:
                    return
                elif global_state_signal == StateSignal.WANTS_STEP:
                    break
                else:
                    raise RuntimeError(f"unknown state signal {global_state_signal}")

            sync_ddp_gradients(
                *[m for m in [
                    getattr(model_for_autocast, 'unet', None),
                    getattr(model_for_autocast, 'text_encoder', None),
                    getattr(model_for_autocast, 'text_encoder_2', None),
                ] if m is not None]
            )

            ed_optimizer.step_optimizer(tv.global_step, tv, log_data=None)

            tv.last_effective_batch_size = tv.backwarded_images_count
            tv.total_trained_samples_count += tv.backwarded_images_count
            tv.optimizer_step += 1
            tv.current_timestep_interval = None
            tv.backwarded_images_count = 0

            if tv.interleaved_bs1_count is not None:
                tv.interleaved_bs1_count += 1

            if (tv.interleaved_bs1_count is None
                    or tv.interleaved_bs1_count >= max(
                        1, tv.desired_effective_batch_size ** getattr(args, 'interleave_batch_size_1_alpha', 1)
                    )):
                tv.desired_effective_batch_size = choose_effective_batch_size(args, train_progress_01)
                if getattr(args, 'interleave_batch_size_1', False):
                    if tv.interleaved_bs1_count is None:
                        tv.interleaved_bs1_count = 0
                    else:
                        tv.interleaved_bs1_count = None

            did_step_optimizer_cb()


@line_profiler.profile
def train_step(
    full_batch: dict,
    model: TrainingModel,
    teacher_models: list,
    tv: TrainingVariables,
    train_progress_01: float,
    ed_optimizer: EveryDreamOptimizer,
    vae_dtype: torch.dtype,

    log_writer: SummaryWriter,
    log_data: LogData,
    steps_pbar,

    plugin_runner: PluginRunner,
    did_step_optimizer_cb: Callable|None,
    args: argparse.Namespace,
):
    t_step_start = time.perf_counter()

    def nibble_loss_fn(batch: dict) -> torch.Tensor:
        """
        SD/SDXL-specific per-nibble forward pass. Handles caption selection,
        noise generation, VAE encoding, text conditioning, model forward, and
        loss computation for all caption variants. Returns a combined scalar loss.
        """
        image_shape = batch["image"].shape
        batch_size = image_shape[0]
        num_images = batch_size

        # Caption selection
        t_caption_start = time.perf_counter()
        loss_scale = args.loss_scale * batch["loss_scale"]
        assert loss_scale.shape[0] == batch["image"].shape[0]

        assert type(batch["captions"]) is dict

        try:
            caption_variants = select_caption_variants(
                batch["captions"],
                requested_variants=args.caption_variants,
                all_caption_variants=args.all_caption_variants,
                caption_cross_concatenation_p=args.caption_cross_concatenation_p,
                caption_cross_concatenation_empty_half_p=args.caption_cross_concatenation_empty_half_p,
            )
        except RuntimeError as e:
            logging.error(f"Caught error {repr(e)} while calling select_caption_variants -> surprise cond dropout")
            logging.info(
                f"surprise cond dropout: ** Apparently no captions: {batch.get('pathnames', '(no paths)')}: {batch['captions']} - treating as cond dropout for this batch **")
            batch["captions"]["default"] = [model.cond_dropout_caption] * image_shape[0]
            batch["tokens"]["default"] = [model.cond_dropout_tokens] * image_shape[0]
            caption_variants = ['default']

        record_performance_timing("2_caption_selection", time.perf_counter() - t_caption_start, num_images)

        # Noise generation
        t_noise_start = time.perf_counter()
        with torch.no_grad():
            noise_shape = (batch_size, 4, image_shape[2] // 8, image_shape[3] // 8)
            noise = get_noise(
                noise_shape,
                device=model.device,
                dtype=model.dtype,
                pyramid_noise_discount=args.pyramid_noise_discount,
                zero_frequency_noise_ratio=args.zero_frequency_noise_ratio,
                batch_share_noise=(full_batch["do_contrastive_learning"] or args.batch_share_noise),
            )
        record_performance_timing("3_noise_generation", time.perf_counter() - t_noise_start, batch_size)

        # VAE encoding
        t_vae_start = time.perf_counter()
        assert batch["runt_size"] == 0  # _encode_latents assumption

        def _do_encode_latents(model_for_vae: TrainingModel):
            model_for_vae.vae.to(vae_dtype)
            these_latents = _encode_latents(model=model_for_vae, image=batch["image"], amp=args.amp, tv=tv)
            if args.offload_vae:
                model_for_vae.load_vae_to_device('cpu')
            return these_latents

        latents = _do_encode_latents(model)
        teacher_latents_list = [
            _do_encode_latents(tm) if tm.is_sdxl != model.is_sdxl else None
            for tm in teacher_models
        ]
        record_performance_timing("4_vae_encoding", time.perf_counter() - t_vae_start, batch_size)

        # Per-caption-variant forward + loss computation
        total_loss_mean = None
        for caption_variant in caption_variants:
            # Timestep generation
            t_timesteps_start = time.perf_counter()
            timesteps = _get_step_timesteps(
                count=batch["image"].shape[0],
                timesteps_range=batch["timesteps_range"],
                train_progress_01=train_progress_01,
                model=model,
                tv=tv,
                share_timesteps=batch["do_contrastive_learning"] or args.batch_share_timesteps,
                args=args,
            )
            # Apply shift for flow-matching schedulers
            if isinstance(model.noise_scheduler, TrainFlowMatchEulerDiscreteScheduler):
                timesteps = TrainFlowMatchEulerDiscreteScheduler.get_shifted_timesteps(
                    timesteps, model.noise_scheduler.timesteps
                )
                t_clamp_min = getattr(args, 'flow_match_t_clamp_min', None)
                t_clamp_max = getattr(args, 'flow_match_t_clamp_max', None)
                if t_clamp_min is not None or t_clamp_max is not None:
                    num_ts = model.noise_scheduler.timesteps.shape[0]
                    idx = torch.bucketize(timesteps, model.noise_scheduler.timesteps.flip(0))
                    idx = idx.clamp(
                        min=t_clamp_min if t_clamp_min is not None else 0,
                        max=t_clamp_max if t_clamp_max is not None else num_ts - 1,
                    )
                    timesteps = model.noise_scheduler.timesteps.flip(0)[idx]
            record_performance_timing("5_timesteps_generation", time.perf_counter() - t_timesteps_start, num_images / len(caption_variants))

            # Conditional dropout
            t_cond_dropout_start = time.perf_counter()
            cond_dropout_mask = build_cond_dropout_mask(
                batch=batch, timesteps=timesteps, model=model, tv=tv, train_progress_01=train_progress_01, args=args
            )
            is_cond_dropout_noise = cond_dropout_mask & (
                torch.rand(batch_size, device=model.unet.device) < args.cond_dropout_noise_p
            )
            record_performance_timing("6_cond_dropout", time.perf_counter() - t_cond_dropout_start, num_images / len(caption_variants))

            # Text conditioning generation
            teacher_mask = _generate_teacher_mask_or_none(
                training_model=model, args=args, teacher_models=teacher_models, timesteps=timesteps
            )
            t_conditioning_start = time.perf_counter()
            conditioning, teacher_conditionings, caption_str = _generate_conditioning(
                batch,
                caption_variant=caption_variant,
                cond_dropout_mask=cond_dropout_mask,
                model=model,
                teacher_models=teacher_models,
                teacher_mask=teacher_mask,
                args=args,
            )
            record_performance_timing("7_conditioning_generation", time.perf_counter() - t_conditioning_start, num_images / len(caption_variants))

            tv.cond_dropout_count += torch.sum(cond_dropout_mask)
            tv.non_cond_dropout_count += torch.sum(~cond_dropout_mask)

            model_forward_slice_size = tv.forward_slice_size

            # Model forward pass
            t_forward_start = time.perf_counter()
            model_forward_result = repeat_with_oom_handling(
                initial_slice_size=tv.forward_slice_size,
                callback=lambda slice_size: _do_model_forward(
                    model=model,
                    batch=batch,
                    latents=latents,
                    noise=noise,
                    conditioning=conditioning,
                    is_cond_dropout_noise=is_cond_dropout_noise,
                    timesteps=timesteps,
                    caption_variant=caption_variant,
                    teacher_models=teacher_models,
                    teacher_latents_list=teacher_latents_list,
                    teacher_mask=teacher_mask,
                    teacher_conditionings=teacher_conditionings,
                    forward_slice_size=slice_size,
                    tv=tv,
                    args=args,
                    debug_teacher=getattr(args, "debug_teacher", False),
                    log_writer=log_writer,
                ),
                oom_log_info=(
                    f"OOM step {tv.global_step} in unet forward with slice size {model_forward_slice_size} "
                    f"for full batch size {batch_size}. loss images accumulated: {tv.accumulated_loss_images_count}"
                ),
            )
            record_performance_timing("8_model_forward", time.perf_counter() - t_forward_start, num_images / len(caption_variants))

            loss_scale_variant = loss_scale[:model_forward_result.model_pred.shape[0]]

            # Loss computation
            t_loss_start = time.perf_counter()
            loss_1d = _do_loss(
                model_forward_result,
                model=model,
                batch=batch,
                is_cond_dropout=cond_dropout_mask,
                timesteps=timesteps,
                conditioning=conditioning,
                teacher_mask=teacher_mask,
                tv=tv,
                log_data=log_data,
                log_writer=log_writer,
                negative_loss_mask=loss_scale_variant < 0,
                args=args,
                verbose=(tv.global_step % 200 == 0),
            )
            record_performance_timing("9_loss_computation", time.perf_counter() - t_loss_start, num_images / len(caption_variants))

            if model.clip_model is not None:
                clip_loss_1d = get_clip_loss(
                    image_embeds=model_forward_result.clip_image_features,
                    text_embeds=model_forward_result.clip_pooled_text_features,
                    model=model,
                    mask=~cond_dropout_mask,
                )
                loss_1d += clip_loss_1d * args.clip_vision_contrastive_loss_lambda

            del model_forward_result

            # Per-timestep and per-image loss logging
            t_loss_accum_start = time.perf_counter()
            for i, used_timestep in enumerate(timesteps):
                used_timestep_detached = int(used_timestep.detach().item())
                current, count = log_data.loss_per_timestep[tv.batch_resolution].get(used_timestep_detached, (0, 0))
                log_data.loss_per_timestep[tv.batch_resolution][used_timestep_detached] = (
                    current + loss_1d[i].mean().detach().item(), count + 1
                )
                pathname = os.path.realpath(batch["pathnames"][i])
                if pathname not in log_data.loss_per_image_and_timestep[tv.batch_resolution]:
                    log_data.loss_per_image_and_timestep[tv.batch_resolution][pathname] = []
                log_data.loss_per_image_and_timestep[tv.batch_resolution][pathname].append(
                    (used_timestep_detached, loss_1d[i].mean().detach().item())
                )

            # Apply hinge and loss scale
            loss_1d = apply_negative_loss_hinge(
                loss_1d, (loss_scale_variant < 0).to(loss_1d.device), margin=args.negative_loss_margin
            )
            loss_1d = loss_1d * loss_scale_variant.abs().to(loss_1d.device)

            # Compute loss_mean for this variant
            if args.loss_mean_over_full_effective_batch:
                loss_mean_divisor = max(1, tv.desired_effective_batch_size)
            else:
                loss_mean_divisor = 1
            loss_mean = loss_1d.sum() / loss_mean_divisor

            log_data.loss_log_step_cd.append(loss_1d[cond_dropout_mask].mean().detach().item())
            log_data.loss_log_step_non_cd.append(loss_1d[~cond_dropout_mask].mean().detach().item())
            log_data.forward_size_coverage[loss_1d.shape[0]] += 1

            loss_step = loss_mean.detach().item()

            for t in timesteps:
                log_data.timestep_coverage[int(t.item())] += 1
                log_data.cumulative_timestep_coverage[int(t.item())] += 1

            steps_pbar.set_postfix({
                "rank": get_rank(),
                "loss/step": loss_step,
                "_f": tv.forward_slice_size,
                "_l": loss_1d.shape[0],
                "l": tv.accumulated_loss_images_count,
                "b": tv.backwarded_images_count,
                "os": str(tv.optimizer_step),
                "N": str(tv.total_trained_samples_count),
                "gs": str(tv.global_step),
            })

            del loss_1d
            log_data.loss_log_step.append(loss_step)
            log_data.loss_epoch.append(loss_step)
            record_performance_timing("10_loss_accumulation", time.perf_counter() - t_loss_accum_start, num_images / len(caption_variants))

            # Accumulate loss across caption variants
            total_loss_mean = loss_mean if total_loss_mean is None else total_loss_mean + loss_mean

        return total_loss_mean

    run_accumulation_loop(
        full_batch=full_batch,
        tv=tv,
        ed_optimizer=ed_optimizer,
        nibble_loss_fn=nibble_loss_fn,
        plugin_runner=plugin_runner,
        log_writer=log_writer,
        steps_pbar=steps_pbar,
        did_step_optimizer_cb=did_step_optimizer_cb,
        args=args,
        train_progress_01=train_progress_01,
        model_for_autocast=model,
    )

    t_step_end = time.perf_counter()
    record_performance_timing("0_total_step_time", t_step_end - t_step_start, full_batch['image'].shape[0])


def build_cond_dropout_mask(batch: dict, timesteps: torch.Tensor, model: TrainingModel, tv: TrainingVariables, train_progress_01: float, args: Namespace):
    # apply cond dropout
    initial_batch_size = args.batch_size if args.initial_batch_size is None else args.initial_batch_size
    final_batch_size = args.batch_size if args.final_batch_size is None else args.final_batch_size
    if final_batch_size == initial_batch_size:
        cdp_01_bs = 0
    else:
        cdp_01_bs = min(1, max(0, (tv.desired_effective_batch_size - initial_batch_size)
                               / (final_batch_size - initial_batch_size)))
    mask_contents = []
    for sample_index in range(timesteps.shape[0]):
        cdp_01_ts = 1 - (timesteps[
                             sample_index].cpu().item() / model.noise_scheduler.config.num_train_timesteps)
        if args.cond_dropout_curriculum_source == 'timestep':
            cdp_01 = cdp_01_ts
        elif args.cond_dropout_curriculum_source == 'batch_size':
            assert args.final_batch_size != args.initial_batch_size
            cdp_01 = cdp_01_bs
        elif args.cond_dropout_curriculum_source == 'global_step':
            cdp_01 = train_progress_01
        elif args.cond_dropout_curriculum_source == 'batch_size_and_timestep':
            assert args.final_batch_size != args.initial_batch_size
            cdp_01 = cdp_01_bs * cdp_01_ts
        else:
            raise ValueError("Unrecognized value for --cond_dropout_curriculum_source")
        final_cond_dropout = args.cond_dropout if args.final_cond_dropout is None else args.final_cond_dropout

        sample_cond_dropout = batch["cond_dropout"][sample_index]
        if sample_cond_dropout is None:
            sample_cond_dropout = args.cond_dropout

        if args.cond_dropout_curriculum_alpha:
            initial_factor = 1.0
            final_factor = final_cond_dropout / args.cond_dropout
            sample_cond_dropout *= get_exponential_scaled_value(cdp_01,
                                                               initial_value=initial_factor,
                                                               final_value=final_factor,
                                                               alpha=args.cond_dropout_curriculum_alpha)
        tv.cond_dropouts.append(sample_cond_dropout)
        mask_contents.append(random.random() <= sample_cond_dropout)

    return torch.tensor(mask_contents, device=model.unet.device, dtype=torch.bool)


def get_nibble_size(training_variables: TrainingVariables) -> int:
    if training_variables.desired_effective_batch_size - training_variables.backwarded_images_count <= 0:
        # The optimizer step threshold has already been reached in this train_step
        # call (deferred-step DDP mode).  Drain the remaining images using the
        # backward-slice size so their gradients fold into the pending step.
        return max(1, training_variables.max_backward_slice_size - training_variables.accumulated_loss_images_count)
    if training_variables.interleaved_bs1_count is not None:
        return 1

    permitted_until_optimizer_step = training_variables.desired_effective_batch_size - (training_variables.accumulated_loss_images_count + training_variables.backwarded_images_count)
    permitted_until_backward_step = training_variables.max_backward_slice_size - training_variables.accumulated_loss_images_count
    return max(1, min(permitted_until_optimizer_step, permitted_until_backward_step))


def nibble_batch(batch, take_count) -> tuple[dict, dict]:
    runt_size = batch["runt_size"]
    current_batch_size = batch["image"].shape[0]
    non_runt_size = current_batch_size - runt_size
    assert non_runt_size > 0

    nibble_size = min(non_runt_size, take_count)
    nibble = _subdivide_batch_part(batch, 0, nibble_size)
    nibble["runt_size"] = 0

    remaining_size = non_runt_size - nibble_size
    if remaining_size == 0:
        remainder = None
    else:
        remainder = _subdivide_batch_part(batch, nibble_size, non_runt_size)
        remainder["runt_size"] = 0
    return nibble, remainder


def subdivide_batch(batch, current_batch_size, desired_batch_size):
    if desired_batch_size >= current_batch_size:
        yield batch
        return
    runt_size = batch["runt_size"]
    non_runt_size = current_batch_size - runt_size
    for i, offset in enumerate(range(0, non_runt_size, desired_batch_size)):
        sub_batch = _subdivide_batch_part(batch, offset, offset + desired_batch_size)
        end = min(current_batch_size, offset + desired_batch_size)
        sub_batch["runt_size"] = max(0, end - non_runt_size)
        yield sub_batch


def _subdivide_batch_part(part, start, end):
    if type(part) is list or type(part) is torch.Tensor:
        return part[start:end]
    elif type(part) is dict:
        return {k: _subdivide_batch_part(v, start, end) for k, v in part.items()}
    else:
        return part


def choose_effective_batch_size(args, train_progress_01):
    batch_size = getattr(args, 'batch_size', 1)
    initial_bs = getattr(args, 'initial_batch_size', None)
    final_bs = getattr(args, 'final_batch_size', None)
    alpha = getattr(args, 'batch_size_curriculum_alpha', 1.0)
    return max(
        1,
        round(
            get_exponential_scaled_value(
                train_progress_01,
                initial_value=batch_size if initial_bs is None else initial_bs,
                final_value=batch_size if final_bs is None else final_bs,
                alpha=alpha,
            )
        ),
    )


def compute_train_process_01(
    epoch, step, steps_per_epoch, max_epochs, max_global_steps
):
    total_steps = steps_per_epoch * max_epochs
    if max_global_steps is not None:
        total_steps = min(total_steps, max_global_steps)
    steps_completed = steps_per_epoch * epoch + step
    return min(1, steps_completed / total_steps)


# ---------------------------------------------------------------------------
# Teacher distillation λ rolloff  (re-exported from core.teacher_lambda)
# ---------------------------------------------------------------------------
# The constants and get_teacher_lambda() function live in core/teacher_lambda.py
# so they can be imported by lightweight test modules without pulling in the
# full training stack.  Re-export everything from here so that existing call
# sites within core/step.py and elsewhere are unaffected.

from core.teacher_lambda import (   # noqa: F401  (re-exports)
    TEACHER_LAMBDA_SNR_ZERO_LO,
    TEACHER_LAMBDA_SNR_FULL_LO,
    TEACHER_LAMBDA_SNR_FULL_HI,
    TEACHER_LAMBDA_SNR_ZERO_HI,
    get_teacher_lambda,
)


def get_exponential_scaled_value(progress_01, initial_value, final_value, alpha=3.0):
    # Apply non-linear scaling with alpha (higher alpha = faster early descent)
    scaled_progress = 1.0 - (1.0 - progress_01) ** alpha
    return initial_value + scaled_progress * (final_value - initial_value)


def get_timestep_curriculum_range(
    progress_01,
    t_min_initial=800,
    t_max_initial=1000,
    t_min_final=0,
    t_max_final=400,
    alpha=3.0,
):
    # Interpolate boundaries
    min_t = min(
        1000,
        max(
            0,
            get_exponential_scaled_value(
                progress_01, t_min_initial, t_min_final, alpha=alpha
            ),
        ),
    )
    max_t = min(
        1000,
        max(
            0,
            get_exponential_scaled_value(
                progress_01, t_max_initial, t_max_final, alpha=alpha
            ),
        ),
    )

    assert min_t <= max_t
    return int(min_t), int(max_t)


def get_image_from_latents(latents, model: TrainingModel, args):
    with torch.no_grad():
        with autocast('cuda',
            enabled=args.amp, dtype=torch.bfloat16 if model.is_sdxl else torch.float16
        ):
            scaling_factor = 0.13025 if model.is_sdxl else 0.18215
            latents = latents / scaling_factor
            pixel_values = model.vae.decode(latents, return_dict=False)[0]
        del latents
        pixel_values = pixel_values.to(torch.float32)
        pixel_values = torch.clamp((pixel_values + 1.0) / 2.0, 0.0, 1.0)
        return pixel_values


def get_slices(batch_size, slice_size):
    num_slices = math.ceil(batch_size / slice_size)
    for slice_index in range(num_slices):
        slice_start = slice_index * slice_size
        slice_end = min(slice_start + slice_size, batch_size)
        if slice_end <= slice_start:
            break
        yield slice_start, slice_end


def get_best_match_resolution(resolutions: list[int], image_pixel_count: int) -> int:
    error = [image_pixel_count / (r*r) for r in resolutions]
    best_resolution_index = min([i for i in range(len(resolutions))], key=lambda i: abs(math.log(error[i])))
    return resolutions[best_resolution_index]





def get_uniform_timesteps(
    batch_size, batch_share_timesteps, device, timesteps_ranges
):
    """if continuous_float_timesteps is True, return float timestamps (continuous), otherwise return int timestamps (discrete)"""
    timesteps = torch.cat(
        [torch.randint(a, b, size=(1,), device=device) for a, b in timesteps_ranges]
    )
    if batch_share_timesteps:
        timesteps = timesteps[:1].repeat((batch_size,))
    timesteps = timesteps.long()
    # logging.info(f"get_timesteps: {timesteps.detach().cpu().tolist()} from ranges: {timesteps_ranges}")
    return timesteps



def _get_step_timesteps(count: int, timesteps_range: tuple, train_progress_01: float, model: TrainingModel, tv: TrainingVariables, share_timesteps: bool, args):
    timesteps = _get_step_timesteps_internal(
        count,
        full_batch_timesteps_range=timesteps_range,
        train_progress_01=train_progress_01,
        model=model,
        tv=tv,
        args=args,
    )
    # share timestep?
    if share_timesteps:
        timestep_index = torch.randint(count, size=(1,))[0]
        timesteps = timesteps[timestep_index].unsqueeze(0).repeat(count)

    return timesteps


_has_checked_bad_distribution = False


def _get_step_timesteps_internal(
    full_batch_size: int,
    full_batch_timesteps_range,
    train_progress_01: float,
    model: TrainingModel,
    tv: TrainingVariables,
    args,
) -> torch.Tensor:

    if args.timesteps_multirank_stratified:
        # the point of multirank stratified is to spread timesteps evenly across the batch.
        # so we need to do a dance here to make sure that we're actually spreading across the
        # desired_effective_batch_size - which will be "nibbled" later in chunks
        while tv.remaining_stratified_timesteps is None or tv.remaining_stratified_timesteps.shape[0] < max(full_batch_size,
                                                                                                            tv.desired_effective_batch_size):
            next_timesteps = get_multirank_stratified_random_timesteps(
                tv.desired_effective_batch_size,
                device=model.unet.device,
                distribution=args.timesteps_multirank_stratified_distribution,
                alpha=args.timesteps_multirank_stratified_alpha,
                beta=args.timesteps_multirank_stratified_beta,
                mode_scale=args.timesteps_multirank_stratified_mode_scale,
                scheduler=model.noise_scheduler,
                stratify=args.timesteps_multirank_stratified_stratify,
            )
            tv.remaining_stratified_timesteps = next_timesteps if tv.remaining_stratified_timesteps is None else torch.cat(
                [tv.remaining_stratified_timesteps, next_timesteps])
        timesteps: torch.Tensor = tv.remaining_stratified_timesteps[:full_batch_size]
        tv.remaining_stratified_timesteps = tv.remaining_stratified_timesteps[full_batch_size:]
        return timesteps
    elif args.timestep_interval_sampling:
        # --- SNR-interval timestep sampling ---
        # All samples in a batch share the same SNR-homogeneous interval.
        # Interval is chosen once per optimizer step and held until cleared after step_optimizer().
        if tv.current_timestep_interval is None:
            tv.current_timestep_interval = random.choice(tv.timestep_intervals)
        t_lo, t_hi = tv.current_timestep_interval
        timesteps = torch.randint(
            low=t_lo,
            high=max(t_lo + 1, t_hi + 1),   # randint high is exclusive
            size=(full_batch_size,),
            device=model.unet.device,
        ).long()
        return timesteps
    else:
        if full_batch_timesteps_range is not None:
            timesteps_ranges_base = full_batch_timesteps_range
        else:
            if args.timestep_curriculum_alpha == 0:
                timestep_range = (args.timestep_start, args.timestep_end)
            else:
                t_min_initial = args.timestep_initial_start
                t_max_initial = args.timestep_initial_end
                t_min_final = args.timestep_start
                t_max_final = args.timestep_end
                timestep_range = get_timestep_curriculum_range(progress_01=train_progress_01,
                                                               t_min_initial=t_min_initial,
                                                               t_max_initial=t_max_initial,
                                                               t_min_final=t_min_final,
                                                               t_max_final=t_max_final,
                                                               alpha=args.timestep_curriculum_alpha)
                print('timestep range:', timestep_range)
            timesteps_ranges_base = [timestep_range] * full_batch_size

        # randomly expand
        num_train_timesteps = model.noise_scheduler.config.num_train_timesteps

        def lerp(x, in_min, in_max, out_min, out_max):
            if (in_max - in_min) == 0:
                return in_min
            pct = (x - in_min) / (in_max - in_min)
            return out_min + pct * (out_max - out_min)

        timesteps_ranges_expanded = [(
            # maybe make 8 a CLI arg?
            min(num_train_timesteps,
                max(0, round(lerp(pow(random.random(), 8), 0, 1, tsr[0], 0)))),
            min(num_train_timesteps,
                max(0, round(lerp(pow(random.random(), 8), 0, 1, tsr[1], num_train_timesteps)))),
        ) for tsr in timesteps_ranges_base]

        timesteps: torch.LongTensor = get_uniform_timesteps(
            batch_size=full_batch_size,
            batch_share_timesteps=False,
            device=model.device,
            timesteps_ranges=timesteps_ranges_expanded,
        )

        return timesteps



def optimizer_backward(optimizer: EveryDreamOptimizer, tv: TrainingVariables, plugin_runner: PluginRunner, log_hint=''):
    """
    Run a backward pass on the currently-accumulated loss.
    """
    if tv.accumulated_loss_images_count == 0:
        logging.warning("no accumulated loss images, not doing backward")
    else:
        try:
            plugin_runner.run_on_optimizer_backward(loss=tv.accumulated_loss)
            gc.collect()
            torch.cuda.empty_cache()
            #print(f"optimizer backward rank {get_rank()} (why: '{log_hint}')")
            optimizer.backward(tv.accumulated_loss)
            #print(f"optimizer backward rank {get_rank()} done")
            if not optimizer.use_grad_scaler:
                # if we have a grad scaler it suppresses NaN/Inf for us.
                # otherwise, we have to suppress NaN/Inf manually
                optimizer.check_for_inf_or_nan(tv, log_hint + 'after backward')
            tv.effective_backward_size = tv.accumulated_loss_images_count
            tv.backwarded_images_count += tv.accumulated_loss_images_count
            tv.clear_accumulated_loss()
            tv.register_backward_oom_or_not(oomed=False)
        except InfOrNanException:
            logging.error(
                "* Caught Inf or NaN during backward pass, clearing accumulated loss and resetting optimizer"
            )
            optimizer.zero_grad(set_to_none=True)
            tv.backwarded_images_count = 0
        except torch.OutOfMemoryError:
            logging.error(f"OOM step {tv.global_step} during optimizer.backward of {tv.accumulated_loss_images_count} accumulated loss images @resolution {tv.batch_resolution}")
            logging.error(f" -> dropping this batch of {tv.accumulated_loss_images_count} accumulated loss images")
            tv.register_backward_oom_or_not(oomed=True)

        tv.clear_accumulated_loss()
        #print(f"leaving optimizer_backward rank {get_rank()}")


def _encode_latents(
    model: TrainingModel,
    image: torch.Tensor,
    amp: bool,
    tv: TrainingVariables
):
    forward_slice_size = tv.forward_slice_size
    batch_size = image.shape[0]
    pixel_values = image.to(memory_format=torch.contiguous_format).to(model.unet.device)
    imagenet_normalize = False #model.is_sdxl
    if imagenet_normalize:
        pixel_values = transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])(pixel_values)
    with torch.no_grad(), torch.amp.autocast('cuda', enabled=amp):
        successfully_encoded_latents = False
        enable_vae_attention_slicing = False
        while not successfully_encoded_latents:
            latents_slices = []
            try:
                for slice_start, slice_end in get_slices(batch_size, forward_slice_size):
                    if enable_vae_attention_slicing:
                        model.vae.enable_slicing()
                    try:
                        latents_slice = model.vae.encode(pixel_values[slice_start:slice_end], return_dict=False)
                    except Exception:
                        logging.error(
                            f"vae.encode failed for batch slice [{slice_start}:{slice_end}] @resolution {tv.batch_resolution}. pixel_values shape is {pixel_values.shape}")
                        raise
                    model.vae.disable_slicing()
                    vae_scaling_factor = 0.13025 if model.is_sdxl else 0.18215
                    latents_slice = latents_slice[0].sample() * vae_scaling_factor
                    latents_slices.append(latents_slice)
                    del latents_slice
                successfully_encoded_latents = True
            except (torch.cuda.OutOfMemoryError, torch.OutOfMemoryError):
                logging.error(
                    f"OOM step {tv.global_step} in vae.encode encoding slice size {forward_slice_size} from pixel values "
                    f"of shape {pixel_values.shape}, from {slice_start} to {slice_end} of {batch_size}. "
                    f"loss images accumulated: {tv.accumulated_loss_images_count}")
                if forward_slice_size > 1:
                    forward_slice_size = forward_slice_size // 2
                    logging.info(f" -> Trying again with forward slice size {forward_slice_size}")
                elif not enable_vae_attention_slicing:
                    enable_vae_attention_slicing = True
                    logging.info(f" -> Trying again with vae attention slicing enabled")
                else:
                    logging.error(
                        " -> Already at forward slice size 1 with vae attention slicing enabled, cannot reduce further")
                    raise

        del pixel_values
        latents = torch.cat(latents_slices).to(model.unet.dtype)
        del latents_slices
        return latents

def _generate_teacher_mask_or_none(training_model: TrainingModel, args: argparse.Namespace, teacher_models: list, timesteps: torch.Tensor) -> torch.Tensor|None:
    if not teacher_models:
        return None
    teacher_model = teacher_models[0]

    if args.teacher_timestep_max is not None:
        # scale teacher_p based on timesteps
        # 1000...args.teacher_timestep_max: p = 0
        # args.teacher_timestep_max...0: ramp from 0 to args.teacher_p
        scale = torch.maximum(
            teacher_timestep_max - timesteps.float(),
            torch.zeros([1])
        ) / teacher_timestep_max
        teacher_p = args.teacher_p * scale

    teacher_lambda = get_teacher_lambda(timesteps, args, training_model.noise_scheduler)
    batch_size = timesteps.shape[0]
    return ((torch.rand(batch_size) < args.teacher_p) & (teacher_lambda > 0)).to(teacher_model.unet.device)

def _generate_conditioning(
    batch: dict,
    caption_variant: str|tuple[str, str],
    cond_dropout_mask: torch.Tensor,
    model: TrainingModel,
    teacher_models: list,
    teacher_mask,
args) -> tuple[Conditioning, list, list[str]]:

    if type(caption_variant) is str:
        caption_variant = (caption_variant, )

    with (
        torch.no_grad()
        if args.disable_textenc_training
        else contextlib.nullcontext()
    ):
        # todo move this logic to the dataloader
        batch_size = batch["image"].shape[0]

        model.load_textenc_to_device(model.device)

        caption_str = []
        tokens = []
        tokens_2 = []
        teacher_tokens = []
        teacher_tokens_2 = []
        add_time_ids = []

        for key in caption_variant:
            this_tokens_2 = None
            this_add_time_ids = None
            this_teacher_tokens = None
            this_teacher_tokens_2 = None
            # Use the first teacher's cond-dropout tokens for teacher-specific tokenisation
            # (v1 limitation: only the first teacher may have a heterogeneous tokenizer).
            first_teacher = teacher_models[0] if teacher_models else None
            if key is None:
                # empty variant half
                this_caption_str = [model.cond_dropout_caption] * batch_size
                this_tokens = model.cond_dropout_tokens.unsqueeze(0).repeat((batch_size, 1))
                if model.is_sdxl:
                    this_tokens_2 = model.cond_dropout_tokens_2.unsqueeze(0).repeat((batch_size, 1))
                if first_teacher is not None:
                    this_teacher_tokens = first_teacher.cond_dropout_tokens.unsqueeze(0).repeat((batch_size, 1))
                    if first_teacher.is_sdxl:
                        this_teacher_tokens_2 = first_teacher.cond_dropout_tokens_2.unsqueeze(0).repeat((batch_size, 1))
            else:
                this_caption_str = [batch["captions"][key][i] for i in range(batch_size)]
                assert all(c is not None for c in caption_str)
                this_tokens = torch.stack([batch["tokens"][key][i] for i in range(batch_size)])
                if model.is_sdxl:
                    this_tokens_2 = torch.stack([batch["tokens_2"][key][i] for i in range(batch_size)])
                if "tokens_teacher" in batch:
                    for i in range(batch_size):
                        if batch["tokens_teacher"][key][i] is None:
                            print("** None in tokens_teacher?")
                            batch["tokens_teacher"][key][i] = batch["tokens"][key][i]
                    this_teacher_tokens = torch.stack([batch["tokens_teacher"][key][i] for i in range(batch_size)])
                if "tokens_teacher_2" in batch:
                    for i in range(batch_size):
                        if batch["tokens_teacher_2"][key][i] is None:
                            print("** None in tokens_teacher_2?")
                            batch["tokens_teacher_2"][key][i] = batch["tokens"][key][i]
                    this_teacher_tokens_2 = torch.stack([batch["tokens_teacher_2"][key][i] for i in range(batch_size)])

            if model.is_sdxl:
                this_add_time_ids = batch["add_time_ids"].to(model.unet.device)

            for i in range(cond_dropout_mask.shape[0]):
                if cond_dropout_mask[i]:
                    this_caption_str[i] = model.cond_dropout_caption
                    this_tokens[i] = model.cond_dropout_tokens
                    if model.is_sdxl:
                        this_tokens_2[i] = model.cond_dropout_tokens_2
                    if first_teacher is not None:
                        this_teacher_tokens[i] = first_teacher.cond_dropout_tokens
                        if first_teacher.is_sdxl:
                            this_teacher_tokens_2[i] = first_teacher.cond_dropout_tokens_2

            caption_str.append(this_caption_str)
            tokens.append(this_tokens)
            tokens_2.append(this_tokens_2)
            teacher_tokens.append(this_teacher_tokens)
            teacher_tokens_2.append(this_teacher_tokens_2)
            add_time_ids.append(this_add_time_ids)

        conditioning_list = []
        # per_teacher_conditioning_lists[j] accumulates Conditioning objects across caption variants
        # for teacher_models[j]. If a teacher has no dedicated text encoder (text_encoder is None),
        # its list remains empty and we store None in teacher_conditionings.
        per_teacher_conditioning_lists: list[list] = [[] for _ in teacher_models]

        for i, key in enumerate(caption_variant):
            encoder_hidden_states, encoder_pooled_embeds, encoder_2_hidden_states, encoder_2_pooled_embeds = get_text_conditioning(
                tokens[i], tokens_2[i], caption_str[i], model, args
            )
            conditioning_list.append(Conditioning(encoder_hidden_states, encoder_pooled_embeds,
                                        encoder_2_hidden_states, encoder_2_pooled_embeds,
                                        add_time_ids[i]))

            if teacher_mask is not None and torch.sum(teacher_mask) > 0:
                for j, tm in enumerate(teacher_models):
                    if tm.text_encoder is None:
                        continue  # shares student text encoder; no separate conditioning needed
                    with torch.no_grad():
                        # teacher[0] may have pre-tokenised teacher tokens from the dataloader;
                        # teacher[j>0] falls back to student tokens (v1 limitation).
                        if j == 0:
                            t_tokens = teacher_tokens[i] if teacher_tokens[i] is not None else tokens[i]
                            t_tokens_2 = teacher_tokens_2[i] if teacher_tokens_2[i] is not None else tokens_2[i]
                        else:
                            t_tokens = tokens[i]
                            t_tokens_2 = tokens_2[i]
                        te_hs, te_pe, te_hs2, te_pe2 = get_text_conditioning(
                            t_tokens, t_tokens_2, caption_str[i], tm, args
                        )
                        per_teacher_conditioning_lists[j].append(Conditioning(
                            te_hs, te_pe, te_hs2, te_pe2,
                            add_time_ids[i] if tm.is_sdxl else None
                        ))

        # concatenate
        conditioning = Conditioning.cat(conditioning_list)
        teacher_conditionings: list = [
            Conditioning.cat(tc_list) if tc_list else None
            for tc_list in per_teacher_conditioning_lists
        ]

        if args.offload_text_encoder:
            model.load_textenc_to_device('cpu')

    return conditioning, teacher_conditionings, caption_str


@dataclass
class ModelForwardReturnType:
    model_pred: torch.Tensor
    target: torch.Tensor
    noisy_latents: torch.Tensor
    lcf_midblock_out: torch.Tensor | None
    lcf_midblock_clean_out: torch.Tensor | None
    teacher_target: torch.Tensor | None
    clip_image_features: torch.Tensor | None
    clip_pooled_text_features: torch.Tensor | None
    self_flow_student_features: torch.Tensor | None
    self_flow_teacher_features: torch.Tensor | None

def _do_model_forward(
    model: TrainingModel,
    batch: dict, latents: torch.Tensor, noise: torch.Tensor, conditioning: Conditioning, is_cond_dropout_noise: torch.Tensor,
    timesteps: torch.Tensor, caption_variant: str|tuple[str, str],
    teacher_models: list, teacher_latents_list: list, teacher_mask: torch.Tensor|None, teacher_conditionings: list,
    forward_slice_size: int,
    tv: TrainingVariables, args: Namespace,
    debug_teacher: bool = False,
    log_writer: SummaryWriter|None = None,
) -> ModelForwardReturnType:
    batch_size = timesteps.shape[0]
    do_local_contrastive_flow_loss = (random.random() < args.local_contrastive_flow_loss_p)
    do_self_flow = (
        model.self_flow_teacher_unet is not None
        and random.random() < args.self_flow_p
    )

    model_pred_all = []
    lcf_midblock_out_all = []
    lcf_midblock_out_clean_all = []
    target_all = []
    noisy_latents_all = []
    teacher_target_all = []
    clip_image_features_all = []
    clip_pooled_text_features_all = []
    self_flow_student_features_all = []
    self_flow_teacher_features_all = []

    slices = list(get_slices(batch_size, forward_slice_size))
    for slice_index, (slice_start, slice_end) in enumerate(slices):
        # print(f'slice {slice_index} @ res {image_shape[2:4]} (base {args.resolution[0]}), sssf {slice_size_scale_factor}, bs {batch_size}, slice size {forward_slice_size}')
        def _safe_slice(x):
            return None if x is None else x[slice_start:slice_end]
        latents_slice = latents[slice_start:slice_end]
        teacher_latents_list_slice = [_safe_slice(tl) for tl in teacher_latents_list]
        noise_slice = noise[slice_start:slice_end]
        timesteps_slice = timesteps[slice_start:slice_end]
        is_cond_dropout_noise_slice = is_cond_dropout_noise[slice_start:slice_end]
        teacher_mask_slice = _safe_slice(teacher_mask)

        conditioning_slice = conditioning.slice(slice_start, slice_end)
        teacher_conditionings_slice = [
            c.slice(slice_start, slice_end) if c is not None else None
            for c in teacher_conditionings
        ]

        try:
            if do_local_contrastive_flow_loss:
                if type(model.noise_scheduler) is not TrainFlowMatchEulerDiscreteScheduler:
                    raise NotImplementedError("Local Contrastive Flow loss only works with flow matching scheduler")
                lcf_timestep_threshold = model.noise_scheduler.get_best_timestep_for_sigma(0.2)
                lcf_mask_slice = (timesteps_slice < lcf_timestep_threshold)
            else:
                lcf_mask_slice = None

            # Sample a second set of timesteps s for Self-Flow representation learning
            if do_self_flow:
                n_ts = model.noise_scheduler.config.num_train_timesteps
                self_flow_s_timesteps_slice = torch.randint(
                    0, n_ts, (slice_end - slice_start,), device=timesteps_slice.device
                )
                # shift
                if isinstance(model.noise_scheduler, TrainFlowMatchEulerDiscreteScheduler):
                    self_flow_s_timesteps_slice = TrainFlowMatchEulerDiscreteScheduler.get_shifted_timesteps(self_flow_s_timesteps_slice, model.noise_scheduler.timesteps)
            else:
                self_flow_s_timesteps_slice = None

            model_pred_result = get_model_prediction_and_target(
                latents=latents_slice,
                conditioning=conditioning_slice,
                noise=noise_slice,
                timesteps=timesteps_slice,
                is_cond_dropout_noise=is_cond_dropout_noise_slice,
                model=model,
                args=args,
                teacher_models=teacher_models,
                teacher_latents_list=teacher_latents_list_slice,
                teacher_mask=teacher_mask_slice,
                teacher_conditionings=teacher_conditionings_slice,
                lcf_mask=lcf_mask_slice,
                self_flow_s_timesteps=self_flow_s_timesteps_slice,
                global_step=tv.global_step,
                debug_teacher=debug_teacher,
                log_writer=log_writer
            )

            model_pred_all.append(model_pred_result.model_pred)
            target_all.append(model_pred_result.target)
            noisy_latents_all.append(model_pred_result.noisy_latents)
            teacher_target_all.append(model_pred_result.teacher_target)
            lcf_midblock_out_all.append(model_pred_result.midblock_out)
            self_flow_student_features_all.append(model_pred_result.self_flow_student_features)
            self_flow_teacher_features_all.append(model_pred_result.self_flow_teacher_features)

            # del
            model_pred_result.target = None
            model_pred_result.noisy_latents = None
            model_pred_result.teacher_target = None
            model_pred_result.midblock_out = None
            model_pred_result.self_flow_student_features = None
            model_pred_result.self_flow_teacher_features = None

            if model.clip_model:
                with torch.no_grad():
                    images_slice = batch["image"][slice_start:slice_end]
                    pixels = model.clip_processor(images=images_slice, return_tensors="pt").pixel_values.to(
                        model.device)  # [B, 3, H, W]
                    image_embeds = model.clip_model.get_image_features(pixels)  # [B, 1024]
                text_embeds = model.clip_model.text_projection(
                    conditioning_slice.pooled_embeds.to(dtype=model.clip_model.text_projection.weight.dtype))  # [B, 1024]
                clip_image_features_all.append(image_embeds)
                clip_pooled_text_features_all.append(text_embeds)
                del image_embeds, text_embeds

            if do_local_contrastive_flow_loss:
                anchor_timestep = model.noise_scheduler.get_best_timestep_for_sigma(0)
                anchor_timesteps_slice = torch.full_like(timesteps_slice, anchor_timestep)

                model_pred_result_clean = get_model_prediction_and_target(
                    latents=latents_slice,
                    conditioning=conditioning_slice,
                    noise=noise_slice,
                    timesteps=anchor_timesteps_slice,
                    is_cond_dropout_noise=is_cond_dropout_noise_slice,
                    model=model,
                    args=args,
                    teacher_models=[],
                    teacher_latents_list=[],
                    teacher_mask=None,
                    teacher_conditionings=[],
                    mask=lcf_mask_slice,
                    lcf_mask=lcf_mask_slice,
                )
                lcf_midblock_out_clean_all.append(model_pred_result_clean.midblock_out)
                del model_pred_result_clean

            del model_pred_result, conditioning_slice

        except (
                torch.cuda.OutOfMemoryError,
                torch.OutOfMemoryError,
        ):
            if type(caption_variant) is tuple:
                caption_variant_actual = caption_variant[1] if caption_variant[0] is None else caption_variant[0]
            else:
                caption_variant_actual = caption_variant
            logging.error(
                f"OOM step {tv.global_step} in unet forward @resolution {tv.batch_resolution} slice size {tv.forward_slice_size}, from latents " 
                f"of shape {latents_slice.shape} (samples {slice_start} to {slice_end} of {batch_size}). "
                f"loss images accumulated: {tv.accumulated_loss_images_count}, caption: f{caption_variant} - batch: {[os.path.realpath(x) for x in batch['pathnames']]}, {batch['captions'][caption_variant_actual]}, timesteps: {timesteps_slice.detach().cpu().tolist()}"
            )
            raise

    model_pred = torch.cat(model_pred_all)
    assert model_pred.shape[0] == batch_size
    assert timesteps.shape[0] == model_pred.shape[0]

    target = torch.cat(target_all)
    noisy_latents = torch.cat(noisy_latents_all)
    if teacher_mask is None or all(t is None for t in teacher_target_all):
        teacher_target = None
    else:
        teacher_target = torch.cat(teacher_target_all)
    lcf_midblock_out = torch.cat(lcf_midblock_out_all) if do_local_contrastive_flow_loss else None
    lcf_midblock_out_clean = torch.cat(lcf_midblock_out_clean_all) if do_local_contrastive_flow_loss else None

    clip_image_features = torch.cat(clip_image_features_all) if model.clip_model else None
    clip_pooled_text_features = torch.cat(clip_pooled_text_features_all) if model.clip_model else None

    if do_self_flow and any(f is not None for f in self_flow_student_features_all):
        self_flow_student_features = torch.cat([f for f in self_flow_student_features_all if f is not None])
        self_flow_teacher_features = torch.cat([f for f in self_flow_teacher_features_all if f is not None])
    else:
        self_flow_student_features = None
        self_flow_teacher_features = None

    return ModelForwardReturnType(
        model_pred=model_pred,
        target=target,
        noisy_latents=noisy_latents,
        lcf_midblock_out=lcf_midblock_out,
        lcf_midblock_clean_out=lcf_midblock_out_clean,
        teacher_target=teacher_target,
        clip_image_features=clip_image_features,
        clip_pooled_text_features=clip_pooled_text_features,
        self_flow_student_features=self_flow_student_features,
        self_flow_teacher_features=self_flow_teacher_features,
    )

def _do_loss(
    model_forward_result: ModelForwardReturnType,
    model: TrainingModel,
    batch: dict,
    is_cond_dropout: torch.Tensor,
    teacher_mask: torch.Tensor,
    timesteps: torch.Tensor,
    negative_loss_mask: torch.Tensor,
    conditioning: Conditioning,
    args: Namespace,
    tv: TrainingVariables,
    log_data: LogData,
    log_writer: SummaryWriter,
    verbose: bool=False,
) -> torch.Tensor:
    """
    Returns 1D loss tensor of shape [B].
    """
    mask_img = None if batch["mask"] is None else batch["mask"][:model_forward_result.model_pred.shape[0]]

    contrastive_loss_scale = 0

    loss = get_loss(
        model_forward_result.model_pred,
        model_forward_result.target,
        is_cond_dropout=is_cond_dropout,
        mask_img=mask_img,
        timesteps=timesteps,
        negative_loss_mask=negative_loss_mask,
        noise_scheduler=model.noise_scheduler,
        prompt_embeds=conditioning.prompt_embeds,
        do_contrastive_learning=batch["do_contrastive_learning"],
        contrastive_loss_scale=contrastive_loss_scale,
        verbose=verbose,
        args=args,
    ).to(dtype=model.dtype)
    if teacher_mask is not None and model_forward_result.teacher_target is not None:
        teacher_loss = get_loss(
            model_forward_result.model_pred,
            model_forward_result.teacher_target,
            is_cond_dropout=is_cond_dropout,
            mask_img=mask_img,
            timesteps=timesteps,
            negative_loss_mask=negative_loss_mask,
            noise_scheduler=model.noise_scheduler,
            prompt_embeds=conditioning.prompt_embeds,
            do_contrastive_learning=False,
            contrastive_loss_scale=0,
            verbose=verbose,
            args=args,
        ).to(dtype=model.dtype)
        # only the masked entries
        teacher_loss[~teacher_mask] = 0
        teacher_lambda = get_teacher_lambda(
            timesteps, args, noise_scheduler=model.noise_scheduler
        ).to(dtype=teacher_loss.dtype, device=teacher_loss.device)
        # reshape for broadcast: [B] -> [B, 1, 1, 1]
        teacher_lambda = teacher_lambda.view(-1, *([1] * (teacher_loss.dim() - 1)))
        loss += teacher_loss * teacher_lambda
        del teacher_loss, teacher_lambda

    if model_forward_result.lcf_midblock_out is not None:
        # doing local contrastive flow loss
        lcf_timestep_threshold = model.noise_scheduler.get_best_timestep_for_sigma(0.2)
        mask = (~negative_loss_mask).to(timesteps.device) & (
            timesteps < lcf_timestep_threshold
        )
        log_writer.add_scalar("loss/LCF sample count", mask.detach().sum().item(), global_step=tv.global_step)
        pathnames_resolved = [
            os.path.realpath(p) for p in batch["pathnames"]
        ]
        loss_lcf = get_local_contrastive_flow_loss(
            model_forward_result.lcf_midblock_out,
            model_forward_result.lcf_midblock_clean_out,
            low_noise_timesteps_mask=mask,
            unique_identifiers=pathnames_resolved,
            temperature=args.local_contrastive_flow_temperature
        )
        loss += (loss_lcf.to(dtype=loss.dtype) * args.local_contrastive_flow_lambda).view(-1, 1, 1,
                                                                                          1).expand_as(loss)
        del mask

    if random.random() < args.contrastive_flow_matching_loss_p:
        pathnames_resolved = [os.path.realpath(p)
                              for p in batch['pathnames']]
        mask = ~negative_loss_mask.cpu() & ~is_cond_dropout.cpu()
        loss_cfm, sample_count = get_contrastive_flow_matching_loss(
            model_forward_result.target,
            model_forward_result.model_pred,
            unique_identifiers=pathnames_resolved,
            loss_type=args.loss_type,
            timesteps=timesteps,
            noise_scheduler=model.noise_scheduler,
            mask=mask,
            amount=args.contrastive_flow_matching_loss_lambda
        )
        log_writer.add_scalar("loss/CF sample count", sample_count, global_step=tv.global_step)
        loss += loss_cfm.to(dtype=loss.dtype)

    log_data.loss_preview_image = torch.cat([model_forward_result.model_pred, model_forward_result.target, loss], dim=-2).detach().clone().cpu()

    # reduce from [B, C, H, W] to [B] with mean
    loss = loss.mean(dim=list(range(1, len(loss.shape))))

    if args.saturation_penalty_scale > 0:
        saturation_penalty = compute_saturation_penalty_loss(
            model_pred=model_forward_result.model_pred,
            noisy_latents=model_forward_result.noisy_latents,
            timesteps=timesteps,
            noise_scheduler=model.noise_scheduler,
            t_max=args.saturation_penalty_t_max,
        ).to(dtype=loss.dtype)
        log_writer.add_scalar("loss/saturation_penalty", saturation_penalty.mean().item(), global_step=tv.global_step)
        loss += saturation_penalty * args.saturation_penalty_scale

    # Self-Flow representation loss
    if (model_forward_result.self_flow_student_features is not None
            and model_forward_result.self_flow_teacher_features is not None
            and model.self_flow_proj_head is not None):
        l_rep = compute_self_flow_loss(
            student_features=model_forward_result.self_flow_student_features,
            teacher_features=model_forward_result.self_flow_teacher_features,
            proj_head=model.self_flow_proj_head,
        )
        log_writer.add_scalar("loss/self_flow_rep", l_rep.item(), global_step=tv.global_step)
        loss = loss + args.self_flow_gamma * l_rep

    return loss

def repeat_with_oom_handling(initial_slice_size: int, callback, oom_log_info: str):
    slice_size = initial_slice_size
    while True:
        try:
            return callback(slice_size)
        except (torch.cuda.OutOfMemoryError, torch.OutOfMemoryError):
            logging.error(
                oom_log_info + f": Slice size {slice_size} caused OOM."
            )
            if slice_size > 1:
                slice_size = slice_size // 2
                logging.info(f" -> Trying again with slice size {slice_size}")
                torch.cuda.empty_cache()
                gc.collect()
            else:
                logging.error(
                    " -> Already at slice size 1, cannot reduce further"
                )
                raise


