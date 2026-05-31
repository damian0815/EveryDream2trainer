"""
train_sana.py — SANA model training entry point for EveryDream2trainer.

Uses 🤗 diffusers for all model components (SanaPipeline, SanaTransformer2DModel,
AutoencoderDC, FlowMatchEulerDiscreteScheduler).  No SANA repo clone required.

Reuses:
  - run_accumulation_loop() from core/step.py  (nibble/accumulation/backward/optimizer step)
  - EveryDreamBatch / DataLoaderMultiAspect    (data loading, aspect-ratio bucketing)
  - EveryDreamOptimizer                        (AdamW, CAME, Prodigy, ...)
  - utils/inference_context.py                 (eval/train guard during sample generation)
  - SampleGenerator                            (sample generation, TensorBoard logging)
"""
import argparse
import gc
import logging
import os
from argparse import Namespace
import torch
from torch.utils.tensorboard import SummaryWriter

import data.aspects as aspects_module
import data.resolver as resolver_module
from core.step import run_accumulation_loop, repeat_with_oom_handling
from core.step import get_best_match_resolution, choose_effective_batch_size
from core.loss import get_multirank_stratified_random_timesteps
from core.loss_sana import compute_sana_loss
from core.flow_match_model import TrainFlowMatchEulerDiscreteScheduler
from model.sana_training_model import SanaTrainingModel, load_sana_model, save_sana_model
from model.sana_text_encoder import encode_prompts
from model.training_model import TrainingVariables
from optimizer.optimizers import EveryDreamOptimizer
from utils.inference_context import inference_guard
from utils.sample_generator import SampleGenerator


# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------

def _expand_to_per_resolution(values: list, resolutions: list, name: str) -> list:
    """
    Expand a 1-element list to one entry per resolution, or validate that a
    multi-element list has exactly one entry per resolution.
    Returns an empty list unchanged (meaning "use defaults").
    """
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


def parse_args() -> Namespace:
    """
    Parses CLI args.  --model_id and --data_root are the only required flags.
    """
    parser = argparse.ArgumentParser(description="Train SANA model with EveryDream2 infrastructure")

    # Model
    parser.add_argument("--model_id", type=str, required=True,
                        help="HuggingFace hub model ID, e.g. "
                             "Efficient-Large-Model/SANA1.5_1.6B_1024px_diffusers")
    parser.add_argument("--resume_from", type=str, default=None,
                        help="Path to a saved transformer .safetensors checkpoint to resume from")

    # Project / logging
    parser.add_argument("--project_name", type=str, default="sana-finetune")
    parser.add_argument("--logdir", type=str, default="logs")

    # Data
    parser.add_argument("--data_root", type=str, required=True,
                        help="Root directory of the training dataset")
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--skip_undersized_images", action="store_true", default=False,
                        help="Drop images smaller than the target resolution bucket")

    # Resolution
    parser.add_argument("--resolution", type=int, nargs="+", default=[1024],
                        help="One or more training resolutions (e.g. --resolution 512 1024). "
                             "A separate dataset scan is performed per resolution.")
    parser.add_argument("--resolution_multiplier", type=float, nargs="+", default=[],
                        help="Per-resolution dataset multiplier. One value per --resolution entry, "
                             "or omit to treat all resolutions equally.")

    # Memory slicing
    parser.add_argument("--forward_slice_size", type=int, nargs="+", default=[],
                        help="Max images per forward-pass slice. One value per resolution "
                             "(or one value applied to all). Halved automatically on OOM.")
    parser.add_argument("--max_backward_slice_size", type=int, nargs="+", default=[],
                        help="Max images accumulated before a backward pass fires. "
                             "One value per resolution (or one applied to all).")

    # Batch sizes
    parser.add_argument("--batch_size", type=int, default=2,
                        help="Data loading batch size — controls aspect-ratio bucketing.")
    parser.add_argument("--optimizer_batch_size", type=int, default=None,
                        help="Effective optimizer batch size (samples between optimizer steps). "
                             "When larger than --batch_size, gradients are accumulated across "
                             "multiple data batches. Defaults to --batch_size.")

    # Training
    parser.add_argument("--lr", type=float, default=2e-5)
    parser.add_argument("--max_steps", type=int, default=10000)
    parser.add_argument("--mixed_precision", type=str, default="bf16",
                        choices=["bf16", "fp16", "no"])
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--amp", action="store_true", default=True)
    parser.add_argument("--max_epochs", type=int, default=100)

    # Checkpointing
    parser.add_argument("--save_every", type=int, default=1000,
                        help="Save transformer weights every N optimizer steps")

    # Optimizer config
    parser.add_argument("--optimizer_config", type=str, default="optimizer.json",
                        help="Path to a JSON optimizer config file (default: optimizer.json)")

    # Sample generation
    parser.add_argument("--sample_every", type=int, default=500,
                        help="Generate validation samples every N optimizer steps (or load from sample_prompts.json)")
    parser.add_argument("--sample_prompts", type=str, default="sample_prompts.json",
                        help="Path to a sample-prompts .json or .txt file for SampleGenerator")

    # Text encoding
    parser.add_argument("--max_sequence_length", type=int, default=300,
                        help="Gemma token budget")

    # Timestep sampling — mirrors train.py exactly
    parser.add_argument("--timesteps_multirank_stratified",
                        action=argparse.BooleanOptionalAction, default=True,
                        help="Use multirank stratified timestep sampling")
    parser.add_argument("--timesteps_multirank_stratified_distribution",
                        type=str,
                        choices=["uniform", "beta", "mode", "boundary-oversampling", "lognormal"],
                        default="lognormal",
                        help="Timestep distribution. For 'lognormal', --alpha controls the std "
                             "of the underlying normal (width of the distribution).")
    parser.add_argument("--timesteps_multirank_stratified_stratify",
                        action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--timesteps_multirank_stratified_alpha", type=float, default=1.0,
                        help="Alpha/std parameter. For lognormal: std of the normal before sigmoid.")
    parser.add_argument("--timesteps_multirank_stratified_beta", type=float, default=1.6)
    parser.add_argument("--timesteps_multirank_stratified_mode_scale", type=float, default=0.5)

    cli = parser.parse_args()

    # Validate & expand per-resolution lists
    resolutions = cli.resolution
    if cli.resolution_multiplier and len(cli.resolution_multiplier) != len(resolutions):
        raise ValueError(
            f"--resolution_multiplier: pass one multiplier per --resolution entry "
            f"(got {len(cli.resolution_multiplier)} for {len(resolutions)} resolutions)"
        )
    forward_slice_size = _expand_to_per_resolution(
        cli.forward_slice_size, resolutions, "forward_slice_size"
    )
    max_backward_slice_size = _expand_to_per_resolution(
        cli.max_backward_slice_size, resolutions, "max_backward_slice_size"
    )

    batch_size = cli.batch_size
    optimizer_batch_size = cli.optimizer_batch_size
    initial_batch_size = optimizer_batch_size if optimizer_batch_size is not None else batch_size
    final_batch_size = optimizer_batch_size if optimizer_batch_size is not None else batch_size
    dl_grad_accum = max(1, initial_batch_size // batch_size)

    return Namespace(
        model_id=cli.model_id,
        resume_from=cli.resume_from,
        project_name=cli.project_name,
        logdir=cli.logdir,
        data_root=cli.data_root,
        num_workers=cli.num_workers,
        skip_undersized_images=cli.skip_undersized_images,
        resolution=resolutions,
        resolution_multiplier=cli.resolution_multiplier,
        forward_slice_size=forward_slice_size,
        max_backward_slice_size=max_backward_slice_size,
        batch_size=batch_size,
        optimizer_batch_size=optimizer_batch_size,
        dl_grad_accum=dl_grad_accum,
        # Fields read by choose_effective_batch_size():
        initial_batch_size=initial_batch_size,
        final_batch_size=final_batch_size,
        batch_size_curriculum_alpha=1.0,
        interleave_batch_size_1=False,
        interleave_batch_size_1_alpha=1.0,
        lr=cli.lr,
        max_steps=cli.max_steps,
        mixed_precision=cli.mixed_precision,
        seed=cli.seed,
        amp=cli.amp,
        max_epochs=cli.max_epochs,
        save_every=cli.save_every,
        sample_every=cli.sample_every,
        sample_prompts=cli.sample_prompts,
        max_sequence_length=cli.max_sequence_length,
        # ---- EveryDreamOptimizer required args ----
        optimizer_config=cli.optimizer_config,
        grad_accum=dl_grad_accum,
        clip_grad_norm=None,
        disable_unet_training=False,
        disable_textenc_training=True,    # SANA text encoder is always frozen
        lora=False,
        amp_without_grad_scaler=True,
        debug_unet_freeze_regex=False,
        unet_freeze_regex=None,
        optimizer_param_grouping=["single"],
        lr_decay_steps=0,
        lr_scheduler="cosine",
        lr_warmup_steps=None,
        lr_advance_steps=None,
        lr_end=None,
        lr_num_restarts=1,
        auto_decay_steps_multiplier=1.1,
        resume_ckpt=cli.resume_from or "",
        optimizer_progressive_unlock=False,
        optimizer_progressive_unlock_by_qk_proximity=False,
        init_grad_scale=None,
        # -------------------------------------------
        timesteps_multirank_stratified=cli.timesteps_multirank_stratified,
        timesteps_multirank_stratified_distribution=cli.timesteps_multirank_stratified_distribution,
        timesteps_multirank_stratified_stratify=cli.timesteps_multirank_stratified_stratify,
        timesteps_multirank_stratified_alpha=cli.timesteps_multirank_stratified_alpha,
        timesteps_multirank_stratified_beta=cli.timesteps_multirank_stratified_beta,
        timesteps_multirank_stratified_mode_scale=cli.timesteps_multirank_stratified_mode_scale,
    )


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

def build_sana_data_loader(args: Namespace, seed: int):
    """
    Resolves ImageTrainItems for every resolution in args.resolution, merges
    them, and returns a (EveryDreamBatch, torch DataLoader) pair.

    Images are returned in [-1, 1] range — the correct input range for
    diffusers AutoencoderDC.
    """
    from data.data_loader import DataLoaderMultiAspect
    from data.every_dream import EveryDreamBatch, build_torch_dataloader
    from data.image_train_item import ImageTrainItem

    image_train_items: list[ImageTrainItem] = []

    for res_index, resolution in enumerate(args.resolution):
        this_aspects = aspects_module.get_aspect_buckets(resolution)
        multiplier = (
            args.resolution_multiplier[res_index]
            if args.resolution_multiplier else 1.0
        )
        if multiplier != 1.0:
            logging.info(f"SANA data: resolution {resolution}, multiplier {multiplier}")

        resolved = resolver_module.resolve(
            args.data_root, args, resolution=resolution, aspects=this_aspects
        )

        for item in resolved:
            if item.error is not None:
                logging.error(f"Skipping corrupt image {item.pathname}: {item.error}")
        resolved = [item for item in resolved if item.error is None]

        if args.skip_undersized_images:
            before = len(resolved)
            resolved = [item for item in resolved if not item.is_undersized]
            dropped = before - len(resolved)
            if dropped:
                logging.info(f"Resolution {resolution}: dropped {dropped} undersized images")

        for item in resolved:
            item.multiplier *= multiplier

        logging.info(f"Resolution {resolution}: {len(resolved)} images")
        image_train_items.extend(resolved)

    if not image_train_items:
        raise RuntimeError(
            f"No training images found in '{args.data_root}'. "
            "Check --data_root and that your folder contains supported image files."
        )

    data_loader_multi_aspect = DataLoaderMultiAspect(
        image_train_items=image_train_items,
        seed=seed,
        batch_size=args.batch_size,
    )

    # Default image_output_range is [-1, 1], matching diffusers AutoencoderDC convention
    dataset = EveryDreamBatch(
        data_loader=data_loader_multi_aspect,
        tokenizer=None,
        seed=seed,
    )

    data_loader = build_torch_dataloader(
        dataset,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
    )

    return dataset, data_loader


# ---------------------------------------------------------------------------
# VAE encoding helper
# ---------------------------------------------------------------------------

def _encode_latents(
    model: SanaTrainingModel,
    images: torch.Tensor,
    device: torch.device,
) -> torch.Tensor:
    """
    Encodes a batch of images (float, range [-1, 1]) via DC-AE.
    Returns scaled latents: encode(x).latent * vae.config.scaling_factor.
    AutoencoderDC is deterministic (no .latent_dist); use .latent directly.
    """
    with torch.no_grad():
        latents = model.vae.encode(images.to(device)).latent
    return latents * model.vae.config.scaling_factor


# ---------------------------------------------------------------------------
# Per-batch resolution bookkeeping
# ---------------------------------------------------------------------------

def _update_tv_for_batch(tv: TrainingVariables, full_batch: dict, args: Namespace) -> None:
    """
    Sets tv.batch_resolution, tv.forward_slice_size, and tv.max_backward_slice_size
    from the actual pixel count of the current batch.  Must be called once per batch
    before train_sana_step().
    """
    image_pixel_count = full_batch["image"].shape[2] * full_batch["image"].shape[3]
    tv.batch_resolution = get_best_match_resolution(args.resolution, image_pixel_count)
    tv.forward_slice_size = tv.default_forward_slice_size[tv.batch_resolution]
    tv.max_backward_slice_size = tv.default_max_backward_slice_size[tv.batch_resolution]


# ---------------------------------------------------------------------------
# Per-step training
# ---------------------------------------------------------------------------

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


def train_sana_step(
    full_batch: dict,
    model: SanaTrainingModel,
    tv: TrainingVariables,
    ed_optimizer: EveryDreamOptimizer,
    log_writer: SummaryWriter,
    steps_pbar,
    device: torch.device,
    args: Namespace,
) -> None:
    """
    Handles per-batch SANA training: text encoding, VAE encoding, timestep sampling,
    then delegates nibbling/accumulation/backward/step to run_accumulation_loop().
    """
    # 1. Text-encode the full batch once (encoder is frozen)
    with torch.no_grad():
        y, y_mask = encode_prompts(
            model.tokenizer,
            model.text_encoder,
            full_batch["captions"]["default"],
            device,
            max_sequence_length=model.max_sequence_length,
            complex_human_instruction=model.complex_human_instruction or None,
            dtype=model.dtype,
        )

    # 2. VAE-encode the full batch once, with OOM retry
    z = repeat_with_oom_handling(
        initial_slice_size=tv.forward_slice_size,
        callback=lambda sz: _encode_latents(model, full_batch["image"][:sz], device),
        oom_log_info=f"OOM gs:{tv.global_step} SANA VAE encode",
    )

    # 3. Sample stratified flow-matching timesteps for the full batch
    full_batch_size = full_batch["image"].shape[0]
    timesteps = _draw_stratified_timesteps(full_batch_size, tv, model, args, device)

    # 4. Build the nibble loss closure (closed over pre-encoded z, y, y_mask, timesteps)
    def nibble_loss_fn(nibble: dict) -> torch.Tensor:
        n = nibble["image"].shape[0]
        return compute_sana_loss(
            model.transformer,
            model.noise_scheduler,
            z[:n],
            y[:n],
            y_mask[:n],
            timesteps[:n],
        )

    # 5. Generic accumulation loop (handles nibbling, OOM, backward, optimizer.step)
    run_accumulation_loop(
        full_batch=full_batch,
        tv=tv,
        ed_optimizer=ed_optimizer,
        nibble_loss_fn=nibble_loss_fn,
        plugin_runner=None,
        log_writer=log_writer,
        steps_pbar=steps_pbar,
        did_step_optimizer_cb=None,
        args=args,
    )


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
) -> None:
    """
    Outer training loop: epochs → batches, with periodic save and sample generation.
    """
    import tqdm

    global_step = 0

    # Initialise the effective batch size before the first step.
    # run_accumulation_loop() re-evaluates this after each optimizer step.
    tv.desired_effective_batch_size = choose_effective_batch_size(args, 0)

    logging.info(
        f"Effective optimizer batch size: {tv.desired_effective_batch_size} images "
        f"(data batch_size={args.batch_size}, "
        f"grad_accum_window={args.dl_grad_accum})"
    )

    for epoch in range(args.max_epochs):
        # Reset stratified timestep buffer at each epoch boundary (matches train.py)
        tv.remaining_stratified_timesteps = None

        dataset.shuffle(epoch_n=epoch, max_epochs=args.max_epochs)
        steps_pbar = tqdm.tqdm(data_loader, desc=f"Epoch {epoch}")

        for full_batch in steps_pbar:
            if global_step >= args.max_steps:
                logging.info(f"Reached max_steps={args.max_steps}, stopping.")
                return

            tv.global_step = global_step

            _update_tv_for_batch(tv, full_batch, args)

            train_sana_step(
                full_batch=full_batch,
                model=model,
                tv=tv,
                ed_optimizer=ed_optimizer,
                log_writer=log_writer,
                steps_pbar=steps_pbar,
                device=device,
                args=args,
            )

            if global_step > 0 and global_step % args.save_every == 0:
                save_path = os.path.join(logdir, f"gs{global_step}")
                logging.info(f"Saving SANA model to {save_path}")
                save_sana_model(save_path, model, global_step)

            if sample_generator.should_generate_samples(global_step, local_step=0):
                logging.info(f"Generating samples at gs:{global_step}")
                with inference_guard(model.transformer):
                    pipe = sample_generator.create_inference_pipe(
                        model_being_trained=model,
                        diffusers_scheduler_config=model.noise_scheduler.config,
                    ).to(device)
                    sample_generator.generate_samples(pipe, global_step)
                    del pipe
                gc.collect()
                torch.cuda.empty_cache()

            global_step += 1

    save_sana_model(os.path.join(logdir, "final"), model, global_step)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main_sana() -> None:
    """
    Main SANA training entry point. Wires together argument parsing, model
    loading, data pipeline, optimiser, training variables, logging, and the
    training loop.
    """
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    args = parse_args()

    logdir = os.path.join(args.logdir, args.project_name)
    os.makedirs(logdir, exist_ok=True)
    log_writer = SummaryWriter(log_dir=logdir)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    logging.info(f"Loading SANA model from {args.model_id}...")
    model = load_sana_model(args)
    model.transformer.to(device)
    model.text_encoder.to(device)
    model.vae.to(device)

    # Only the transformer is trained
    import json
    from plugins.plugins import PluginRunner
    optimizer_config_path = (
        args.optimizer_config
        if os.path.isabs(args.optimizer_config)
        else os.path.join(os.path.dirname(os.path.abspath(__file__)), args.optimizer_config)
    )
    if not os.path.exists(optimizer_config_path):
        raise FileNotFoundError(
            f"Optimizer config not found: {optimizer_config_path}. "
            "Pass --optimizer_config pointing to a valid JSON file (e.g. optimizer.json)."
        )
    with open(optimizer_config_path) as f:
        optimizer_config = json.load(f)

    logging.info(f"Building data loader for resolutions: {args.resolution}")
    dataset, data_loader = build_sana_data_loader(args, seed=args.seed)
    epoch_len = len(data_loader)

    ed_optimizer = EveryDreamOptimizer(
        args=args,
        optimizer_config=optimizer_config,
        model=model,
        epoch_len=epoch_len,
        plugin_runner=PluginRunner(),
        log_writer=log_writer,
    )

    tv = setup_sana_training_variables(args)


    sample_prompts_path = (
        args.sample_prompts
        if os.path.isabs(args.sample_prompts)
        else os.path.join(os.path.dirname(os.path.abspath(__file__)), args.sample_prompts)
    )
    sample_generator = SampleGenerator(
        log_folder=logdir,
        log_writer=log_writer,
        default_resolution=args.resolution[0],
        config_file_path=sample_prompts_path if os.path.exists(sample_prompts_path) else None,
        batch_size=1,
        default_seed=args.seed,
        default_sample_steps=args.sample_every,
    )

    logging.info(
        f"Starting SANA training — logdir={logdir}, "
        f"model={args.model_id}, "
        f"resolutions={args.resolution}, "
        f"batch_size={args.batch_size}"
    )

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
        logdir=logdir,
    )

    log_writer.close()
    logging.info("SANA training complete.")


if __name__ == "__main__":
    main_sana()
