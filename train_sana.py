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
from utils.train_args import parse_train_args


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
    if not getattr(args, 'resume_from', None) and getattr(args, 'resume_ckpt', None):
        args.resume_from = args.resume_ckpt
    if not getattr(args, 'resume_ckpt', None):
        args.resume_ckpt = args.resume_from or ""

    # Map sample_steps → used by SampleGenerator as default_sample_steps
    # (shared arg is sample_steps; train_sana historically called it sample_every)
    if not hasattr(args, 'sample_every'):
        args.sample_every = args.sample_steps

    # Derive gradient-accumulation multiplier expected by choose_effective_batch_size()
    batch_size = args.batch_size
    optimizer_batch_size = args.optimizer_batch_size
    initial_bs = optimizer_batch_size if optimizer_batch_size is not None else batch_size
    final_bs = optimizer_batch_size if optimizer_batch_size is not None else batch_size
    args.initial_batch_size = initial_bs
    args.final_batch_size = final_bs
    args.dl_grad_accum = max(1, initial_bs // batch_size)
    args.grad_accum = args.dl_grad_accum

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
        image_output_range='[0,255]'
    )

    data_loader = build_torch_dataloader(
        dataset,
        batch_size=args.batch_size,
        num_workers=args.num_dataloader_workers or 4,
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

    args = parse_sana_args()

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


    sample_generator = SampleGenerator(
        log_folder=logdir,
        log_writer=log_writer,
        default_resolution=args.resolution[0],
        config_file_path=args.sample_prompts,
        batch_size=1,
        default_seed=args.seed,
        default_sample_steps=args.sample_steps,
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
