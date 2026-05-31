"""
train_sana.py — SANA model training entry point for EveryDream2trainer.

Uses 🤗 diffusers for all model components (SanaPipeline, SanaTransformer2DModel,
AutoencoderDC, FlowMatchEulerDiscreteScheduler).  No SANA repo clone required.

Reuses:
  - run_accumulation_loop() from core/step.py  (nibble/accumulation/backward/optimizer step)
  - EveryDreamBatch / DataLoaderMultiAspect    (data loading, aspect-ratio bucketing)
  - EveryDreamOptimizer                        (AdamW, CAME, Prodigy, ...)
  - utils/inference_context.py                 (eval/train guard during sample generation)
"""
import argparse
import copy
import gc
import logging
import os
from argparse import Namespace
from typing import Optional

import torch
from torch.utils.tensorboard import SummaryWriter

import data.aspects as aspects_module
import data.resolver as resolver_module
from core.step import run_accumulation_loop, repeat_with_oom_handling
from core.step import get_best_match_resolution, choose_effective_batch_size
from core.loss_sana import sample_flow_sigmas, compute_sana_loss
from model.sana_training_model import SanaTrainingModel, load_sana_model, save_sana_model
from model.sana_text_encoder import encode_prompts, encode_null_prompt
from model.training_model import TrainingVariables
from optimizer.optimizers import EveryDreamOptimizer
from utils.inference_context import inference_guard


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

    # Sample generation
    parser.add_argument("--sample_every", type=int, default=500,
                        help="Generate validation samples every N optimizer steps")
    parser.add_argument("--guidance_scale", type=float, default=4.5,
                        help="Classifier-free guidance scale for sample generation")
    parser.add_argument("--sample_height", type=int, default=None,
                        help="Height for generated samples (defaults to first --resolution)")
    parser.add_argument("--sample_width", type=int, default=None,
                        help="Width for generated samples (defaults to first --resolution)")

    # Text encoding
    parser.add_argument("--max_sequence_length", type=int, default=300,
                        help="Gemma token budget")

    # Flow-matching scheduler
    parser.add_argument("--num_train_timesteps", type=int, default=1000)
    parser.add_argument("--weighting_scheme", type=str, default="logit_normal",
                        choices=["uniform", "logit_normal"])
    parser.add_argument("--logit_mean", type=float, default=0.0)
    parser.add_argument("--logit_std", type=float, default=1.0)

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
        guidance_scale=cli.guidance_scale,
        sample_height=cli.sample_height or resolutions[0],
        sample_width=cli.sample_width or resolutions[0],
        max_sequence_length=cli.max_sequence_length,
        num_train_timesteps=cli.num_train_timesteps,
        weighting_scheme=cli.weighting_scheme,
        logit_mean=cli.logit_mean,
        logit_std=cli.logit_std,
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
    Handles per-batch SANA training: text encoding, VAE encoding, sigma sampling,
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

    # 3. Sample flow-matching sigmas and timesteps for the full batch
    sigma, timestep_t = sample_flow_sigmas(
        batch_size=full_batch["image"].shape[0],
        weighting_scheme=args.weighting_scheme,
        logit_mean=args.logit_mean,
        logit_std=args.logit_std,
        device=device,
        num_train_timesteps=args.num_train_timesteps,
    )

    # 4. Build the nibble loss closure (closed over pre-encoded z, y, y_mask, sigma, timestep_t)
    def nibble_loss_fn(nibble: dict) -> torch.Tensor:
        n = nibble["image"].shape[0]
        return compute_sana_loss(
            model.transformer,
            z[:n],
            y[:n],
            y_mask[:n],
            sigma[:n],
            timestep_t[:n],
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
# Sample generation
# ---------------------------------------------------------------------------

def generate_sana_samples(
    model: SanaTrainingModel,
    global_step: int,
    log_writer: SummaryWriter,
    sample_prompts: list[str],
    output_dir: str,
    device: torch.device,
    args: Namespace,
) -> None:
    """
    Runs inference for each prompt via a temporary SanaPipeline constructed from
    the live model components, then saves images to TensorBoard and disk.
    """
    import numpy as np
    from diffusers import SanaPipeline
    from PIL import Image as PILImage

    samples_dir = os.path.join(output_dir, "samples-sana", f"gs{global_step}")
    os.makedirs(samples_dir, exist_ok=True)

    with inference_guard(model.transformer):
        # Build a pipeline from the live components.
        # Deepcopy the scheduler so the training scheduler's state is not mutated.
        pipe = SanaPipeline(
            transformer=model.transformer,
            text_encoder=model.text_encoder,
            tokenizer=model.tokenizer,
            vae=model.vae,
            scheduler=copy.deepcopy(model.scheduler),
        )
        pipe.to(device)

        for i, prompt in enumerate(sample_prompts):
            try:
                with torch.no_grad():
                    result = pipe(
                        prompt=prompt,
                        height=args.sample_height,
                        width=args.sample_width,
                        guidance_scale=args.guidance_scale,
                        num_inference_steps=20,
                    )
                image: PILImage.Image = result.images[0]

                img_tensor = torch.from_numpy(
                    np.array(image).astype("float32") / 255.0
                ).permute(2, 0, 1)
                log_writer.add_image(
                    f"samples-sana/{prompt[:40]}",
                    img_tensor,
                    global_step=global_step,
                )

                out_path = os.path.join(
                    samples_dir, f"{i:03d}_{prompt[:40].replace('/', '_')}.webp"
                )
                image.save(out_path, format="webp", quality=90)

            except Exception as e:
                logging.warning(f"generate_sana_samples: failed for '{prompt[:40]}': {e}")


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
    sample_prompts: list[str],
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

            if global_step % args.sample_every == 0 and sample_prompts:
                logging.info(f"Generating samples at gs:{global_step}")
                generate_sana_samples(
                    model, global_step, log_writer, sample_prompts, logdir, device, args
                )
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
    ed_optimizer = EveryDreamOptimizer(
        args=args,
        optimizer_config=None,
        text_encoder=None,
        unet=model.transformer,
        global_step=0,
    )

    tv = setup_sana_training_variables(args)

    logging.info(f"Building data loader for resolutions: {args.resolution}")
    dataset, data_loader = build_sana_data_loader(args, seed=args.seed)

    sample_prompts = []
    prompts_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "sample_prompts.txt")
    if os.path.exists(prompts_path):
        with open(prompts_path) as f:
            sample_prompts = [line.strip() for line in f if line.strip()]

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
        sample_prompts=sample_prompts,
        logdir=logdir,
    )

    log_writer.close()
    logging.info("SANA training complete.")


if __name__ == "__main__":
    main_sana()
