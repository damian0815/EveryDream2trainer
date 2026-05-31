"""
train_sana.py — SANA model training entry point for EveryDream2trainer.

Reuses:
  - run_accumulation_loop() from core/step.py for nibble/accumulation/backward/optimizer-step
  - EveryDreamBatch / DataLoaderMultiAspect for data loading (image_output_range="[0,255]")
  - EveryDreamOptimizer for parameter optimisation
  - utils/inference_context.py for eval/train guard during sample generation
"""
import argparse
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
from core.loss_sana import sample_sana_timesteps, compute_sana_loss
from model.sana_training_model import SanaTrainingModel, load_sana_model, save_sana_model
from model.sana_text_encoder import encode_sana_text, encode_sana_null_text
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


def parse_args() -> tuple[Namespace, object]:
    """
    Parses CLI args. --sana_config is required (path to a SanaConfig YAML).
    CLI flags override the corresponding SanaConfig fields after YAML loading.
    """
    import pyrallis
    from diffusion.utils.config import SanaConfig

    parser = argparse.ArgumentParser(description="Train SANA model with EveryDream2 infrastructure")
    parser.add_argument("--sana_config", type=str, required=True,
                        help="Path to SanaConfig YAML file")

    # Resolution
    parser.add_argument("--resolution", type=int, nargs="+", default=None,
                        help="One or more training resolutions (e.g. --resolution 512 1024). "
                             "Defaults to sana_config.model.image_size. "
                             "A separate dataset scan is performed per resolution.")
    parser.add_argument("--resolution_multiplier", type=float, nargs="+", default=[],
                        help="Per-resolution dataset multiplier. One value per --resolution entry, "
                             "or omit to treat all resolutions equally.")

    # Memory slicing
    parser.add_argument("--forward_slice_size", type=int, nargs="+", default=[],
                        help="Max images per forward-pass slice. One value per resolution (or one "
                             "value applied to all). Halved automatically on OOM.")
    parser.add_argument("--max_backward_slice_size", type=int, nargs="+", default=[],
                        help="Max images accumulated before a backward pass is fired. One value "
                             "per resolution (or one applied to all).")

    # Optional overrides
    parser.add_argument("--project_name", type=str, default=None)
    parser.add_argument("--logdir", type=str, default=None)
    parser.add_argument("--data_root", type=str, default=None)
    parser.add_argument("--batch_size", type=int, default=None,
                        help="Data loading batch size — controls aspect-ratio bucketing. "
                             "Each data batch contains this many images of the same aspect ratio.")
    parser.add_argument("--optimizer_batch_size", type=int, default=None,
                        help="Effective optimizer batch size (samples between optimizer steps). "
                             "When larger than --batch_size the trainer accumulates gradients across "
                             "multiple data batches before stepping. Defaults to --batch_size.")
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument("--max_steps", type=int, default=None)
    parser.add_argument("--grad_accum", type=int, default=None)
    parser.add_argument("--mixed_precision", type=str, default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--resume_from", type=str, default=None)
    parser.add_argument("--save_every", type=int, default=None)
    parser.add_argument("--sample_every", type=int, default=500)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--amp", action="store_true", default=True)
    parser.add_argument("--max_epochs", type=int, default=100)
    parser.add_argument("--skip_undersized_images", action="store_true", default=False,
                        help="Drop images smaller than the target resolution bucket.")

    cli_args = parser.parse_args()

    # Load the base SanaConfig from YAML
    with open(cli_args.sana_config) as f:
        sana_config = pyrallis.load(SanaConfig, f)

    # Apply CLI overrides to sana_config
    if cli_args.batch_size is not None:
        sana_config.train.train_batch_size = cli_args.batch_size
    if cli_args.lr is not None:
        sana_config.train.optimizer.lr = cli_args.lr
    if cli_args.max_steps is not None:
        sana_config.train.num_steps = cli_args.max_steps
    if cli_args.grad_accum is not None:
        sana_config.train.gradient_accumulation_steps = cli_args.grad_accum
    if cli_args.mixed_precision is not None:
        sana_config.model.mixed_precision = cli_args.mixed_precision
    if cli_args.seed is not None:
        sana_config.train.seed = cli_args.seed
    if cli_args.resume_from is not None:
        sana_config.model.resume_from = cli_args.resume_from
    if cli_args.save_every is not None:
        sana_config.train.save_model_steps = cli_args.save_every

    # Resolve resolutions: CLI wins, else YAML image_size
    resolutions = cli_args.resolution or [sana_config.model.image_size]

    # Validate & expand per-resolution lists
    if cli_args.resolution_multiplier and len(cli_args.resolution_multiplier) != len(resolutions):
        raise ValueError(
            f"--resolution_multiplier: pass one multiplier per --resolution entry "
            f"(got {len(cli_args.resolution_multiplier)} multipliers for {len(resolutions)} resolutions)"
        )
    forward_slice_size = _expand_to_per_resolution(
        cli_args.forward_slice_size, resolutions, "forward_slice_size"
    )
    max_backward_slice_size = _expand_to_per_resolution(
        cli_args.max_backward_slice_size, resolutions, "max_backward_slice_size"
    )

    batch_size = sana_config.train.train_batch_size

    # --optimizer_batch_size sets the effective batch size (samples between optimizer
    # steps).  It mirrors the train.py pattern: it sets initial_batch_size and
    # final_batch_size, both of which choose_effective_batch_size() reads.
    optimizer_batch_size = cli_args.optimizer_batch_size  # may be None
    initial_batch_size = optimizer_batch_size if optimizer_batch_size is not None else batch_size
    final_batch_size = optimizer_batch_size if optimizer_batch_size is not None else batch_size

    # grad_accum for the DataLoader: how many data batches make one effective step.
    # Passing this to DataLoaderMultiAspect lets it keep same-aspect images together
    # within each effective optimizer step window.
    dl_grad_accum = max(1, initial_batch_size // batch_size)

    logdir = cli_args.logdir or getattr(sana_config.train, 'output_dir', 'logs/sana')
    project_name = cli_args.project_name or os.path.basename(sana_config.work_dir)

    args = Namespace(
        sana_config=cli_args.sana_config,
        project_name=project_name,
        logdir=logdir,
        data_root=cli_args.data_root or sana_config.data.data_dir,
        batch_size=batch_size,
        optimizer_batch_size=optimizer_batch_size,
        dl_grad_accum=dl_grad_accum,
        resolution=resolutions,
        resolution_multiplier=cli_args.resolution_multiplier,
        forward_slice_size=forward_slice_size,
        max_backward_slice_size=max_backward_slice_size,
        skip_undersized_images=cli_args.skip_undersized_images,
        lr=sana_config.train.optimizer.lr,
        max_steps=sana_config.train.num_steps,
        grad_accum=sana_config.train.gradient_accumulation_steps,
        seed=sana_config.train.seed,
        save_every=getattr(sana_config.train, 'save_model_steps', 1000),
        sample_every=cli_args.sample_every,
        num_workers=cli_args.num_workers,
        amp=cli_args.amp,
        max_epochs=cli_args.max_epochs,
        # Effective batch size — read by choose_effective_batch_size()
        initial_batch_size=initial_batch_size,
        final_batch_size=final_batch_size,
        batch_size_curriculum_alpha=1.0,
        # Interleaved-BS1 curriculum (disabled for SANA)
        interleave_batch_size_1=False,
        interleave_batch_size_1_alpha=1.0,
    )

    return args, sana_config


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

    Mirrors the multi-resolution scan in train.py so that the same dataset
    folder appears once per resolution bucket — identical to SD/SDXL behaviour.
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

        # Resolve items for this resolution
        resolved = resolver_module.resolve(
            args.data_root, args, resolution=resolution, aspects=this_aspects
        )

        # Drop corrupt items
        for item in resolved:
            if item.error is not None:
                logging.error(f"Skipping corrupt image {item.pathname}: {item.error}")
        resolved = [item for item in resolved if item.error is None]

        # Apply undersized filter
        if args.skip_undersized_images:
            before = len(resolved)
            resolved = [item for item in resolved if not item.is_undersized]
            dropped = before - len(resolved)
            if dropped:
                logging.info(f"Resolution {resolution}: dropped {dropped} undersized images")

        # Apply per-resolution multiplier
        for item in resolved:
            item.multiplier *= multiplier

        logging.info(f"Resolution {resolution}: {len(resolved)} images")
        image_train_items.extend(resolved)

    if not image_train_items:
        raise RuntimeError(
            f"No training images found in '{args.data_root}'. "
            "Check --data_root and that your folder contains supported image files."
        )

    data_loader_multi_aspect = DataLoaderMultiAspect(image_train_items=image_train_items, seed=seed,
                                                     batch_size=args.batch_size)

    dataset = EveryDreamBatch(
        data_loader=data_loader_multi_aspect,
        tokenizer=None,               # SANA handles its own tokenization
        seed=seed,
        image_output_range="[0,255]",
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

def _encode_sana_latents(
    model: SanaTrainingModel,
    images: torch.Tensor,
    device: torch.device,
) -> torch.Tensor:
    """
    Encodes a batch of images (float32, values in [0, 255]) via SANA's DC-AE VAE.
    Returns latent tensors on the same device.
    """
    from diffusion.model.builder import vae_encode
    with torch.no_grad():
        return vae_encode(model.vae, images.to(device), model.vae_config.sample_posterior)


# ---------------------------------------------------------------------------
# Per-batch resolution bookkeeping
# ---------------------------------------------------------------------------

def _update_tv_for_batch(tv: TrainingVariables, full_batch: dict, args: Namespace) -> None:
    """
    Sets tv.batch_resolution, tv.forward_slice_size, and tv.max_backward_slice_size
    based on the actual pixel count of the current batch.  Must be called once per
    batch before train_sana_step().
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
    Handles per-batch SANA training: text encoding, VAE encoding, timestep sampling,
    then delegates nibbling/accumulation/backward/step to run_accumulation_loop().
    """
    # 1. Text-encode the full batch once (encoder is frozen)
    with torch.no_grad():
        y, y_mask = encode_sana_text(
            model.tokenizer,
            model.text_encoder,
            full_batch["captions"]["default"],
            model.sana_config,
            device,
        )

    # 2. VAE-encode the full batch once, with OOM retry
    z = repeat_with_oom_handling(
        initial_slice_size=tv.forward_slice_size,
        callback=lambda sz: _encode_sana_latents(model, full_batch["image"][:sz], device),
        oom_log_info=f"OOM gs:{tv.global_step} SANA VAE encode",
    )

    # 3. Sample timesteps for the full batch
    timesteps = sample_sana_timesteps(
        batch_size=full_batch["image"].shape[0],
        train_sampling_steps=model.sana_config.scheduler.train_sampling_steps,
        weighting_scheme=getattr(model.sana_config.scheduler, 'weighting_scheme', 'uniform'),
        logit_mean=getattr(model.sana_config.scheduler, 'logit_mean', 0.0),
        logit_std=getattr(model.sana_config.scheduler, 'logit_std', 1.0),
        device=device,
    )

    # 4. Build the nibble loss closure (closed over pre-encoded z, y, y_mask, timesteps)
    def nibble_loss_fn(nibble: dict) -> torch.Tensor:
        n = nibble["image"].shape[0]
        img_h = full_batch["image"].shape[2]
        img_w = full_batch["image"].shape[3]
        data_info = {
            "img_hw": nibble.get(
                "img_hw",
                torch.tensor([[img_h, img_w]] * n, dtype=torch.float),
            ),
            "aspect_ratio": nibble.get("aspect_ratio", torch.ones(n)),
        }
        return compute_sana_loss(model, z[:n], y[:n], y_mask[:n], timesteps[:n], data_info)

    # 5. Generic accumulation loop
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
) -> None:
    """
    Runs inference for each prompt and saves images to TensorBoard and to disk.
    """
    from PIL import Image as PILImage

    vis_sampler = getattr(model.sana_config.scheduler, 'vis_sampler', 'flow_euler')
    image_size = model.sana_config.model.image_size
    latent_dim = getattr(model.sana_config.vae, 'vae_latent_dim', 32)
    latent_size = image_size // getattr(model.sana_config.vae, 'vae_downsample_rate', 32)

    null_cache = os.path.join(output_dir, "null_embed_cache.pt")
    null_y = encode_sana_null_text(
        model.tokenizer, model.text_encoder, model.sana_config, device, cache_path=null_cache
    )

    samples_dir = os.path.join(output_dir, "samples-sana", f"gs{global_step}")
    os.makedirs(samples_dir, exist_ok=True)

    with inference_guard(model.transformer):
        for i, prompt in enumerate(sample_prompts):
            try:
                with torch.no_grad():
                    y, y_mask = encode_sana_text(
                        model.tokenizer, model.text_encoder, [prompt], model.sana_config, device
                    )
                    z_init = torch.randn(1, latent_dim, latent_size, latent_size, device=device)
                    hw = torch.tensor([[image_size, image_size]], dtype=torch.float, device=device)
                    ar = torch.tensor([[1.0]], device=device)

                    if vis_sampler in ('flow_euler', 'FlowEuler'):
                        from diffusion import FlowEuler
                        sampler = FlowEuler(
                            model.transformer,
                            condition=y,
                            uncondition=null_y,
                            cfg_scale=getattr(model.sana_config.scheduler, 'cfg_scale', 4.5),
                            model_kwargs=dict(
                                data_info={'img_hw': hw, 'aspect_ratio': ar}, mask=y_mask
                            ),
                        )
                        samples = sampler.sample(z_init)
                    else:
                        from diffusion import DPMS
                        sampler = DPMS(
                            model.transformer,
                            condition=y,
                            uncondition=null_y,
                            cfg_scale=getattr(model.sana_config.scheduler, 'cfg_scale', 4.5),
                            model_kwargs=dict(
                                data_info={'img_hw': hw, 'aspect_ratio': ar}, mask=y_mask
                            ),
                        )
                        samples = sampler.sample(
                            z_init,
                            steps=getattr(model.sana_config.scheduler, 'vis_sampling_steps', 20),
                        )

                    from diffusion.model.builder import vae_decode
                    pixel_values = vae_decode(
                        model.vae, samples / model.sana_config.vae.scale_factor
                    )
                    pixel_values = (pixel_values.clamp(-1, 1) + 1.0) / 2.0

                log_writer.add_image(
                    f"samples-sana/{prompt[:40]}",
                    pixel_values[0],
                    global_step=global_step,
                )

                img_np = (pixel_values[0].cpu().permute(1, 2, 0).numpy() * 255).astype('uint8')
                out_path = os.path.join(
                    samples_dir, f"{i:03d}_{prompt[:40].replace('/', '_')}.webp"
                )
                PILImage.fromarray(img_np).save(out_path, format="webp", quality=90)

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
    max_steps = args.max_steps

    # Initialise the effective batch size before the first step.
    # run_accumulation_loop() re-calls this after each optimizer step.
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
            if max_steps is not None and global_step >= max_steps:
                logging.info(f"Reached max_steps={max_steps}, stopping.")
                return

            tv.global_step = global_step

            # Set per-resolution slice sizes from the current batch dimensions
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

            # Periodic model save
            save_every = args.save_every
            if global_step > 0 and global_step % save_every == 0:
                save_path = os.path.join(logdir, f"gs{global_step}")
                logging.info(f"Saving SANA model to {save_path}")
                save_sana_model(save_path, model, global_step)

            # Periodic sample generation
            if global_step % args.sample_every == 0 and sample_prompts:
                logging.info(f"Generating samples at gs:{global_step}")
                generate_sana_samples(
                    model, global_step, log_writer, sample_prompts, logdir, device
                )
                gc.collect()
                torch.cuda.empty_cache()

            global_step += 1

    # Final save
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

    args, sana_config = parse_args()

    logdir = os.path.join(args.logdir, args.project_name)
    os.makedirs(logdir, exist_ok=True)
    log_writer = SummaryWriter(log_dir=logdir)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load model
    logging.info("Loading SANA model...")
    model = load_sana_model(args, sana_config)
    model.transformer.to(device)
    model.text_encoder.to(device)
    model.vae.to(device)

    # Optimizer — only the transformer is trained
    ed_optimizer = EveryDreamOptimizer(
        args=args,
        optimizer_config=None,
        text_encoder=None,
        unet=model.transformer,
        global_step=0,
    )

    # Training variables (builds per-resolution slice-size maps)
    tv = setup_sana_training_variables(args)

    # Data loader
    logging.info(f"Building data loader for resolutions: {args.resolution}")
    dataset, data_loader = build_sana_data_loader(args, seed=sana_config.train.seed)

    # Sample prompts
    sample_prompts = []
    prompts_file = os.path.join(os.path.dirname(args.sana_config), "sample_prompts.txt")
    if os.path.exists(prompts_file):
        with open(prompts_file) as f:
            sample_prompts = [line.strip() for line in f if line.strip()]

    logging.info(
        f"Starting SANA training — logdir={logdir}, "
        f"resolutions={args.resolution}, batch_size={args.batch_size}"
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






