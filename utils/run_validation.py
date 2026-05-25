#!/usr/bin/env python3
"""
Standalone CLI tool to run EveryDreamValidator against an already-trained
(or in-progress) model checkpoint and write the collected metrics to JSON
or print them to the console.

Because no active TensorBoard process is required, all SummaryWriter calls
are intercepted by a lightweight shim that stores values in plain Python
dicts.

Usage examples
--------------
# Print to console
python utils/run_validation.py \
    --model path/to/model \
    --val_config validation_default.json \
    --resolution 512

# Write to a JSON file
python utils/run_validation.py \
    --model path/to/model \
    --val_config validation_default.json \
    --resolution 512 \
    --out metrics.json

# SDXL model, custom batch size
python utils/run_validation.py \
    --model path/to/sdxl_model \
    --val_config validation_default.json \
    --resolution 1024 \
    --batch_size 4 \
    --out sdxl_metrics.json
"""

import argparse
import json
import logging
import os
import sys
from collections import defaultdict
from typing import Optional

import torch
import numpy as np

# ---------------------------------------------------------------------------
# Shim SummaryWriter
# ---------------------------------------------------------------------------

class JsonSummaryWriter:
    """
    Drop-in shim for ``torch.utils.tensorboard.SummaryWriter`` that stores all
    logged values in memory as plain Python objects and never touches disk or
    a TensorBoard binary format.

    Collected data is accessible via:
      * ``writer.scalars``   – dict[tag, list[{"step": int, "value": float}]]
      * ``writer.histograms`` – dict[tag, list[{"step": int, "values": list[float]}]]
    """

    def __init__(self):
        self.scalars: dict[str, list[dict]] = defaultdict(list)
        self.histograms: dict[str, list[dict]] = defaultdict(list)

    # ---- SummaryWriter interface (only the subset used by the validator) ----

    def add_scalar(self, tag: str, scalar_value: float, global_step: int = 0, **_):
        self.scalars[tag].append({"step": global_step, "value": float(scalar_value)})

    def add_histogram(self, tag: str, values, global_step: int = 0, **_):
        if isinstance(values, torch.Tensor):
            values = values.detach().cpu().tolist()
        elif isinstance(values, np.ndarray):
            values = values.tolist()
        self.histograms[tag].append({"step": global_step, "values": list(values)})

    # Stubs so nothing crashes if the validator ever calls these
    def add_image(self, *args, **kwargs): pass
    def add_images(self, *args, **kwargs): pass
    def add_text(self, *args, **kwargs): pass
    def flush(self): pass
    def close(self): pass

    # ---- Serialisation ----

    def to_dict(self) -> dict:
        """Return all collected metrics as a plain dict suitable for JSON."""
        return {
            "scalars": dict(self.scalars),
            "histograms": dict(self.histograms),
        }

    def print_summary(self):
        """Pretty-print the most recent scalar value for every tracked tag."""
        if not self.scalars:
            print("(no scalar metrics collected)")
            return
        max_tag_len = max(len(t) for t in self.scalars)
        print(f"\n{'Tag':<{max_tag_len}}  {'Latest value':>18}  {'Step':>8}")
        print("-" * (max_tag_len + 30))
        for tag in sorted(self.scalars):
            latest = self.scalars[tag][-1]
            print(f"{tag:<{max_tag_len}}  {latest['value']:>18.6f}  {latest['step']:>8}")
        if self.histograms:
            print(f"\nHistograms logged: {', '.join(sorted(self.histograms))}")


# ---------------------------------------------------------------------------
# Model loader
# ---------------------------------------------------------------------------

def _load_model(model_path: str, device: torch.device):
    """
    Load a Stable Diffusion or SDXL checkpoint (diffusers folder or .safetensors)
    and return a ``TrainingModel`` instance.
    """
    from diffusers import StableDiffusionPipeline, StableDiffusionXLPipeline
    from model.training_model import TrainingModel, convert_to_hf

    model_path, _, yaml = convert_to_hf(model_path)

    logging.info(f"Loading model from {model_path} …")

    # Try SDXL first; fall back to SD1/2
    try:
        pipe = StableDiffusionXLPipeline.from_pretrained(
            model_path,
            torch_dtype=torch.float16,
            variant="fp16",
        )
        logging.info("Loaded as Stable Diffusion XL model.")
    except Exception:
        try:
            pipe = StableDiffusionXLPipeline.from_pretrained(
                model_path,
                torch_dtype=torch.float16,
            )
            logging.info("Loaded as Stable Diffusion XL model (no fp16 variant).")
        except Exception:
            pipe = StableDiffusionPipeline.from_pretrained(
                model_path,
                torch_dtype=torch.float16,
                safety_checker=None,
            )
            logging.info("Loaded as Stable Diffusion 1/2 model.")

    pipe = pipe.to(device)
    model = TrainingModel.from_pipeline(pipe, yaml=yaml)
    return model


# ---------------------------------------------------------------------------
# Validation callable
# ---------------------------------------------------------------------------

def _make_validation_callable(model, device, timestep_start: int = 0, timestep_end: int = 1000):
    """
    Build a ``get_model_prediction_and_target`` callable compatible with
    ``EveryDreamValidator.do_validation``.

    Mirrors the wrapper built inside ``train.py`` but without the full
    training-time argument namespace dependency.
    """
    from core.loss import get_model_prediction_and_target as _core_pred, get_noise
    from core.step import get_uniform_timesteps
    from data.every_dream_validation import ValidationStepResult
    from model.training_model import Conditioning

    num_timesteps = model.noise_scheduler.config.num_train_timesteps

    # clamp to valid scheduler range
    t_start = max(0, min(timestep_start, num_timesteps - 1))
    t_end   = max(t_start + 1, min(timestep_end, num_timesteps))

    @torch.no_grad()
    def _callable(image: torch.Tensor, conditioning: Conditioning) -> ValidationStepResult:
        batch_size = image.shape[0]
        timesteps = get_uniform_timesteps(
            batch_size=batch_size,
            batch_share_timesteps=False,
            device=device,
            timesteps_ranges=[(t_start, t_end)] * batch_size,
        )

        # get latents
        from core.loss import encode_with_vae_to_scaled_latents
        from types import SimpleNamespace
        # amp=False: no autocast during standalone eval (safe on any hardware).
        # latents_perturbation=0: referenced unconditionally inside core/loss.py.
        dummy_args = SimpleNamespace(amp=False, latents_perturbation=0.0)
        latents = encode_with_vae_to_scaled_latents(image, model, device=device, args=dummy_args)

        noise = get_noise(
            latents.shape, device, image.dtype,
            pyramid_noise_discount=0.0,
            zero_frequency_noise_ratio=0.0,
            batch_share_noise=False,
        )

        result = _core_pred(
            latents,
            conditioning,
            noise,
            timesteps,
            model=model,
            args=None,
            skip_contrastive=True,
        )

        return ValidationStepResult(
            model_pred=result.model_pred,
            target=result.target,
            timesteps=timesteps,
            noisy_latents=result.noisy_latents,
        )

    return _callable


# ---------------------------------------------------------------------------
# Pipe factory (only needed when fdd or anomaly is enabled)
# ---------------------------------------------------------------------------

def _make_pipe_factory(model, device):
    from diffusers import DDIMScheduler

    def factory():
        pipe = model.build_inference_pipeline()
        pipe.scheduler = DDIMScheduler.from_config(pipe.scheduler.config)
        return pipe.to(device)

    return factory


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Run EveryDreamValidator on a trained checkpoint and output metrics.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--model",       required=True,
                        help="Path to a diffusers model folder or .safetensors checkpoint.")
    parser.add_argument("--val_config",  default=None,
                        help="Path to validation config JSON (same format used during training). "
                             "If omitted the validator uses its built-in defaults, which requires "
                             "--val_data_root or manual datasets in the config.")
    parser.add_argument("--val_data_root", default=None,
                        help="Shortcut: path to a folder of validation images. Sets val_split_mode "
                             "to 'manual' and points manual_data_root at this folder. Ignored if "
                             "--val_config already specifies data.")
    parser.add_argument("--resolution",  type=int, default=512,
                        help="Training resolution (used when building aspect buckets).")
    parser.add_argument("--batch_size",  type=int, default=1,
                        help="Validation batch size.")
    parser.add_argument("--timestep_start", type=int, default=0,
                        help="Lower bound of the uniform timestep sampling range.")
    parser.add_argument("--timestep_end",   type=int, default=1000,
                        help="Upper bound of the uniform timestep sampling range.")
    parser.add_argument("--device",      default="cuda" if torch.cuda.is_available() else "cpu",
                        help="Device to run on.")
    parser.add_argument("--out",         default=None,
                        help="Write metrics JSON to this path. If omitted, print to stdout.")
    parser.add_argument("--pretty",      action="store_true",
                        help="Pretty-print the JSON output (indent=2).")
    parser.add_argument("--log_level",   default="INFO",
                        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
                        help="Python logging level.")
    args = parser.parse_args()

    logging.basicConfig(
        format="%(levelname)s | %(message)s",
        level=getattr(logging, args.log_level),
    )

    # ---- ensure project root is on sys.path ----
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    if project_root not in sys.path:
        sys.path.insert(0, project_root)

    device = torch.device(args.device)

    # ---- build / patch val config if --val_data_root supplied ----
    val_config_path = args.val_config
    _tmp_config_path = None
    if args.val_data_root and val_config_path is None:
        import tempfile
        cfg = {
            "val_split_mode": "manual",
            "manual_data_root": args.val_data_root,
            "batch_size": args.batch_size,
            "validate_training": True,
        }
        tmp = tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False)
        json.dump(cfg, tmp)
        tmp.close()
        _tmp_config_path = tmp.name
        val_config_path = _tmp_config_path
        logging.info(f"Created temporary val config at {val_config_path}")

    # ---- load model ----
    model = _load_model(args.model, device)

    # ---- set up shim writer ----
    writer = JsonSummaryWriter()

    # ---- build validator ----
    from data.every_dream_validation import EveryDreamValidator

    validator = EveryDreamValidator(
        val_config_path=val_config_path,
        default_batch_size=args.batch_size,
        resolution=args.resolution,
        log_writer=writer,
    )

    # ---- load training items (needed for automatic split) ----
    # For manual mode items list can be empty; the validator loads from the root.
    # For automatic mode we need real items — warn the user.
    val_split_mode = validator.config.get("val_split_mode", "automatic")
    if val_split_mode == "automatic":
        logging.warning(
            "val_split_mode is 'automatic', which steals items from the train set. "
            "For standalone evaluation, prefer 'manual' or pass --val_data_root."
        )

    train_items = []  # stand-alone: no training items to donate
    validator.prepare_validation_splits(train_items, model)

    if not validator.validation_datasets:
        logging.error(
            "No validation datasets were constructed. "
            "Provide a val_config with valid data paths or use --val_data_root."
        )
        sys.exit(1)

    # ---- build callable & optional pipe factory ----
    callable_ = _make_validation_callable(
        model, device,
        timestep_start=args.timestep_start,
        timestep_end=args.timestep_end,
    )

    needs_pipe = (
        validator.anomaly_checkpoint is not None
        or validator.fdd_enabled
    )
    pipe_factory = _make_pipe_factory(model, device) if needs_pipe else None

    # ---- run validation ----
    logging.info("Running validation …")
    try:
        validator.do_validation(
            model=model,
            global_step=0,
            get_model_prediction_and_target_callable=callable_,
            pipe_factory=pipe_factory,
        )
    finally:
        # clean up temp file if we created one
        if _tmp_config_path and os.path.exists(_tmp_config_path):
            os.unlink(_tmp_config_path)

    # ---- output results ----
    metrics = writer.to_dict()

    if args.out:
        with open(args.out, "w") as fout:
            json.dump(metrics, fout, indent=2 if args.pretty else None)
        logging.info(f"Metrics written to {args.out}")
    else:
        # human-readable summary first, then raw JSON
        writer.print_summary()
        print("\n--- Full metrics (JSON) ---")
        print(json.dumps(metrics, indent=2 if args.pretty else None))


if __name__ == "__main__":
    main()


