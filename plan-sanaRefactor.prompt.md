# SANA Refactor: Clarified Implementation Plan

## Goal
Add SANA model training support (`train_sana.py`) to EveryDream2trainer while reusing as much
existing infrastructure as possible — specifically the OOM-robust nibble/accumulation/backward/
optimizer-step machinery in `core/step.py`, the data pipeline, the optimizer wrapper, and the
logging utilities.

**Do the phases in order.** Each phase produces a working, testable state before the next begins.

---

## Phase 1 — Extract `run_accumulation_loop()` from `core/step.py`

**Why first?** This is the highest-risk change. Doing it in isolation — before any SANA code exists
— means you can validate it against the existing SD test suite before any other work.

### What the change is

`train_step()` currently mixes two unrelated responsibilities in one ~400-line function:

- **Generic accumulation machinery** (nibble slicing, OOM retry, loss accumulation, backward,
  optimizer step, TrainingVariables bookkeeping)
- **SD/SDXL-specific model forward** (noise, VAE encoding, conditioning, UNet forward, loss
  computation)

You will split these into two functions:
- `run_accumulation_loop()` — the generic part, callable by any model
- `train_step()` — now just builds an SD-specific loss closure and delegates to the above

### Step 1.1 — Define the `NibbleLossFn` type alias

At the top of `core/step.py`, after the existing imports, add:

```python
# Type alias: any callable that takes a nibble dict and returns a scalar loss tensor with grad.
NibbleLossFn = Callable[
    [dict],         # nibble batch dict (same structure as full_batch but smaller)
    torch.Tensor,   # scalar loss tensor WITH grad attached (mean over nibble images)
]
```

### Step 1.2 — Create `run_accumulation_loop()`

Add this new function to `core/step.py`, placed immediately *above* the existing `train_step()`
function:

```python
def run_accumulation_loop(
    full_batch: dict,
    tv: TrainingVariables,
    ed_optimizer: EveryDreamOptimizer,
    nibble_loss_fn: NibbleLossFn,
    plugin_runner: PluginRunner | None,
    log_writer: SummaryWriter,
    steps_pbar,
    did_step_optimizer_cb: Callable | None,
    args: argparse.Namespace,
    train_progress_01: float = 0.0,
    model_for_autocast=None,      # used only to determine bfloat16 vs float16 for autocast
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
    It may log to log_writer as a side effect.
    """
    if did_step_optimizer_cb is None:
        did_step_optimizer_cb = lambda: True

    remaining_batch = full_batch

    while remaining_batch is not None and remaining_batch['image'].shape[0] > 0:
        batch, remaining_batch = nibble_batch(remaining_batch, get_nibble_size(training_variables=tv))
        assert batch["runt_size"] == 0

        num_images = batch["image"].shape[0]

        # Pre-emptive backward: fire if adding this nibble would exceed max_backward_slice_size
        if tv.accumulated_loss_images_count > 0 and tv.accumulated_loss_images_count + num_images > tv.max_backward_slice_size:
            use_bfloat16 = (model_for_autocast is not None and (getattr(model_for_autocast, 'is_sdxl', False) or getattr(args, 'force_bfloat16', False)))
            with torch.amp.autocast('cuda', enabled=args.amp, dtype=torch.bfloat16 if use_bfloat16 else torch.float16):
                optimizer_backward(ed_optimizer, tv, plugin_runner,
                                   f'pre-emptive backward @{tv.accumulated_loss_images_count}/{tv.max_backward_slice_size}: ')

        # Call the model-specific nibble loss function
        loss_mean = nibble_loss_fn(batch)

        # Accumulate loss into tv
        try:
            tv.accumulate_loss(loss_mean,
                               pathnames=batch.get("pathnames", [])[:num_images],
                               captions=batch.get("captions", {}).get("default", [])[:num_images],
                               timesteps=[])
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

        if ((should_step_optimizer and tv.accumulated_loss_images_count > 0) or
                tv.accumulated_loss_images_count >= tv.max_backward_slice_size):
            use_bfloat16 = (model_for_autocast is not None and (getattr(model_for_autocast, 'is_sdxl', False) or getattr(args, 'force_bfloat16', False)))
            with torch.amp.autocast('cuda', enabled=args.amp, dtype=torch.bfloat16 if use_bfloat16 else torch.float16):
                optimizer_backward(ed_optimizer, tv, plugin_runner, 'regular backward: ')

        if should_step_optimizer and tv.backwarded_images_count > 0:
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

            if tv.interleaved_bs1_count is None or tv.interleaved_bs1_count >= max(1, tv.desired_effective_batch_size ** getattr(args, 'interleave_batch_size_1_alpha', 1)):
                tv.desired_effective_batch_size = choose_effective_batch_size(args, train_progress_01)
                if getattr(args, 'interleave_batch_size_1', False):
                    if tv.interleaved_bs1_count is None:
                        tv.interleaved_bs1_count = 0
                    else:
                        tv.interleaved_bs1_count = None

            did_step_optimizer_cb()
```

### Step 1.3 — Refactor `train_step()` to use `run_accumulation_loop()`

`train_step()` must keep its **exact same function signature** — nothing in `train.py` needs to
change. Internally, it now builds a closure (`nibble_loss_fn`) over the existing SD-specific
forward logic, then calls `run_accumulation_loop()`.

The restructured `train_step()` looks like this (the body of `nibble_loss_fn` is the existing
SD-specific code that currently lives inside the nibble `while` loop):

```python
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
    did_step_optimizer_cb: Callable | None,
    args: argparse.Namespace,
):
    def nibble_loss_fn(batch: dict) -> torch.Tensor:
        # === all of the existing SD-specific per-nibble logic goes here ===
        # (caption selection, noise generation, VAE encoding, text conditioning,
        #  model forward, loss computation, timestep/image loss logging)
        # Returns: loss_mean (scalar Tensor with grad)
        ...
        return loss_mean

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
```

> **Note on what moves where:** Everything currently inside the `while remaining_batch is not None`
> loop in `train_step()` from lines ~78–406 becomes the body of `nibble_loss_fn`. The loop itself,
> the pre-emptive backward trigger, the backward+step block, and the `tv` bookkeeping after
> `optimizer.step()` all move into `run_accumulation_loop()`.

### Step 1.4 — Validate the refactor

Run the existing test suite and a short smoke-train before continuing:

```bash
pytest test/
# Then run a 50-step smoke train on your usual SD/SDXL checkpoint
```

Confirm that:
- All existing tests pass
- Loss values, `tv.optimizer_step`, `tv.total_trained_samples_count` after 50 steps match the
  `main` branch exactly

**Do not proceed to Phase 2 until Phase 1 is green.**

---

## Phase 2 — Create `utils/inference_context.py`

This is a small, self-contained utility. Create the file
`utils/inference_context.py` with the following content:

```python
from contextlib import contextmanager
import torch
from utils.isolate_rng import isolate_rng


@contextmanager
def inference_guard(*modules: torch.nn.Module):
    """
    Sets each module to eval(), isolates the RNG state, then restores training
    mode for any module that was training before.

    Usage:
        with inference_guard(model.unet, model.text_encoder):
            run_inference(...)
    """
    was_training = [m.training for m in modules]
    try:
        for m in modules:
            m.eval()
        with isolate_rng():
            yield
    finally:
        for m, was in zip(modules, was_training):
            if was:
                m.train()
```

Then update `generate_samples()` in `train.py` (around line 1257). Currently it manually calls
`model.unet.eval()`, `model.text_encoder.eval()`, and restores `.train()` in a try/finally block.
Replace those lines with a `with inference_guard(model.unet, model.text_encoder, ...)` block.
The behaviour is **identical** — this is a mechanical cleanup only.

---

## Phase 3 — Create `model/sana_training_model.py`

Create `model/sana_training_model.py`. This file defines a dataclass that holds all SANA
model components, plus factory and save functions.

### 3.1 The dataclass

```python
from __future__ import annotations
from dataclasses import dataclass
from typing import Any, Optional
import torch
import torch.nn as nn


@dataclass
class SanaTrainingModel:
    transformer: nn.Module          # SanaMS — the sole trained component
    text_encoder: nn.Module         # T5/Gemma/Qwen — frozen, not trained
    tokenizer: Any                  # matching tokenizer
    vae: nn.Module                  # DC-AE — frozen, not trained
    vae_config: Any                 # SanaVaeConfig (vae_type, vae_downsample_rate, etc.)
    train_diffusion: Any            # SANA Scheduler with .training_losses()
    sana_config: Any                # full SanaConfig pyrallis object

    transformer_ema: Optional[nn.Module] = None  # reserved for future EMA support

    @property
    def device(self) -> torch.device:
        return next(self.transformer.parameters()).device

    @property
    def dtype(self) -> torch.dtype:
        return next(self.transformer.parameters()).dtype
```

### 3.2 The `load_sana_model()` factory function

This function mirrors steps 3–4 of `sana-train-ref.py` (lines ~200–350). It:
1. Loads the `SanaConfig` (already parsed by the caller)
2. Calls `build_model(sana_config)` to instantiate the transformer
3. Calls `get_tokenizer_and_text_encoder(sana_config)` for the text encoder
4. Calls `get_vae(sana_config)` for DC-AE
5. Instantiates the `Scheduler` from `diffusion.Scheduler`
6. Optionally loads a checkpoint via `load_checkpoint(sana_config)`
7. Returns a populated `SanaTrainingModel`

```python
from argparse import Namespace
from diffusion.model.builder import build_model, get_tokenizer_and_text_encoder, get_vae
from diffusion.utils.config import SanaConfig
from diffusion import Scheduler
from diffusion.utils.checkpoint import load_checkpoint


def load_sana_model(args: Namespace, sana_config: SanaConfig) -> SanaTrainingModel:
    """
    Instantiates and returns a SanaTrainingModel from sana_config.
    Loads a checkpoint if sana_config.model.resume_from or .load_from is set.
    Freezes text_encoder and vae parameters (requires_grad = False).
    """
    transformer = build_model(sana_config)

    tokenizer, text_encoder = get_tokenizer_and_text_encoder(sana_config)
    for p in text_encoder.parameters():
        p.requires_grad_(False)

    vae = get_vae(sana_config)
    for p in vae.parameters():
        p.requires_grad_(False)

    train_diffusion = Scheduler(
        sana_config.scheduler.train_sampling_steps,
        noise_schedule=sana_config.scheduler.noise_schedule,
        predict_v=sana_config.scheduler.predict_v,
        snr_shift_scale=sana_config.scheduler.snr_shift_scale,
    )

    model = SanaTrainingModel(
        transformer=transformer,
        text_encoder=text_encoder,
        tokenizer=tokenizer,
        vae=vae,
        vae_config=sana_config.vae,
        train_diffusion=train_diffusion,
        sana_config=sana_config,
    )

    if sana_config.model.resume_from or sana_config.model.load_from:
        load_checkpoint(model.transformer, sana_config)

    return model
```

### 3.3 The `save_sana_model()` function

```python
import os
import pyrallis
from safetensors.torch import save_file


def save_sana_model(path: str, model: SanaTrainingModel, global_step: int) -> None:
    """
    Saves only the transformer (the trained component) as a safetensors file.
    Saves the config as config.yaml.
    Text encoder and VAE are NOT saved (they are frozen/unchanged).
    """
    os.makedirs(path, exist_ok=True)
    weights_path = os.path.join(path, f"transformer_gs{global_step}.safetensors")
    save_file(model.transformer.state_dict(), weights_path)

    config_path = os.path.join(path, "config.yaml")
    with open(config_path, "w") as f:
        pyrallis.dump(model.sana_config, f)
```

---

## Phase 4 — Create `model/sana_text_encoder.py`

Create `model/sana_text_encoder.py`. Lift the text-encoding logic directly from
`sana-train-ref.py` lines 347–385 and 754–806.

```python
from __future__ import annotations
import os
import torch
import torch.nn as nn
from diffusion.utils.config import SanaConfig


def encode_sana_text(
    tokenizer,
    text_encoder: nn.Module,
    captions: list[str],
    config: SanaConfig,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Encodes a list of caption strings using the appropriate text encoder branch
    (T5, Gemma, or Qwen — selected by config.text_encoder.text_encoder_name).

    Returns:
        y      : (B, 1, N, C) text embeddings
        y_mask : (B, 1, 1, N) attention mask

    Copy the multi-branch tokenize+encode logic from sana-train-ref.py lines ~754–806 here.
    The three branches are:
      "T5":    tokenizer with padding/truncation → text_encoder → reshape to (B, 1, N, C)
      "Gemma": similar but with different attention mask shape
      "Qwen":  similar but uses chat template formatting before tokenizing
    """
    ...


def encode_sana_null_text(
    tokenizer,
    text_encoder: nn.Module,
    config: SanaConfig,
    device: torch.device,
    cache_path: str | None = None,
) -> torch.Tensor:
    """
    Encodes an empty/null caption for unconditional guidance.
    If cache_path is given, writes the result to disk on first call and reads
    from disk on subsequent calls (avoids re-encoding every validation pass).
    """
    if cache_path and os.path.exists(cache_path):
        return torch.load(cache_path, map_location=device)

    null_embedding = encode_sana_text(tokenizer, text_encoder, [""], config, device)[0]

    if cache_path:
        os.makedirs(os.path.dirname(cache_path), exist_ok=True)
        torch.save(null_embedding, cache_path)

    return null_embedding
```

---

## Phase 5 — Create `core/loss_sana.py`

Create `core/loss_sana.py`. This is a thin wrapper around SANA's scheduler, mirroring
`sana-train-ref.py` lines 388–401.

```python
from __future__ import annotations
import torch
from diffusion.model.respace import compute_density_for_timestep_sampling
from model.sana_training_model import SanaTrainingModel


def sample_sana_timesteps(
    batch_size: int,
    train_sampling_steps: int,
    weighting_scheme: str,
    logit_mean: float,
    logit_std: float,
    device: torch.device,
) -> torch.Tensor:
    """
    Draws `batch_size` timestep indices.

    If weighting_scheme is "logit_normal": uses logit-normal distribution
    (via compute_density_for_timestep_sampling from sana-train-ref.py line 395).
    Otherwise: uniform random integers in [0, train_sampling_steps).

    Returns a 1D LongTensor of shape (batch_size,).
    """
    if weighting_scheme == "logit_normal":
        u = compute_density_for_timestep_sampling(
            weighting_scheme=weighting_scheme,
            batch_size=batch_size,
            logit_mean=logit_mean,
            logit_std=logit_std,
        )
        return (u * train_sampling_steps).long().clamp(0, train_sampling_steps - 1).to(device)
    else:
        return torch.randint(0, train_sampling_steps, (batch_size,), device=device)


def compute_sana_loss(
    model: SanaTrainingModel,
    z: torch.Tensor,
    y: torch.Tensor,
    y_mask: torch.Tensor,
    timesteps: torch.Tensor,
    data_info: dict,
) -> torch.Tensor:
    """
    Runs one forward pass through the SANA transformer and returns the mean loss.

    z        : VAE-encoded latents, shape (B, C, H, W)
    y        : text embeddings from encode_sana_text, shape (B, 1, N, C)
    y_mask   : attention mask from encode_sana_text, shape (B, 1, 1, N)
    timesteps: integer timestep indices, shape (B,)
    data_info: dict with keys "img_hw" and "aspect_ratio" (required by SanaMS forward)

    Returns a scalar Tensor with grad attached.
    """
    loss_dict = model.train_diffusion.training_losses(
        model.transformer,
        z,
        timesteps,
        model_kwargs=dict(y=y, mask=y_mask, data_info=data_info),
    )
    return loss_dict["loss"].mean()
```

---

## Phase 6 — Modify `data/every_dream.py`

SANA's DC-AE VAE requires pixel values in the range `[0, 255]`. The existing `EveryDreamBatch`
normalises to `[-1, 1]` by default. You need to make this configurable.

### Step 6.1 — Add `image_output_range` parameter to `EveryDreamBatch.__init__`

In `data/every_dream.py`, find the `__init__` method of `EveryDreamBatch` (around line 44).
Add a new parameter after the existing `name='train'` parameter:

```python
image_output_range: str = "[-1,1]",   # or "[0,255]" for SANA/DC-AE
```

Store it as an instance attribute: `self.image_output_range = image_output_range`

### Step 6.2 — Apply the range conversion in `__getitem__`

In `EveryDreamBatch.__getitem__`, find the section that produces the final `image` tensor
in `[-1, 1]` range. Immediately after that normalisation step, add:

```python
if self.image_output_range == "[0,255]":
    image = (image * 0.5 + 0.5) * 255.0   # [-1, 1] → [0, 255]
```

The default value is `"[-1,1]"`, so **no existing callers are affected by this change**.

---

## Phase 7 — Create `train_sana.py`

This is the new SANA training entry point. Structure it to mirror `train.py`'s organisation:
high-level functions at the top, helpers below.

### Step 7.1 — Argument parsing

```python
def parse_args() -> tuple[argparse.Namespace, SanaConfig]:
    """
    Parses CLI args. --sana_config is required (path to a SanaConfig YAML).
    All other args are optional overrides that take precedence over the YAML.
    """
```

Supported CLI overrides (each mapped to the corresponding SanaConfig field after loading):

| CLI flag | SanaConfig field |
|---|---|
| `--project_name` | `config.work_dir` basename |
| `--logdir` | logging output path |
| `--batch_size` | `config.train.train_batch_size` |
| `--lr` | `config.train.optimizer.lr` |
| `--max_steps` | `config.train.num_steps` |
| `--grad_accum` | `config.train.gradient_accumulation_steps` |
| `--mixed_precision` | `config.model.mixed_precision` |
| `--seed` | `config.train.seed` |
| `--resume_from` | `config.model.resume_from` |
| `--save_every` | `config.train.save_model_steps` |
| `--num_workers` | dataloader num_workers |

Resolution order: YAML file is loaded first, then any provided CLI flags overwrite the
corresponding fields.

### Step 7.2 — `_encode_sana_latents()` helper

This private helper is defined at the bottom of `train_sana.py`:

```python
def _encode_sana_latents(
    model: SanaTrainingModel,
    images: torch.Tensor,
    device: torch.device,
) -> torch.Tensor:
    """
    Encodes a batch of images (float32, values in [0, 255]) using SANA's DC-AE VAE.
    Returns latent tensors on the same device.
    """
    from diffusion.model.builder import vae_encode
    with torch.no_grad():
        return vae_encode(model.vae, images.to(device), model.vae_config.sample_posterior)
```

### Step 7.3 — `train_sana_step()`

This function handles per-batch pre-work (text encoding, VAE encoding, timestep sampling), then
delegates all nibbling, accumulation, backward, and optimizer step to `run_accumulation_loop()`.

```python
def train_sana_step(
    full_batch: dict,
    model: SanaTrainingModel,
    tv: TrainingVariables,
    ed_optimizer: EveryDreamOptimizer,
    log_writer: SummaryWriter,
    steps_pbar,
    device: torch.device,
    args: argparse.Namespace,
) -> None:
    # 1. Text-encode the full batch once (encoder is frozen; no per-nibble re-encoding needed)
    with torch.no_grad():
        y, y_mask = encode_sana_text(
            model.tokenizer, model.text_encoder,
            full_batch["captions"]["default"], model.sana_config, device,
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
        weighting_scheme=model.sana_config.scheduler.weighting_scheme,
        logit_mean=model.sana_config.scheduler.logit_mean,
        logit_std=model.sana_config.scheduler.logit_std,
        device=device,
    )

    # 4. Build the nibble loss closure (closed over pre-encoded z, y, y_mask, timesteps)
    def nibble_loss_fn(nibble: dict) -> torch.Tensor:
        n = nibble["image"].shape[0]
        data_info = {
            "img_hw": nibble.get(
                "img_hw",
                torch.tensor([[full_batch["image"].shape[2], full_batch["image"].shape[3]]] * n),
            ),
            "aspect_ratio": nibble.get("aspect_ratio", torch.ones(n)),
        }
        return compute_sana_loss(model, z[:n], y[:n], y_mask[:n], timesteps[:n], data_info)

    # 5. Generic accumulation loop — handles all nibbling, OOM, backward, optimizer step
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
```

### Step 7.4 — `generate_sana_samples()`

```python
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
    Adapted from sana-train-ref.py log_validation() (lines ~97–200).
    """
```

Key implementation points:
- Wrap the entire inference block with `with inference_guard(model.transformer):` (from Phase 2)
- Encode null text once before the prompt loop via `encode_sana_null_text()`
- Use `DPMS` or `FlowEuler` sampler selected by `model.sana_config.scheduler.vis_sampler`
- Write images to TensorBoard under `"samples-sana/{prompt}"`
- Save `.webp` tiles to `{output_dir}/samples-sana/gs{global_step}/`

### Step 7.5 — `setup_sana_training_variables()`

`TrainingVariables.setup_default_slice_sizes()` expects `args.resolution`, `args.batch_size`,
and `args.forward_slice_size`. Construct a compatible shim from `SanaConfig`:

```python
def setup_sana_training_variables(sana_config: SanaConfig) -> TrainingVariables:
    """
    Creates and initialises a TrainingVariables instance using SANA config values.
    Uses an argparse.Namespace shim to satisfy TrainingVariables.setup_default_slice_sizes().
    """
    tv = TrainingVariables()
    sana_tv_args = Namespace(
        resolution=[sana_config.model.image_size],
        batch_size=sana_config.train.train_batch_size,
        forward_slice_size=None,       # None causes setup_default_slice_sizes to use batch_size
        max_backward_slice_size=None,  # same
    )
    tv.setup_default_slice_sizes(sana_tv_args)
    return tv
```

### Step 7.6 — Overall structure of `main_sana()` and `train_sana_loop()`

Wire everything together following the structure below:

```
main_sana()
  ├─ parse_args()                                    # Step 7.1
  ├─ load_sana_model(args, sana_config)              # Phase 3
  ├─ EveryDreamBatch(                                # Phase 6
  │     data_loader=...,
  │     tokenizer=None,                              # SANA handles its own tokenization
  │     image_output_range="[0,255]",
  │  )
  ├─ EveryDreamOptimizer(model.transformer params)   # ED2 optimizer, transformer params only
  ├─ setup_sana_training_variables(sana_config)      # Step 7.5
  ├─ setup_local_logger() / SummaryWriter            # reused from core/log.py
  └─ train_sana_loop()
       └─ for epoch → for full_batch:
            ├─ train_sana_step()                     # Step 7.3
            ├─ (every save_every steps)
            │    save_sana_model(logdir, model, tv.global_step)
            └─ (every sample_every steps)
                 generate_sana_samples(...)          # Step 7.4
```

Multi-GPU: wrap `model.transformer`, `ed_optimizer`, and the dataloader with
`accelerate.Accelerator.prepare()`. In `train_sana_step`, pass
`accelerator.unwrap_model(model.transformer)` wherever the raw transformer is needed.

---

## Phase 8 — Update `requirements.txt`

Add the following lines to `requirements.txt`:

```
pyrallis
termcolor
# SANA library — install manually from NVIDIA's repo:
# git+https://github.com/NVlabs/Sana.git
```

---

## Files Summary

| Action | File |
|---|---|
| **Modify** | `core/step.py` — add `NibbleLossFn` alias, add `run_accumulation_loop()`, refactor `train_step()` to use it |
| **Create** | `utils/inference_context.py` — `inference_guard()` context manager |
| **Modify** | `train.py` — replace inline eval/train guard in `generate_samples()` with `inference_guard()` |
| **Create** | `model/sana_training_model.py` — `SanaTrainingModel`, `load_sana_model()`, `save_sana_model()` |
| **Create** | `model/sana_text_encoder.py` — `encode_sana_text()`, `encode_sana_null_text()` |
| **Create** | `core/loss_sana.py` — `sample_sana_timesteps()`, `compute_sana_loss()` |
| **Modify** | `data/every_dream.py` — add `image_output_range` parameter to `EveryDreamBatch` |
| **Create** | `train_sana.py` — full SANA training entry point |
| **Modify** | `requirements.txt` — add `pyrallis`, `termcolor`, SANA library note |

---

## Deferred (not in scope for this implementation)

- **EMA**: The `transformer_ema` slot on `SanaTrainingModel` is reserved but not implemented.
  When added later, follow the `ema_update()` pattern in `sana-train-ref.py` (lines 88–93).
- **Pre-computed VAE/text-encoder feature caching** (`load_vae_feat`, `load_text_feat`):
  deferred to a later pass; can be added to `data/latent_cache.py`.
- **Online metrics / FSDP**: not in scope.

