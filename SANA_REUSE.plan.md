# SANA Reuse Plan — Wire SANA into existing ED2 infrastructure

## Goal

Replace three ad-hoc SANA-only reimplementations with the infrastructure that
already exists for SD2/SDXL flow-matching training:

| What we built fresh | What already exists | Action |
|---|---|---|
| `sample_flow_sigmas()` + `--weighting_scheme/logit_mean/logit_std/num_train_timesteps` | `get_multirank_stratified_random_timesteps()` + `--timesteps_multirank_stratified*` | Delete custom code, add missing args to `train_sana.py` |
| Custom `generate_sana_samples()` with hand-rolled pipeline | `SampleGenerator` + `SampleRequest` used by `train.py` | Delete custom function, wire `SampleGenerator` |
| Raw `FlowMatchEulerDiscreteScheduler` stored in `SanaTrainingModel` | `TrainFlowMatchEulerDiscreteScheduler` from `core/flow_match_model.py` | Swap at model-load time, rename `scheduler` → `noise_scheduler` |

---

## Background

### Existing timestep sampling

`get_multirank_stratified_random_timesteps()` in `core/loss.py` already implements
all the distributions we need:

| `--timesteps_multirank_stratified_distribution` | Note |
|---|---|
| `uniform` | Stratified uniform over [0, 1000) |
| `beta` | Beta PPF; `alpha`/`beta` params |
| `mode` | Mode-weighted; `mode_scale` param |
| `boundary-oversampling` | Heavy-tails sampling |
| `lognormal` | **`alpha` = std of the underlying normal**, mean fixed at 0 — equivalent to the old `logit_std` / `logit_mean=0` |

The function returns a `LongTensor` of integer indices in `[0, 999]`.

For flow-matching models `TrainFlowMatchEulerDiscreteScheduler.get_shifted_timesteps(
indices, noise_scheduler.timesteps)` converts those indices to the float timestep
values required by the transformer forward pass (incorporating any configured shift).

The buffer of pre-drawn timesteps is maintained in `tv.remaining_stratified_timesteps`
(already a field on `TrainingVariables`).

### Existing sample-generation infrastructure

`SampleGenerator` (`utils/sample_generator.py`) already handles:
- Reading prompts from a `.txt` or `.json` config file
- `SampleRequest` carrying `prompt`, `negative_prompt`, `seed`, `size`
- Multiple CFG scales per image (the side-by-side grid)
- TensorBoard image logging via `save_sample_image()`
- Periodic triggering via `should_generate_samples(global_step, local_step)`
- `create_inference_pipe(model, scheduler_config, …)` — auto-overrides to
  `flow-matching` scheduler when `model.is_flow_matching` is True

The only SANA-specific things it cannot do yet are:
1. Skip `CompelForSD` for `SanaPipeline` (currently skipped only for SDXL)
2. Omit SD/SDXL-only kwargs (`guidance_rescale`, `pooled_prompt_embeds`) when
   calling a `SanaPipeline`

### `TrainFlowMatchEulerDiscreteScheduler`

Already in `core/flow_match_model.py`. Subclasses `FlowMatchEulerDiscreteScheduler`
and adds:
- `get_shifted_timesteps(integer_indices, scheduler.timesteps)` → float timesteps
- `get_sigmas_for_timesteps(timesteps)` → sigma values
- `add_noise(latents, noise, timesteps)` → delegates to `scale_noise()`
  (the same noising formula the rest of ED2 uses)

This is the right scheduler to store on `SanaTrainingModel` during training.
At inference time `SDPipelineInferenceFlowMatchEulerDiscreteScheduler` is used
(already the inference scheduler for SDXL/SD3 in `SampleGenerator._create_scheduler`).

---

## Files changed

| File | Action |
|---|---|
| `model/sana_training_model.py` | Rename `scheduler` → `noise_scheduler`; store `TrainFlowMatchEulerDiscreteScheduler`; add `is_flow_matching` and `build_inference_pipeline()` |
| `core/loss_sana.py` | Remove `sample_flow_sigmas()`; update `compute_sana_loss()` to accept `noise_scheduler` + integer-derived float `timesteps` |
| `train_sana.py` | Remove 6 custom args; add `--timesteps_multirank_stratified*` args; replace `generate_sana_samples()` with `SampleGenerator`; update `train_sana_step()` |
| `utils/sample_generator.py` | Add `SanaPipeline` detection alongside existing SDXL branch |

---

## Phase 1 — `model/sana_training_model.py`

### 1a. Rename `scheduler` → `noise_scheduler`

Every reference throughout the codebase uses `model.noise_scheduler` (see
`core/step.py`, `core/loss.py`, `model/training_model.py`). Rename the field on
`SanaTrainingModel` to match.

### 1b. Use `TrainFlowMatchEulerDiscreteScheduler` at load time

In `load_sana_model()`, after `SanaPipeline.from_pretrained(...)`:

```python
from core.flow_match_model import TrainFlowMatchEulerDiscreteScheduler
noise_scheduler = TrainFlowMatchEulerDiscreteScheduler.from_config(pipe.scheduler.config)
```

Store this as `model.noise_scheduler`.  The stock `pipe.scheduler` is discarded;
it is only needed for inference (see Phase 1c).

### 1c. Add `is_flow_matching` property and `build_inference_pipeline()`

`SampleGenerator.create_inference_pipe()` calls these two on whatever model it
receives.  Add them to `SanaTrainingModel` so it satisfies the same duck-type
interface as `TrainingModel`:

```python
@property
def is_flow_matching(self) -> bool:
    return True

def build_inference_pipeline(self, scheduler=None):
    from diffusers import SanaPipeline
    from core.flow_match_model import SDPipelineInferenceFlowMatchEulerDiscreteScheduler
    inf_scheduler = scheduler or SDPipelineInferenceFlowMatchEulerDiscreteScheduler.from_config(
        self.noise_scheduler.config
    )
    return SanaPipeline(
        transformer=self.transformer,
        text_encoder=self.text_encoder,
        tokenizer=self.tokenizer,
        vae=self.vae,
        scheduler=inf_scheduler,
    )
```

### 1d. Remove `guidance_scale` field

This will come from `SampleGenerator`'s `cfgs` list (config file or CLI default),
not from `SanaTrainingModel`.  Drop the field.

---

## Phase 2 — `core/loss_sana.py`

### 2a. Remove `sample_flow_sigmas()`

Delete the function entirely.  Timestep sampling is now the caller's responsibility
(via `get_multirank_stratified_random_timesteps` + `get_shifted_timesteps`).

### 2b. Update `compute_sana_loss()` signature

Old signature:
```python
def compute_sana_loss(transformer, z, y, y_mask, sigma, timestep_t) -> Tensor
```

New signature:
```python
def compute_sana_loss(transformer, noise_scheduler, z, y, y_mask, timesteps) -> Tensor
```

where `timesteps` are float tensors produced by
`TrainFlowMatchEulerDiscreteScheduler.get_shifted_timesteps(...)`.

### 2c. Rewrite the body

```python
noise = torch.randn_like(z)

# Noise the latents using the shared add_noise path
# (TrainFlowMatchEulerDiscreteScheduler.add_noise → scale_noise)
noisy_z = noise_scheduler.add_noise(z, noise, timesteps)

# Flow-matching velocity target: v = ε − z₀
target = noise - z

model_pred = transformer(
    hidden_states=noisy_z,
    encoder_hidden_states=y,
    timestep=timesteps,
    encoder_attention_mask=y_mask,
).sample

return F.mse_loss(model_pred.float(), target.float())
```

This is now structurally identical to the flow-matching path in `core/loss.py`
`_get_noisy_latents` / `_get_target`, making it easy to unify later if desired.

---

## Phase 3 — `train_sana.py`

### 3a. Remove 6 now-redundant CLI args

Remove from `parse_args()` and the returned `Namespace`:

| Removed arg | Replaced by |
|---|---|
| `--weighting_scheme` | `--timesteps_multirank_stratified_distribution lognormal` |
| `--logit_mean` | (lognormal mean is fixed at 0 in the existing implementation) |
| `--logit_std` | `--timesteps_multirank_stratified_alpha <value>` |
| `--num_train_timesteps` | read from `noise_scheduler.config.num_train_timesteps` |
| `--guidance_scale` | `SampleGenerator` config (`cfgs` list) |
| `--sample_height` / `--sample_width` | `SampleRequest.size` from prompt config file |

### 3b. Add `--timesteps_multirank_stratified*` args

Mirror `train.py` exactly:

```python
parser.add_argument("--timesteps_multirank_stratified",
                    action=argparse.BooleanOptionalAction, default=True,
                    help="Use multirank stratified timestep sampling")
parser.add_argument("--timesteps_multirank_stratified_distribution",
                    type=str,
                    choices=['uniform', 'beta', 'mode', 'boundary-oversampling', 'lognormal'],
                    default='lognormal',
                    help="Timestep distribution. For 'lognormal', --alpha controls the std "
                         "of the underlying normal (width of the distribution).")
parser.add_argument("--timesteps_multirank_stratified_stratify",
                    action=argparse.BooleanOptionalAction, default=True)
parser.add_argument("--timesteps_multirank_stratified_alpha", type=float, default=1.0,
                    help="Alpha/std parameter. For lognormal: std of the normal before sigmoid.")
parser.add_argument("--timesteps_multirank_stratified_beta",  type=float, default=1.6)
parser.add_argument("--timesteps_multirank_stratified_mode_scale", type=float, default=0.5)
```

Default distribution is `lognormal` with `alpha=1.0` (moderate width, equivalent
to the old `logit_std=1.0`).  Users who want wider/narrower coverage simply change
`--timesteps_multirank_stratified_alpha`.

### 3c. Add `--sample_prompts` arg

```python
parser.add_argument("--sample_prompts", type=str, default="sample_prompts.json",
                    help="Path to a sample-prompts .json or .txt file for SampleGenerator")
```

The existing `sample_prompts.json` in the repo root already has the right format.

### 3d. Update `train_sana_step()` — timestep sampling

Replace the `sample_flow_sigmas()` call with the stratified path:

```python
from core.loss import get_multirank_stratified_random_timesteps
from core.flow_match_model import TrainFlowMatchEulerDiscreteScheduler

# 1. Draw stratified integer indices into the timestep schedule [0, 999]
if args.timesteps_multirank_stratified:
    while (tv.remaining_stratified_timesteps is None
           or tv.remaining_stratified_timesteps.shape[0] < full_batch_size):
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
    timestep_indices = tv.remaining_stratified_timesteps[:full_batch_size]
    tv.remaining_stratified_timesteps = tv.remaining_stratified_timesteps[full_batch_size:]
else:
    timestep_indices = torch.randint(0, model.noise_scheduler.config.num_train_timesteps,
                                     (full_batch_size,))

# 2. Convert integer indices → float timesteps incorporating any scheduler shift
timesteps = TrainFlowMatchEulerDiscreteScheduler.get_shifted_timesteps(
    timestep_indices, model.noise_scheduler.timesteps
).to(device)
```

Then pass `timesteps` (float) to `compute_sana_loss(model.transformer, model.noise_scheduler, z, y, y_mask, timesteps)`.

Also reset `tv.remaining_stratified_timesteps = None` at the start of each epoch
(mirrors `train.py` line 1433).

### 3e. Replace `generate_sana_samples()` with `SampleGenerator`

Delete `generate_sana_samples()` entirely.

In `main_sana()`, set up a `SampleGenerator`:

```python
from utils.sample_generator import SampleGenerator

sample_generator = SampleGenerator(
    log_folder=logdir,
    log_writer=log_writer,
    default_resolution=args.resolution[0],
    config_file_path=args.sample_prompts if os.path.exists(args.sample_prompts) else None,
    batch_size=1,
    default_seed=args.seed,
    default_sample_steps=args.sample_every,
)
```

In `train_sana_loop()`, replace the `if global_step % args.sample_every == 0` block with:

```python
if sample_generator.should_generate_samples(global_step, local_step=0):
    with inference_guard(model.transformer):
        pipe = sample_generator.create_inference_pipe(
            model_being_trained=model,
            diffusers_scheduler_config=model.noise_scheduler.config,
        ).to(device)
        sample_generator.generate_samples(pipe, global_step)
    del pipe
    gc.collect()
    torch.cuda.empty_cache()
```

`create_inference_pipe` will auto-detect `model.is_flow_matching == True` and switch
the scheduler to `flow-matching` (i.e. `SDPipelineInferenceFlowMatchEulerDiscreteScheduler`),
then call `model.build_inference_pipeline(scheduler=...)` to get a `SanaPipeline`.

### 3f. Clean up now-unused imports

Remove `copy` (was used for `deepcopy(model.scheduler)`).
Remove `encode_null_prompt` import (was used only by the deleted function).

---

## Phase 4 — `utils/sample_generator.py`

### 4a. Detect `SanaPipeline` in `generate_samples()`

In the `do_regular_samples` branch, the `compel` setup currently reads:

```python
if type(pipe) is StableDiffusionXLPipeline:
    compel = None
else:
    compel = CompelForSD(pipe)
```

Extend to:

```python
from diffusers import SanaPipeline

if type(pipe) in (StableDiffusionXLPipeline, SanaPipeline):
    compel = None
else:
    compel = CompelForSD(pipe)
```

### 4b. Skip SD/SDXL-only kwargs for `SanaPipeline`

The `pipe(...)` call currently always passes `guidance_rescale` and, when `compel`
is not None, `prompt_embeds` / `pooled_prompt_embeds`.  `SanaPipeline` does not
accept these.

Add a guard around the `guidance_rescale` kwarg:

```python
extra_pipe_kwargs = {}
if not isinstance(pipe, SanaPipeline):
    extra_pipe_kwargs['guidance_rescale'] = self.guidance_rescale

images = pipe(
    prompt=prompt,
    prompt_embeds=embeds,          # None for SANA (compel=None)
    pooled_prompt_embeds=pooled_prompt_embeds,  # None for SANA
    negative_prompt=negative_prompt,
    negative_prompt_embeds=negative_embeds,     # None for SANA
    negative_pooled_prompt_embeds=negative_pooled_embeds,  # None for SANA
    num_inference_steps=self.num_inference_steps,
    num_images_per_prompt=1,
    guidance_scale=cfg,
    generator=generators,
    width=size[0],
    height=size[1],
    **extra_pipe_kwargs,
).images
```

`SanaPipeline.__call__` ignores `None` values for the embedding kwargs it doesn't
use, so no additional branching is required for those.

### 4c. Relax type annotation on `create_inference_pipe()`

Change the parameter type from `TrainingModel` to `Any` (or introduce a
`SupportsInferencePipeline` Protocol if stricter typing is desired later):

```python
def create_inference_pipe(self, model_being_trained, diffusers_scheduler_config, ...)
```

---

## Arg mapping cheatsheet

| Old `train_sana.py` arg | Replacement |
|---|---|
| `--weighting_scheme logit_normal` | `--timesteps_multirank_stratified --timesteps_multirank_stratified_distribution lognormal` |
| `--logit_std 1.0` | `--timesteps_multirank_stratified_alpha 1.0` |
| `--logit_mean 0.0` | (lognormal mean is always 0 in the implementation) |
| `--num_train_timesteps 1000` | read from `noise_scheduler.config.num_train_timesteps` |
| `--guidance_scale 4.5` | set `cfgs: [4.5]` in `sample_prompts.json` |
| `--sample_height 1024 --sample_width 1024` | set `resolution: 1024` in `sample_prompts.json`, or rely on `default_resolution=args.resolution[0]` |

---

## Success criteria

1. `pytest test/` — all 84 tests pass, no regressions.
2. `python train_sana.py --help` — lists `--timesteps_multirank_stratified*` flags;
   does NOT list `--weighting_scheme`, `--logit_mean`, `--logit_std`.
3. `grep -r "from diffusion\|import pyrallis\|sample_flow_sigmas\|generate_sana_samples"
   train_sana.py core/loss_sana.py model/sana_training_model.py` — zero hits.
4. `SanaTrainingModel` has `is_flow_matching`, `build_inference_pipeline()`, and
   `noise_scheduler` (of type `TrainFlowMatchEulerDiscreteScheduler`).

