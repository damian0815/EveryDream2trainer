# Timestep Interval Sampling Plan

## Goal

Construct batches where all samples share the **same narrow timestep interval**, so that every
gradient in the batch is teaching the UNet to solve the same sub-problem (e.g. "build global
structure" or "refine textures"). The interval rotates every optimizer step so the full
trajectory is covered across training.

---

## On Interval Boundaries: SNR-Based Clustering (DP)

Equal-width intervals are a poor fit for diffusion/flow-matching schedules because SNR varies
**highly nonlinearly** with timestep — the SNR curve is nearly flat at high noise levels and
changes rapidly near t=0. Equal-width buckets would produce intervals that are semantically
very unequal: some containing large SNR swings (heterogeneous gradients), others containing
nearly identical SNR values.

Instead, we use a **dynamic programming** approach that finds the `k` contiguous intervals
that minimise within-interval SNR heterogeneity. The cost function for an interval `[l, r]`
is the sum of absolute deviations of all SNR values in the interval from the interval's
centre SNR value — a median-deviation measure. An interval is "cheap" if all timesteps in it
have similar SNR values. The DP finds the globally optimal k-partition of `[0, T]` under
this cost.

This is computed **once at training startup** from the scheduler's SNR curve, and the
resulting `k` boundary pairs are stored. Each optimizer step picks one interval uniformly
at random.

### SNR source per scheduler type

| Scheduler | SNR formula |
|-----------|-------------|
| Flow matching | `SNR(t) = t² / (1 − t)²` where `t ∈ [0, 1]` mapped from integer timesteps via `t = timestep / num_train_timesteps` |
| DDPM / v-pred | `compute_snr(torch.arange(1000), noise_scheduler)` — existing function in `core/loss.py` |

---

## Algorithm

### Parameters (new CLI args)

| Arg | Type | Default | Meaning |
|-----|------|---------|---------|
| `--timestep_interval_n` | `int` | `10` | Number of SNR-homogeneous intervals (k in the DP) |
| `--timestep_interval_sampling` | flag | `False` | Enable this mode (mutually exclusive with `--timesteps_multirank_stratified`) |

### One-time startup: compute SNR clusters

```python
def snr_based_clustering(snr: np.ndarray, k: int) -> list[tuple[int, int]]:
    """
    Dynamic programming partition of [0, len(snr)-1] into k contiguous intervals
    minimising within-interval SNR heterogeneity.

    Cost of interval [left, right]: sum of |snr[t] - snr[center]| for t in [left, right].
    Returns k (left, right) inclusive timestep pairs.
    """
    n = snr.shape[0]
    D = np.full((n, k), np.inf)   # D[i,j] = min cost to cover [0..i] with j+1 clusters
    S = np.zeros((n, k), dtype=int)

    def interval_cost(left, right):
        if left == right:
            return 0.0
        center = round((left + right + 1) / 2)
        return np.abs(snr[left:right + 1] - snr[center]).sum()

    # Fill DP table
    for j in range(k):
        for i in reversed(range(n)):
            if j == 0:
                D[i, j] = interval_cost(0, i)
            elif i >= j:
                costs = np.full(i, np.inf)
                for L in range(j, i):
                    costs[L] = D[L, j - 1] + interval_cost(L + 1, i)
                D[i, j] = costs.min()
                S[i, j] = costs.argmin()

    # Backtrack to recover boundaries
    bounds = []
    b = 0
    for j in reversed(range(k)):
        b = S[-1, k - 1] if j == k - 1 else S[int(b), j]
        bounds.append(int(b))

    # Build (left, right) inclusive pairs
    clusters = []
    reversed_bounds = list(reversed(bounds))
    for idx in range(k):
        left  = reversed_bounds[idx]
        right = reversed_bounds[idx + 1] - 1 if idx + 1 < k else n - 1
        clusters.append((int(left), int(right)))
    return clusters


def compute_timestep_intervals(noise_scheduler, k: int,
                               t_start: int, t_end: int) -> list[tuple[int, int]]:
    """
    Compute k SNR-homogeneous intervals over [t_start, t_end].
    Works for both flow-matching and DDPM schedulers.
    """
    import numpy as np
    from core.loss import compute_snr
    from core.flow_match_model import TrainFlowMatchEulerDiscreteScheduler
    from diffusers import FlowMatchEulerDiscreteScheduler

    all_t = torch.arange(t_start, t_end)
    if isinstance(noise_scheduler, (TrainFlowMatchEulerDiscreteScheduler,
                                    FlowMatchEulerDiscreteScheduler)):
        t_norm = all_t.float() / noise_scheduler.config.num_train_timesteps
        snr = (t_norm ** 2 / (1 - t_norm + 1e-8) ** 2).numpy()
    else:
        snr = compute_snr(all_t, noise_scheduler).numpy()

    clusters = snr_based_clustering(snr, k)
    # clusters are relative to t_start — shift back to absolute timestep indices
    return [(t_start + l, t_start + r) for l, r in clusters]
```

Both functions live in **`core/loss.py`**, alongside the existing `compute_snr`.

### New `TrainingVariables` fields

```python
# model/training_model.py — add to TrainingVariables dataclass
current_timestep_interval: tuple[int, int] | None = None
timestep_intervals: list[tuple[int, int]] | None = None   # pre-computed SNR clusters
```

`timestep_intervals` is populated once in `train.py` after the scheduler is available;
`current_timestep_interval` is the per-step latch, set at first nibble and cleared after
`step_optimizer`.

### Startup (train.py, after scheduler is ready)

```python
if args.timestep_interval_sampling:
    if args.timesteps_multirank_stratified:
        raise ValueError("--timestep_interval_sampling and --timesteps_multirank_stratified "
                         "are mutually exclusive")
    tv.timestep_intervals = compute_timestep_intervals(
        model.noise_scheduler,
        k=args.timestep_interval_n,
        t_start=args.timestep_start,
        t_end=args.timestep_end,
    )
    logging.info(f"Timestep interval sampling: {len(tv.timestep_intervals)} SNR-based intervals: "
                 f"{tv.timestep_intervals}")
```

### Step-level logic (`_get_step_timesteps_internal`, new `elif` branch)

```python
elif args.timestep_interval_sampling:
    if tv.current_timestep_interval is None:
        # pick one SNR-homogeneous interval uniformly for this optimizer step
        tv.current_timestep_interval = random.choice(tv.timestep_intervals)

    t_lo, t_hi = tv.current_timestep_interval
    timesteps = torch.randint(
        low=t_lo,
        high=max(t_lo + 1, t_hi + 1),   # randint high is exclusive
        size=(full_batch_size,),
        device=model.unet.device,
    ).long()
    return timesteps
```

### Clear interval after optimizer step (`train_step`, one new line)

```python
# immediately after tv.optimizer_step += 1:
tv.current_timestep_interval = None   # force new interval draw on next step
```

---

## What Does NOT Change

- `get_multirank_stratified_random_timesteps` — untouched
- `get_uniform_timesteps` — untouched
- The nibble loop, backward/optimizer step logic — untouched except the one-line clear
- Validation, logging, loss computation — all unaffected; timesteps are just integers downstream

---

## Interaction with Existing Features

| Feature | Interaction |
|---------|-------------|
| `--timesteps_multirank_stratified` | Mutually exclusive. `ValueError` at startup. |
| `--batch_share_timesteps` | Compatible but redundant: interval sampling already narrows the range. |
| `--timestep_curriculum_alpha` | Not compatible in this version — clusters are pre-computed from the initial `t_start`/`t_end`. Warn and disable curriculum if both are set. |
| `--timestep_start` / `--timestep_end` | Respected as the outer bounds; SNR is computed and clustered within this range. |
| Nibble loop | Correct by design: `current_timestep_interval` persists across nibbles within one optimizer step, cleared only after `step_optimizer`. |
| `full_batch_timesteps_range` (per-image overrides) | Ignored when interval sampling is active — document this. |
| Flow matching shift | Clusters are computed on the **unshifted** integer timestep axis (same as all other timestep logic). Shift is applied afterward in `train_step` as normal. |

---

## Complexity Note

The DP is O(k · n²) where n = `t_end − t_start` (≤ 1000) and k = `--timestep_interval_n`
(≤ ~20 in practice). At n=1000, k=10 this is ~10M operations — runs in well under a second
at startup, numpy-only. No impact on training throughput.

---

## Summary of File Changes

| File | Lines changed | Nature |
|------|--------------|--------|
| `core/loss.py` | +~50 | `snr_based_clustering` and `compute_timestep_intervals` functions |
| `model/training_model.py` | +2 | Two new fields on `TrainingVariables` |
| `core/step.py` | +~12 | New `elif` branch in `_get_step_timesteps_internal`; +1 line after optimizer step |
| `train.py` | +~10 | Two new argparser entries; startup cluster computation |

Total: ~75 lines of new code. No existing logic is touched.
