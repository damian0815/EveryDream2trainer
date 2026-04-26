# Plan: `DifficultyEstimator` for Adaptive Image Scheduling

A new `DifficultyEstimator` class (in `data/difficulty_estimator.py`) ingests per-image, per-timestep losses from `LogData` after each epoch, computes per-image difficulty scores by normalising raw losses against a reference mean-loss-per-timestep curve, and updates each `ImageTrainItem.multiplier` before the next `train_batch.shuffle()`. The design separates difficulty *estimation* from difficulty *scheduling policy* via a `DifficultyScheduler` strategy interface, making Schedule Type B a drop-in addition later.

## Steps

### 1. Create `data/difficulty_estimator.py` with three components

- **Loss normalisation via `core.mean_loss_per_timestep.normalize_loss_by_timestep`** — call `normalize_loss_by_timestep(loss_1d, timesteps, model_type_identifier)` (imported from `core.mean_loss_per_timestep`) to obtain per-sample losses relative to the per-timestep mean for that model type. The `model_type_identifier` string (e.g. `'flowmatching-sd2'`) is stored on `DifficultyEstimator` at construction and threaded in from training args (see Step 3). Currently only `'flowmatching-sd2'` is implemented; adding new model types means adding a new key to `_mean_loss_per_timestep` in `core/mean_loss_per_timestep.py` with no changes required in `DifficultyEstimator`.

- **`DifficultyEstimator` class**: stores `model_type_identifier: str` at construction; loads/saves a JSON database (keyed by `os.path.realpath(pathname)`) of per-image EMA difficulty scores and observation counts; `ingest_epoch_losses(log_data: LogData)` reads `log_data.loss_per_image_and_timestep`, calls `normalize_loss_by_timestep` to normalise each raw loss, computes `mean(log(normalised_loss))` per image, and updates EMA scores; `save(path)` persists the DB.

- **A `DifficultyScheduler` abstract base** with a single method `update_multipliers(items: list[ImageTrainItem], scores: dict[str, float])`, plus a concrete `TypeAScheduler(min_multiplier, max_multiplier, expand_factor)` that maps `exp(score * expand_factor)` clamped to `[min_multiplier, max_multiplier]` onto `item.multiplier`. Type B will subclass `DifficultyScheduler` and interact with `EveryDreamBatch`/`DataLoaderMultiAspect` directly.

### 2. Expose `DataLoaderMultiAspect.recompute_expected_epoch_size()` in `data/data_loader.py`

A one-liner `self.expected_epoch_size = math.floor(sum(i.multiplier for i in self.prepared_train_data))` — called after multipliers are mutated so that `get_shuffled_image_buckets` continues to request the correct `required_count`.

### 3. Add `--difficulty_estimator` arg group to `train.py`

- `--difficulty_estimator <path/to/db.json>` — enables the feature; path used for both loading and saving the DB
- `--difficulty_estimator_scheduler` (`typeA`; default) — selects the `DifficultyScheduler` subclass
- `--difficulty_estimator_min_multiplier`
- `--difficulty_estimator_max_multiplier`
- `--difficulty_estimator_expand_factor`
- `--difficulty_estimator_ema_alpha`
- `--difficulty_estimator_slab_size` — (Type B) number of batches per slab; default e.g. 100
- `--difficulty_estimator_base_interval` — (Type B) inter-slab interval for a neutral-scored image; defaults to `ceil(dataset_size / slab_size)` (one epoch's worth of slabs). Must be > 1 to produce any scheduling differentiation between hard and neutral images.
- `--difficulty_estimator_min_observation_count` — minimum number of `(timestep, loss)` observations an image must have accumulated before its difficulty score is used to adjust its scheduled appearances. Until this threshold is reached the image is treated as unscored (see cold-start policy in Step 6).
- `--difficulty_estimator_model_type` — the `model_type_identifier` string passed to `normalize_loss_by_timestep` (e.g. `'flowmatching-sd2'`). Auto-inferred where possible (e.g. `args.train_sampler == "flow-matching"` → `'flowmatching-sd2'`), with this flag as an explicit override. `DifficultyEstimator` will raise a clear error if the identifier is not recognised, so supporting a new model type only requires extending `core/mean_loss_per_timestep.py`.

### 4. Wire into the epoch-end loop in `train.py` (around line 1371)

Before `train_batch.shuffle(...)`, call:
1. `difficulty_estimator.ingest_epoch_losses(log_data)`
2. `difficulty_estimator.scheduler.update_multipliers(train_batch.data_loader.prepared_train_data, difficulty_estimator.scores)`
3. `train_batch.data_loader.recompute_expected_epoch_size()`
4. `difficulty_estimator.save(args.difficulty_estimator)`

Gate the whole block with `if difficulty_estimator is not None`.

### 5. Preserve the original `item.multiplier` as `item.base_multiplier`

Set once at resolve time in `resolve_image_train_items` in `train.py` so that `TypeAScheduler` can scale *relative to the user-configured multiplier* rather than overwriting it, keeping the DB difficulty score decoupled from the item's data-weighting multiplier.

### 6. Cold-start policy: unscored images are always guaranteed a slot

When `DifficultyEstimator` computes multipliers (for both Type A and Type B), images whose `observation_count < min_observation_count` are **unscored**. Their `multiplier` is left at `base_multiplier` unconditionally and they are exempt from any reduction below that value. This guarantees that every image reaches `min_observation_count` before difficulty adjustments begin.

The critical corollary for Type B is that unscored images must not be crowded out by high-difficulty scored images. Unscored images advance through the slab schedule at `base_interval` (equivalent to a medium-difficulty scored image), but they have **priority over scored images when competing for capacity**. Concretely: `build_next_slab` first fills due unscored image slots, then fills remaining capacity with due scored images. This means high-difficulty scored images can legitimately repeat within a slab before every image has been seen — they simply cannot displace an unscored image that is due.

---

## Type B: Spaced Repetition via Epoch-Slab Reconstruction

Type B replaces the Type A model of per-epoch multiplier adjustment with a finer-grained schedule: each image has a *next-due step* derived from its difficulty score, and the dataset is reconstructed every `slab_size` training steps to reflect the current schedule. This is implemented as **slab reconstruction** — the least invasive approach compatible with the existing multi-worker DataLoader architecture.

### How it works

1. Add `--difficulty_estimator_slab_size <K>` (number of batches per slab; default e.g. 100) and `--difficulty_estimator_base_interval <N>` (default: `epoch_size_in_slabs`, i.e. `ceil(dataset_size / slab_size)`).
2. Add `TypeBScheduler(slab_size, base_interval, min_multiplier, max_multiplier, expand_factor)` as a subclass of `DifficultyScheduler`. Instead of setting `item.multiplier` once per epoch, it maintains a `dict[str, int]` of `next_due_slab` for each image (keyed by realpath). Difficulty score determines inter-slab interval: `interval = round(base_interval / exp(score * expand_factor))` clamped to `[1, max_interval]` slabs. A neutral image (score=0) therefore appears exactly once per epoch; a hard image appears more often; an easy image less often. **`base_interval` must be greater than 1 to produce any differentiation between hard and neutral images** — with `base_interval = 1` both hard and neutral round to interval 1 (the minimum clamp), giving identical scheduling. Setting `base_interval = epoch_size_in_slabs` is the natural default: it anchors neutral images to once-per-epoch and lets difficulty compress or stretch that rate symmetrically.
3. `TypeBScheduler.build_next_slab(items, scores, current_slab, n_batches) -> list[ImageTrainItem]` replaces `update_multipliers` for active use. It:
   a. Partitions items into **unscored** (`observation_count < min_observation_count`) and **scored**.
   b. Schedules unscored items at `base_interval` (like a neutral-difficulty scored item); selects those whose `next_due_slab <= current_slab` and reserves their slots with priority — they are filled before scored items compete for capacity.
   c. For scored items, selects those whose `next_due_slab <= current_slab`, weighted by difficulty, to fill remaining slab capacity.
   d. Updates `next_due_slab` for each selected item.
   e. Returns the assembled `list[ImageTrainItem]` (may include duplicates for high-difficulty items).
4. In `train.py`, the inner training loop checks `if step % slab_size == 0 and difficulty_estimator is not None and isinstance(difficulty_estimator.scheduler, TypeBScheduler)`. When true, it calls `build_next_slab`, replaces `train_batch.image_train_items` in-place, and discards the current DataLoader iterator, rebuilding it around the updated dataset. The outer epoch loop and `train_batch.shuffle()` continue to exist for compatibility with Type A and for checkpoint/logging cadence.

### Interactions with the DataLoader

- The existing multi-worker DataLoader (map-style `EveryDreamBatch`, `num_workers>0`) is preserved. Only the item list that backs `EveryDreamBatch` is replaced at slab boundaries.
- At each slab boundary, the current DataLoader iterator is torn down. Any items already prefetched (`prefetch_factor × num_workers`, typically 2–8 items) are discarded — a minor, acceptable wastage.
- Worker processes are not communicated with directly; they are simply stopped (iterator exhausted/discarded) and restarted with the new item list. This avoids all IPC complexity.
- Spaced repetition intervals are measured in slabs, not individual steps. This is a practical granularity — a single gradient step rarely shifts an image's difficulty score enough to justify step-level rescheduling.

### Epoch semantics under Type B

The outer epoch counter and all epoch-level side-effects (logging, LR scheduler steps, checkpoint saves, `train_batch.shuffle()`) are preserved unchanged. Inside each epoch, `train_batch.shuffle()` still runs at epoch start and provides the initial full-dataset item list that `TypeBScheduler` then sub-samples into slabs. This means Type B degrades gracefully to Type A behaviour when `slab_size >= epoch_size`.

---

## Further Considerations

1. **Extending to new model types**: To support additional model types (e.g. `epsilon`, `v_prediction`), add the corresponding key and per-timestep loss list to `_mean_loss_per_timestep` in `core/mean_loss_per_timestep.py` and update the `Literal` type hint on `normalize_loss_by_timestep`. No other files need to change.

2. **`TypeBScheduler.update_multipliers` compatibility**: `TypeBScheduler` still implements the `update_multipliers` abstract method (e.g. as a no-op or to sync EMA scores) so that the epoch-end wiring in Step 4 does not need to branch on scheduler type. The slab-level logic lives in `build_next_slab`, which is only called from within the step loop.

3. **`base_interval` degeneracy at 1**: With `base_interval = 1`, `round(1 / exp(score))` evaluates to 0 for any positive score, which clamps to 1 — identical to a neutral image's interval. Hard images therefore receive no scheduling advantage over neutral ones, and the only differentiation comes from the weighted spare-capacity fill, which only activates when easy/very-easy images skip a slab. The default of `base_interval = ceil(dataset_size / slab_size)` avoids this and should be enforced with a warning if the user sets `base_interval = 1`.

4. **Cold-start ordering fairness**: Because epoch 0 visits images in a random but sequential order, images assigned to early slabs accumulate more overdue debt by the time scoring begins, causing them to appear disproportionately often in epoch 1. This self-corrects within 2–3 epochs as `next_due_slab` values spread across the timeline, but it means epoch 1 appearances per image are uneven even for images with identical difficulty scores. No action required; document as expected behaviour.

