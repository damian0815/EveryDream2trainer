# FIX_DDP_DEADLOCK_PLAN.md

## 1. CURRENT LOGIC

### Training loop skeleton

```
train.py  outer loop
  for epoch in epochs:
    for step, full_batch in enumerate(train_dataloader):        ← heterogeneous batches

      # --- RESOLUTION CHANGE CHECK (train.py ~L1355) ---
      if max_backward_slice_size <= accumulated_loss_images_count:
          optimizer_backward(..., 'truncated backward', no_sync_ctx=ddp_no_sync_ctx(...))

      # --- EMERGENCY BACKWARD (train.py ~L1379) ---
      if max_safe_forward_size == 0 and accumulated_loss_images_count > 0:
          optimizer_backward(..., 'emergency backward', no_sync_ctx=ddp_no_sync_ctx(...))

      train_step(full_batch, ...)

        ← inside train_step (core/step.py):

        while remaining_batch not empty:
          nibble = nibble_batch(...)         ← variable-size nibble from heterogeneous data

          for caption_variant in caption_variants:
            forward()
            loss = compute_loss()
            tv.accumulate_loss(loss)         ← tv.accumulated_loss_images_count grows

            # PRE-EMPTIVE BACKWARD (core/step.py ~L143)
            if accumulated_loss_images_count + latents.shape[0] > max_backward_slice_size:
                optimizer_backward(..., 'pre-emptive backward', no_sync_ctx=ddp_no_sync_ctx(...))

            should_step_optimizer = (backwarded + accumulated >= desired_effective_batch_size)

            # REGULAR BACKWARD (core/step.py ~L355)
            if should_step_optimizer and accumulated > 0:
                optimizer_backward(..., 'regular backward', no_sync_ctx=ddp_no_sync_ctx(...))

            # OPTIMIZER STEP (core/step.py ~L372)
            if should_step_optimizer and backwarded > 0:
                ed_optimizer.step_optimizer(...)
                    ← calls sync_ddp_gradients(unet, te, te2)  ← MANUAL ALL-REDUCE
                    ← then optimizer.step()
```

### How `no_sync` is currently used

Every individual call to `optimizer_backward()` constructs a **fresh** `ddp_no_sync_ctx(...)` context manager and passes it in.  Inside `optimizer_backward()`:

```python
# core/step.py optimizer_backward()
ctx = no_sync_ctx if no_sync_ctx is not None else contextlib.nullcontext()
with ctx:                        # ← ENTER no_sync
    optimizer.backward(tv.accumulated_loss)
# ← EXIT no_sync  (DDP hooks re-enabled here)
```

`ddp_no_sync_ctx` (in `utils/distributed.py`) enters `module.no_sync()` on every DDP-wrapped module (unet, text_encoder, text_encoder_2 where applicable) for the duration of the `with` block.

After all backward passes for a step are done, `step_optimizer()` calls `sync_ddp_gradients(unet, te, te2)` which manually `dist.all_reduce`s every parameter gradient before `optimizer.step()`.

### Why this deadlocks

The key invariant for DDP safety is: **every collective operation (all-reduce, broadcast, barrier) must be entered by ALL ranks simultaneously**.

Because the dataset is heterogeneous, different resolution buckets produce different numbers of nibbles, and `nibble_batch` is guided by `get_nibble_size()` which factors in `accumulated_loss_images_count`, `backwarded_images_count`, and `max_backward_slice_size`.  This means:

- Rank 0 may decide to call `backward()` after accumulating N images.
- Rank 1 may not reach the same threshold at the same dataloader iteration because its shard has a different mix of resolutions.

The mismatch happens on **context-enter / context-exit** of `no_sync()`, *not* purely on the all-reduce.  PyTorch's DDP `no_sync()` implementation may schedule a *reducer* flush on the way out of the context (when `finalize_backward()` / `_sync_params_and_buffers` or related hooks are reset), and that flush is a collective.  Even if no flush fires immediately, the *re-arming* of allreduce hooks at context exit means the very next `backward()` on one rank can fire an all-reduce that the other rank is not executing.  When the other rank is still in a **different** state of the loop (different step index), it never comes to participate, and both ranks stall.

Concretely, from the deadlock log:

```
(rank 0 step=13, rank 1 step=14)          ← ranks diverged
optimizer backward rank 1 entered context  ← rank 1 inside no_sync
                                           ← rank 0 never reaches this point
** DEADLOCK **
```

Rank 1 called `backward()` (with a fresh `no_sync()` context it already entered), but the DDP all-reduce that was re-armed when the *previous* `no_sync()` exited is now triggered by rank 1's backward – and rank 0 is not there to meet it.

### Sites where `optimizer_backward` is called (all currently use per-call `ddp_no_sync_ctx`)

| Location | Label | Notes |
|---|---|---|
| `train.py` ~L1357 | `'truncated backward: '` | Resolution changes mid-accumulation |
| `train.py` ~L1379 | `'emergency backward: '` | OOM prevention |
| `core/step.py` ~L143 | `'pre-emptive backward @…: '` | Prevent exceeding `max_backward_slice_size` |
| `core/step.py` ~L355 | `'regular backward: '` | Main backward before optimizer step |

---

## 2. CHANGES REQUIRED

### Core principle

Replace the **per-call** `no_sync()` wrapping with a **persistent** `no_sync()` context that spans the entire training loop.  The only time gradients are allowed to synchronise is immediately before `optimizer.step()`, and that step is triggered whenever **any** rank is ready, via `dist.all_reduce(MAX)`.

### Summary of changes

1. **Remove `no_sync_ctx` argument from all `optimizer_backward()` call sites.**  
   `optimizer_backward` itself becomes context-agnostic: it simply calls `optimizer.backward(accumulated_loss)` without any no_sync wrapping.

2. **Introduce a persistent no_sync context manager** that is entered once before the epoch loop and exited at the very end of training.

3. **Introduce a `want_to_step` all-reduce(MAX)** called on every dataloader iteration (same count on every rank — guaranteed by equal-length per-rank dataloaders). Each rank votes 1 if it is locally ready to step **or** if this is the final step of the final epoch; 0 otherwise. If any rank votes 1, all ranks step together. The "final step" vote ensures that a rank sitting at the all_reduce on its last iteration is never left hanging by a peer that has already exited its loop.

4. **Exit the persistent no_sync context around `step_optimizer()`**, re-enter it immediately afterward.  At this point DDP's hooks are re-enabled, `sync_ddp_gradients()` fires the real all-reduce (all ranks participate), then `optimizer.step()` runs, then `no_sync` is re-entered.

5. **`sync_ddp_gradients()` in `step_optimizer()` remains**, but now it fires inside a brief window where `no_sync` is NOT active.  No other changes to `optimizers.py` are needed.

6. **All intermediate backward calls** (truncated, emergency, pre-emptive, regular) continue to fire without any no_sync wrapping at the call site; the persistent outer context suppresses all-reduces for them automatically.

7. **Post-step bookkeeping** (`backwarded_images_count = 0`, `optimizer_step += 1`, etc.) is only executed by ranks that locally had `should_step_optimizer == True` and `backwarded_images_count > 0`.  A rank that was force-stepped (by another rank's MAX signal or by `is_final_step`) still participates in the collectives inside `step_optimizer` but skips `optimizer.step()` and the accounting reset when it has nothing to step on.

### Pseudocode for the new control flow

```python
# --- BEFORE EPOCH LOOP ---
no_sync_stack = DDPPersistentNoSync(model.unet, model.text_encoder, ...)
no_sync_stack.enter()

for epoch in epochs:
    epoch_steps = list(enumerate(train_dataloader))
    last_step_idx = len(epoch_steps) - 1
    for step, full_batch in epoch_steps:
        is_final_step = (epoch == last_epoch) and (step == last_step_idx)

        # --- resolution change / truncated backward ---
        if max_backward_slice_size <= accumulated_loss_images_count:
            optimizer_backward(...)       # no no_sync_ctx arg; outer context covers it

        train_step(..., is_final_step=is_final_step)
        # train_step internally does the want_to_step MAX gate (see step 4b below)

# --- AFTER EPOCH LOOP ---
no_sync_stack.exit()
dist.barrier()   # clean shutdown before checkpoint save
```

Inside `train_step`, replacing the current `if should_step_optimizer` block:

```python
# Force vote=1 on the final step of training so that a peer sitting at this
# all_reduce (because it fired MAX for a mid-training step) is never stranded.
is_final_step = tv.is_final_step   # set by train.py before calling train_step
vote = int(is_final_step or (should_step_optimizer and tv.backwarded_images_count > 0))
want_to_step = torch.tensor(vote, dtype=torch.int32, device=model.unet.device)
if _is_distributed:
    dist.all_reduce(want_to_step, op=dist.ReduceOp.MAX)
any_rank_wants_to_step = want_to_step.item() > 0

if any_rank_wants_to_step:
    tv.ddp_no_sync.exit()            # brief window: hooks re-enabled for sync
    try:
        # step_optimizer is always called so all ranks participate in
        # sync_ddp_gradients (a collective); the actual optimizer.step()
        # inside it is guarded by backwarded_images_count > 0.
        ed_optimizer.step_optimizer(tv.global_step, tv, log_data=log_data)
    finally:
        tv.ddp_no_sync.enter()       # re-suppress immediately

    # Bookkeeping only for ranks that were locally ready
    if should_step_optimizer and tv.backwarded_images_count > 0:
        tv.last_effective_batch_size = tv.backwarded_images_count
        tv.total_trained_samples_count += tv.backwarded_images_count
        tv.optimizer_step += 1
        tv.current_timestep_interval = None
        tv.backwarded_images_count = 0
        # ... interleaved_bs1_count logic, desired_effective_batch_size update ...
        did_step_optimizer_cb()
```

### What does NOT change

- `sync_ddp_gradients()` inside `step_optimizer()` — stays exactly as-is.
- Gradient clipping, scaler unscale — unchanged.
- `optimizer.zero_grad()` — unchanged.
- All loss accumulation, nibble logic, forward pass slicing — unchanged.

---

## 3. IMPLEMENTATION PLAN

### Step 1 — Add `DDPPersistentNoSync` to `utils/distributed.py`

Add a new class whose `enter()`/`exit()` can be called imperatively across the training loop:

```python
class DDPPersistentNoSync:
    """
    Persistent no_sync context that can be manually entered/exited multiple
    times.  Used to keep DDP gradient all-reduce suppressed across many
    backward() calls, exiting only for the brief window around optimizer.step().
    """

    def __init__(self, *modules):
        self._ddp_mods = [m for m in modules
                          if m is not None and isinstance(m, DistributedDataParallel)]
        self._ctxs: list = []

    def enter(self):
        if not self._ddp_mods:
            return
        self._ctxs = [m.no_sync() for m in self._ddp_mods]
        for ctx in self._ctxs:
            ctx.__enter__()
        logging.debug(f"[rank {get_rank()}] DDPPersistentNoSync: entered ({len(self._ctxs)} modules)")

    def exit(self):
        if not self._ctxs:
            return
        for ctx in reversed(self._ctxs):
            ctx.__exit__(None, None, None)
        self._ctxs = []
        logging.debug(f"[rank {get_rank()}] DDPPersistentNoSync: exited")

    # Allow use as a regular context manager too (e.g. in tests)
    def __enter__(self):
        self.enter()
        return self

    def __exit__(self, *args):
        self.exit()
```

### Step 2 — Guard `optimizer.step()` in `step_optimizer()` by `backwarded_images_count`

A rank force-stepped by another rank's MAX vote (or by `is_final_step`) must still call `sync_ddp_gradients` (a collective requiring all ranks), but must NOT update weights with zero/empty gradients.  Add a `tv` check around the actual `optimizer.step()` call:

```python
# optimizers.py  step_optimizer()
def step_optimizer(self, global_step, tv, log_data):
    if self.scaler is not None:
        for optimizer in self.optimizers:
            self.scaler.unscale_(optimizer)

    # Gradient sync — collective, always called by all ranks in the sync window.
    sync_ddp_gradients(self.unet, self.text_encoder, self.text_encoder_2)

    # ... grad norm logging, clipping (unchanged) ...

    # Guard: skip weight update if this rank contributed no gradients this step.
    if tv.backwarded_images_count == 0:
        self.zero_grad(set_to_none=True)
        return

    # ... existing scaler.step / optimizer.step / zero_grad logic (unchanged) ...
```

### Step 3 — Remove `no_sync_ctx` from `optimizer_backward()` in `core/step.py`

```python
# Before:
def optimizer_backward(optimizer, tv, plugin_runner, log_hint='', no_sync_ctx=None):
    ctx = no_sync_ctx if no_sync_ctx is not None else contextlib.nullcontext()
    with ctx:
        optimizer.backward(tv.accumulated_loss)

# After:
def optimizer_backward(optimizer, tv, plugin_runner, log_hint=''):
    # Caller maintains a persistent no_sync context; no wrapping needed here.
    optimizer.backward(tv.accumulated_loss)
```

### Step 4 — Remove `no_sync_ctx` kwarg from all four call sites

| File | Line (approx) | Change |
|---|---|---|
| `train.py` | ~L1357 | Remove `no_sync_ctx=ddp_no_sync_ctx(...)` from `truncated backward` call |
| `train.py` | ~L1379 | Remove `no_sync_ctx=ddp_no_sync_ctx(...)` from `emergency backward` call |
| `core/step.py` | ~L143 | Remove `no_sync_ctx=ddp_no_sync_ctx(...)` from `pre-emptive backward` call |
| `core/step.py` | ~L355 | Remove `no_sync_ctx=ddp_no_sync_ctx(...)` from `regular backward` call |

### Step 5 — Thread `DDPPersistentNoSync` and `is_final_step` into the training loop in `train.py`

#### 5a. Create the persistent context before the epoch loop

```python
from utils.distributed import DDPPersistentNoSync

_ddp_no_sync = DDPPersistentNoSync(
    model.unet,
    model.text_encoder,
    getattr(model, 'text_encoder_2', None)
)
_ddp_no_sync.enter()
```

#### 5b. Compute `is_final_step` and pass it into `train_step`

```python
for epoch in range(args.max_epochs):
    for step, full_batch in enumerate(train_dataloader):
        is_final_step = (epoch == args.max_epochs - 1) and (step == epoch_len - 1)
        tv.is_final_step = is_final_step   # or pass as a kwarg to train_step

        train_step(..., tv=tv, ...)
```

`max_steps` early-exit: if `args.max_steps` is set, `is_final_step` should also be `True` when `tv.global_step + 1 >= args.max_steps`.

#### 5c. Implement the `want_to_step` gate inside `train_step` (replaces the current `if should_step_optimizer` block)

See pseudocode in Section 2 above.

#### 5d. Store `_ddp_no_sync` on `TrainingVariables`

```python
# model/training_model.py  TrainingVariables dataclass
ddp_no_sync: Optional['DDPPersistentNoSync'] = None
is_final_step: bool = False
```

```python
# train.py, after creating tv:
tv.ddp_no_sync = _ddp_no_sync
```

#### 5e. Clean exit after the training loop

```python
_ddp_no_sync.exit()
if _is_distributed:
    dist.barrier()
save_checkpoint(...)
```

### Step 6 — Handle `dist.is_initialized()` guard in `DDPPersistentNoSync`

If `self._ddp_mods` is empty (single-GPU), `enter()` and `exit()` are both no-ops.  No changes needed for single-GPU code paths.  The `want_to_step` tensor is still constructed but the `dist.all_reduce` is skipped; `any_rank_wants_to_step` equals the local vote directly.

### Step 7 — Handle the `interleaved_bs1_count` path

The `interleaved_bs1_count` path unconditionally sets `should_step_optimizer = True`.  It participates in the `want_to_step` vote the same way as the regular path — no special casing needed.

### Step 8 — Update docstrings and comments

- `optimizer_backward`: remove documentation about `no_sync_ctx`.
- `ddp_no_sync_ctx`: update to note it is superseded by `DDPPersistentNoSync` for the main loop; keep for any one-off uses.
- `sync_ddp_gradients`: note it must be called inside the brief window where `DDPPersistentNoSync` has been exited (guaranteed by the `want_to_step` gate).

---

## Risk notes and edge cases

| Scenario | Handling |
|---|---|
| One rank not yet ready when another fires MAX | `all_reduce(MAX)=1` → all ranks step. The lagging rank's `backwarded_images_count == 0` (or > 0 but less than its target); `step_optimizer` is called (collective), but the early-exit guard skips `optimizer.step()` for that rank. Counters are not reset; the rank continues accumulating from where it was. |
| Both ranks have `backwarded_images_count == 0` | Vote is `(0, 0)`, MAX = 0 → no step, no collectives. Safe. |
| One rank exits its epoch loop while the other is at `all_reduce(MAX)` | On its **last step of the last epoch**, every rank forces `vote=1` via `is_final_step`. This ensures the peer sitting at the all_reduce receives MAX=1 and can proceed. After that final step both ranks exit their loops cleanly. |
| `is_final_step` fires but this rank has `backwarded_images_count == 0` | Still calls `step_optimizer` (to participate in `sync_ddp_gradients`), but the early-exit guard skips `optimizer.step()`. No weight update, no counter reset. |
| OOM inside `step_optimizer` after `exit()` | The `finally` block re-enters `no_sync` unconditionally to avoid leaving the context in an undefined state. |
| `truncated`/`emergency` backwards in `train.py` before `train_step` | Covered by the persistent outer context; no special treatment needed. |
| Epoch boundary with un-stepped gradients | The `is_final_step` flush handles end-of-training. Between epochs the no_sync context stays open; the regular `want_to_step` gate on the first step of the new epoch will pick up any leftover gradients. |
| `did_step_optimizer_cb` callback | Called only on ranks where `should_step_optimizer` was locally True and `backwarded_images_count > 0`. |
| `interleaved_bs1_count` path | Sets `should_step_optimizer = True` unconditionally; participates in the MAX vote the same as the regular path. |
| `max_steps` early exit | `is_final_step` must also be set to `True` when `tv.global_step + 1 >= args.max_steps` to handle the case where training ends before the last epoch completes. |

