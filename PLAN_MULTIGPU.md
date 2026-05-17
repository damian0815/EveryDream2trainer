# Multi-GPU Plan (Throughput-Focused)

## Goals and constraints

- Hardware target: `2x24GB` or `3x24GB`
- Primary goal: faster training throughput (not model sharding)
- Current model already fits on one GPU in bf16
- Teacher model stays enabled
- EMA can be de-prioritized (or rank-0 only)

## Recommendation

Use **DDP (DistributedDataParallel)** with `torchrun`, not FSDP/ZeRO.

Why this matches your case:

- DDP gives near-linear speedup when the model fits per GPU
- Lowest implementation risk for this codebase vs sharded strategies
- Keeps checkpoint format and most training internals intact
- Avoids heavy refactors required by FSDP/ZeRO around optimizer/checkpoint/EMA

Expected scaling (typical, not guaranteed):

- `2 GPUs`: ~`1.6x-1.9x`
- `3 GPUs`: ~`2.2x-2.8x`

## Non-goals (for this phase)

- No FSDP / ZeRO Stage 3 model sharding
- No major rewrite of `EveryDreamOptimizer`
- No format change to existing checkpoints

## Design decisions

1. **Launcher**: use `torchrun --nproc_per_node=N train.py ...`
2. **Distributed init**: initialize `torch.distributed` from `RANK`, `LOCAL_RANK`, `WORLD_SIZE`
3. **Per-rank device**: `cuda:{LOCAL_RANK}` overrides `--gpuid` during distributed runs
4. **Data split**: shard `image_train_items` by rank before building `DataLoaderMultiAspect`
5. **Model wrapping**:
   - Wrap trainable modules in `DDP` (UNet, text encoder(s) if trainable)
   - Keep teacher frozen and local on each rank
6. **Rank-0 ownership**: only rank 0 does logging/sampling/validation/checkpoint writes
7. **Sync points**: barriers around save/validation epoch boundaries

## Implementation checklist

### Phase 1 - Distributed bootstrap (`train.py`)

- [ ] Add helper(s):
  - [ ] `is_distributed = int(os.environ.get("WORLD_SIZE", "1")) > 1`
  - [ ] `init_distributed()` and `cleanup_distributed()`
  - [ ] `is_main_process = (rank == 0)`
- [ ] When distributed:
  - [ ] `torch.cuda.set_device(local_rank)`
  - [ ] use `device = torch.device(f"cuda:{local_rank}")`
  - [ ] ignore `--gpuid` for process placement

### Phase 2 - Data sharding with existing pipeline

- [ ] After `image_train_items` is built (and after optional validation split prep), shard:
  - [ ] `image_train_items = image_train_items[rank::world_size]`
- [ ] Keep `DataLoaderMultiAspect` unchanged (rank gets independent subset)
- [ ] Log per-rank sample counts on startup for sanity

### Phase 3 - DDP module wrapping

- [ ] Wrap trainable model parts with `DistributedDataParallel`
- [ ] Keep access paths stable for code that expects `.config`, `.dtype`, etc.
- [ ] If needed, add lightweight forward indirection (e.g., `unet_for_train_forward`) to avoid breaking attribute usage patterns

### Phase 4 - Rank-0 gating and synchronization

- [ ] Guard to rank 0 only:
  - [ ] `wandb` init
  - [ ] TensorBoard writer output
  - [ ] sample generation
  - [ ] validation runs
  - [ ] checkpoint saving (`save_model`, `save_model_lora`)
- [ ] Add `dist.barrier()` where needed so non-zero ranks do not race ahead around validation/save boundaries

### Phase 5 - Numerical and logging consistency

- [ ] Aggregate key scalar logs across ranks via `all_reduce(..., AVG)` (loss, throughput)
- [ ] Keep gradient accumulation behavior unchanged first; optimize comms later if needed

### Phase 6 - Exit safety

- [ ] Ensure SIGINT/stop path does not trigger duplicate saves from non-main ranks
- [ ] Call distributed cleanup on normal and exceptional exits

## Minimal command changes

Single GPU (unchanged):

```bash
python train.py --config train.json
```

Multi-GPU (2 GPUs):

```bash
torchrun --nproc_per_node=2 train.py --config train.json
```

Multi-GPU (3 GPUs):

```bash
torchrun --nproc_per_node=3 train.py --config train.json
```

Notes:

- Your JSON + CLI override flow remains valid under `torchrun`; each rank receives the same args.
- If needed on macOS shell sessions, set allocator/env flags in the same command line as today.

## Validation and acceptance criteria

- [ ] Functional:
  - [ ] Run 1 epoch on `--nproc_per_node=2` without crashes
  - [ ] Rank 0 is the only process writing checkpoints/samples/validation artifacts
- [ ] Correctness:
  - [ ] Loss curve trend is comparable to single-GPU baseline
  - [ ] Checkpoint can resume in single-GPU mode
- [ ] Performance:
  - [ ] Throughput improves materially vs single GPU at similar effective batch settings

## Risks and mitigations

- **Risk**: DDP wrapper breaks code paths assuming raw module attributes
  - **Mitigation**: keep raw module references for metadata; use wrapper only for forward
- **Risk**: Uneven data split with custom bucket logic
  - **Mitigation**: rank-wise item counts/logging + shuffled slicing by stride (`rank::world_size`)
- **Risk**: Duplicate side effects from non-zero ranks
  - **Mitigation**: strict rank-0 gating + barriers

## Future optimization (optional)

- Add `no_sync()` during intermediate micro-backward passes to reduce DDP comm overhead in nibble/accumulation loops
- Consider ZeRO Stage 2 only if optimizer state becomes limiting on 24GB cards
- Add a convenience script (e.g., `scripts/train_ddp.sh`) for repeatable launch defaults

