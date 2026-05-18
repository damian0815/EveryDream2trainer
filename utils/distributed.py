"""
Distributed training utilities for multi-GPU DDP training.

Usage (single GPU – unchanged):
    python train.py --config train.json

Usage (multi-GPU, 2 GPUs):
    torchrun --nproc_per_node=2 train.py --config train.json
"""
import logging
import os
from contextlib import contextmanager, ExitStack
from enum import Enum
from typing import Optional

import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel


# ---------------------------------------------------------------------------
# Core helpers
# ---------------------------------------------------------------------------

def is_distributed() -> bool:
    """Returns True when running inside a torchrun / multi-process launch."""
    return int(os.environ.get("WORLD_SIZE", "1")) > 1


def get_rank() -> int:
    """Global rank of this process (0 in single-GPU mode)."""
    return int(os.environ.get("RANK", "0"))


def get_local_rank() -> int:
    """Local rank on this node (0 in single-GPU mode)."""
    return int(os.environ.get("LOCAL_RANK", "0"))


def get_world_size() -> int:
    """Total number of processes (1 in single-GPU mode)."""
    return int(os.environ.get("WORLD_SIZE", "1"))


def is_main_process() -> bool:
    """True only on global rank 0."""
    return get_rank() == 0


# ---------------------------------------------------------------------------
# Lifecycle helpers
# ---------------------------------------------------------------------------

def init_distributed() -> tuple[int, int, int]:
    """
    Initialise torch.distributed and pin this process to its local GPU.

    Returns:
        (rank, local_rank, world_size)
    """
    rank = get_rank()
    local_rank = get_local_rank()
    world_size = get_world_size()

    if not dist.is_initialized():
        dist.init_process_group(backend="nccl")
        torch.cuda.set_device(local_rank)

    logging.info(
        f"[rank {rank}/{world_size}] Distributed init complete – device cuda:{local_rank}"
    )
    return rank, local_rank, world_size


def cleanup_distributed() -> None:
    """Destroy the process group on normal or exceptional exit."""
    if dist.is_initialized():
        dist.destroy_process_group()


# ---------------------------------------------------------------------------
# Collective helpers
# ---------------------------------------------------------------------------

def barrier() -> None:
    """Synchronise all ranks.  No-op in single-GPU mode."""
    if dist.is_initialized():
        dist.barrier()


class StateSignal(Enum):
    ACCUMULATING = 0  # still processing training items, not yet ready to optimizer.step()
    READY = 1         # ready to optimizer.step()
    DONE = 2          # no more data available

def get_distributed_state_signal(local_signal: StateSignal, device) -> StateSignal:
    """
    Get state signal from all ranks across multiple GPUs.
    | Rank 0 | Rank 1 | Rank 2 | Global signal (all ranks) | Interpretation |
    | 0 | 1 | 2 | 0 | 0 is still accumulating (everybody else no-op then ask again) |
    | 1 | 1 | 2 | 1 | everybody is ready or done -> step optimizer |
    | 2 | 2 | 2 | 2 | everybody is done -> exit training loop |
    | 0 | 0 | 2 | 0 | one done, others still working -> 0 and 1 continue, 2 should no-op |
    """
    sig = torch.tensor(local_signal, dtype=torch.int, device=device)
    dist.all_reduce(sig, op=dist.ReduceOp.MIN)
    global_sig = sig.item()
    return global_sig


def all_reduce_mean(tensor: torch.Tensor) -> torch.Tensor:
    """
    All-reduce *tensor* across all ranks using mean aggregation.
    Returns the averaged tensor.  No-op in single-GPU mode.
    """
    if not dist.is_initialized():
        return tensor
    dist.all_reduce(tensor, op=dist.ReduceOp.AVG)
    return tensor


# ---------------------------------------------------------------------------
# DDP gradient-accumulation helpers
# ---------------------------------------------------------------------------

@contextmanager
def ddp_no_sync_ctx(*modules):
    """
    Context manager that puts every DistributedDataParallel module in
    ``no_sync()`` mode simultaneously.

    .. deprecated::
        For the main training loop use :class:`DDPPersistentNoSync` instead,
        which holds no_sync open across many backward passes and only exits it
        for the brief window around ``optimizer.step()``.  This one-shot helper
        is kept for any isolated one-off uses (e.g. tests).

    Non-DDP modules (and ``None``) are silently ignored, making this safe
    to call in single-GPU mode.
    """
    ddp_mods = [m for m in modules
                if m is not None and isinstance(m, DistributedDataParallel)]
    if not ddp_mods:
        yield
        return
    prefix = f'ddp_no_sync_ctx rank {get_rank()}'
    with ExitStack() as stack:
        logging.debug(f"{prefix} entering contexts on {len(ddp_mods)} DDP modules")
        for m in ddp_mods:
            stack.enter_context(m.no_sync())
        yield


class DDPPersistentNoSync:
    """
    Persistent no_sync context that can be manually entered and exited
    multiple times across the training loop.

    Enter once before the epoch loop and exit once at the very end of
    training.  The only time DDP gradient all-reduces are allowed to fire
    is inside the brief window around ``optimizer.step()``:

    .. code-block:: python

        no_sync.exit()          # re-arm DDP hooks
        try:
            step_optimizer(...)  # sync_ddp_gradients fires here
        finally:
            no_sync.enter()     # suppress again immediately

    Non-DDP modules (and ``None``) are silently ignored, making this
    a no-op in single-GPU mode.  Can also be used as a regular context
    manager (``with DDPPersistentNoSync(...): ...``).
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


def sync_ddp_gradients(*modules) -> None:
    """
    Manually all-reduce (average) gradients across all ranks for every
    DDP-wrapped module in *modules*.

    **Must** be called inside the brief window where
    :class:`DDPPersistentNoSync` has been exited (guaranteed by the
    ``want_to_step`` gate in the training loop).  All ranks must enter
    this function simultaneously.

    Non-DDP modules (and ``None``) are silently ignored.  No-op in
    single-GPU mode.

    The all-reduce uses ``ReduceOp.AVG``, matching DDP's default behaviour.
    """
    if not dist.is_initialized():
        return
    for m in modules:
        if m is None or not isinstance(m, DistributedDataParallel):
            continue
        for p in m.parameters():
            if p.grad is not None:
                dist.all_reduce(p.grad, op=dist.ReduceOp.AVG)


# ---------------------------------------------------------------------------
# Data sharding
# ---------------------------------------------------------------------------

def shard_items(items: list, rank: int, world_size: int) -> list:
    """
    Stride-sample *items* so rank *rank* gets every *world_size*-th element.

    Properties:
    - Non-overlapping across ranks.
    - Union over all ranks == original list.
    - Each rank gets floor(len/world_size) or ceil(len/world_size) items.

    Example::
        shard_items([0,1,2,3,4], rank=1, world_size=2)
        # → [1, 3]
    """
    return items[rank::world_size]


# ---------------------------------------------------------------------------
# DDP module wrapper
# ---------------------------------------------------------------------------

class DDPWrapper(torch.nn.parallel.DistributedDataParallel):
    """
    Thin DDP subclass that transparently proxies *any* attribute (including
    diffusers model extras like ``.config``, ``.dtype``, ``.yaml`` …) to the
    underlying module.

    This lets the rest of the codebase continue to use::

        model.unet.config           # still works
        model.unet.dtype            # still works
        model.unet(latents, ...)    # goes through DDP (gradient sync)

    without knowing whether ``model.unet`` is wrapped or not.
    """

    def __getattr__(self, name: str):
        try:
            # Let nn.Module / DDP handle its own registered attrs first
            # (parameters, buffers, submodules, DDP internals …)
            return super().__getattr__(name)
        except AttributeError:
            # Fall through to the wrapped module for everything else
            return getattr(self.module, name)

