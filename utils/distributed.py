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


class StateSignal(int, Enum):
    WANTS_STEP = 1    # ready to optimizer.step()
    DONE = 2          # no more data available

def get_distributed_state_signal(local_signal: StateSignal, device) -> StateSignal:
    if not dist.is_initialized():
        return local_signal
    print(f"rank {get_rank()} sending local signal {local_signal}")
    sig = torch.tensor(local_signal, dtype=torch.int, device=device)
    dist.all_reduce(sig, op=dist.ReduceOp.MAX)
    global_sig = sig.item()
    print(f"rank {get_rank()} got global signal {global_sig}")
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

    Non-DDP modules (and ``None``) are silently ignored, making this safe
    to call in single-GPU mode.
    """
    ddp_mods = [m for m in modules
                if m is not None and isinstance(m, DistributedDataParallel)]
    if not ddp_mods:
        yield
        return
    with ExitStack() as stack:
        for m in ddp_mods:
            stack.enter_context(m.no_sync())
        yield


def sync_ddp_gradients(*modules) -> None:
    """
    Manually all-reduce (average) gradients across all ranks for every
    DDP-wrapped module in *modules*.

    Non-DDP modules (and ``None``) are silently ignored.  No-op in
    single-GPU mode.

    The all-reduce uses ``ReduceOp.AVG``, matching DDP's default behaviour.
    """
    if not dist.is_initialized():
        return
    print(f"rank {get_rank()} syncing grads before optimizer step")
    for m in modules:
        if m is None or not isinstance(m, DistributedDataParallel):
            continue
        for p in m.parameters():
            if p.grad is not None:
                dist.all_reduce(p.grad, op=dist.ReduceOp.SUM)


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

