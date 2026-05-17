"""
Unit tests for distributed bootstrap helpers (utils/distributed.py).
"""
import os
import unittest
from unittest.mock import patch

import torch
import torch.nn as nn

from utils.distributed import (
    is_distributed,
    get_rank,
    get_local_rank,
    get_world_size,
    is_main_process,
    shard_items,
    DDPWrapper,
)


class TestEnvironmentHelpers(unittest.TestCase):

    def _env(self, **kw):
        return patch.dict(os.environ, kw, clear=False)

    def test_not_distributed_world_size_1(self):
        with self._env(WORLD_SIZE="1"):
            self.assertFalse(is_distributed())

    def test_distributed_world_size_gt_1(self):
        with self._env(WORLD_SIZE="3"):
            self.assertTrue(is_distributed())

    def test_rank_from_env(self):
        with self._env(RANK="2"):
            self.assertEqual(get_rank(), 2)

    def test_rank_default_zero(self):
        env = {k: v for k, v in os.environ.items() if k != "RANK"}
        with patch.dict(os.environ, env, clear=True):
            self.assertEqual(get_rank(), 0)

    def test_local_rank_from_env(self):
        with self._env(LOCAL_RANK="1"):
            self.assertEqual(get_local_rank(), 1)

    def test_world_size_from_env(self):
        with self._env(WORLD_SIZE="4"):
            self.assertEqual(get_world_size(), 4)

    def test_main_process_rank_0(self):
        with self._env(RANK="0"):
            self.assertTrue(is_main_process())

    def test_not_main_process_rank_1(self):
        with self._env(RANK="1"):
            self.assertFalse(is_main_process())


class TestShardItems(unittest.TestCase):

    def test_two_ranks_even(self):
        items = list(range(10))
        self.assertEqual(shard_items(items, 0, 2), [0, 2, 4, 6, 8])
        self.assertEqual(shard_items(items, 1, 2), [1, 3, 5, 7, 9])

    def test_three_ranks(self):
        items = list(range(9))
        self.assertEqual(shard_items(items, 0, 3), [0, 3, 6])
        self.assertEqual(shard_items(items, 1, 3), [1, 4, 7])
        self.assertEqual(shard_items(items, 2, 3), [2, 5, 8])

    def test_union_exhaustive(self):
        for size in [1, 5, 10, 11, 100]:
            for ws in [1, 2, 3, 4]:
                items = list(range(size))
                union = sorted(x for r in range(ws) for x in shard_items(items, r, ws))
                self.assertEqual(union, items, f"size={size}, ws={ws}")

    def test_no_overlap(self):
        items = list(range(17))
        for ws in [2, 3, 4]:
            flat = [x for r in range(ws) for x in shard_items(items, r, ws)]
            self.assertEqual(len(flat), len(set(flat)), f"ws={ws}")

    def test_counts_balanced(self):
        for n, ws in [(7, 3), (10, 3), (9, 3), (8, 2)]:
            items = list(range(n))
            counts = [len(shard_items(items, r, ws)) for r in range(ws)]
            self.assertEqual(sum(counts), n)
            self.assertLessEqual(max(counts) - min(counts), 1)

    def test_single_rank_identity(self):
        items = list(range(5))
        self.assertEqual(shard_items(items, 0, 1), items)

    def test_empty_list(self):
        self.assertEqual(shard_items([], 0, 2), [])
        self.assertEqual(shard_items([], 1, 2), [])


class TestDDPWrapperAttrProxy(unittest.TestCase):
    """Verify DDPWrapper.__getattr__ falls through to the wrapped module."""

    def _make_fake_wrapper(self, module):
        """Build a DDPWrapper-like shell without a real distributed process group."""
        wrapper = object.__new__(DDPWrapper)
        object.__setattr__(wrapper, "_parameters", {})
        object.__setattr__(wrapper, "_buffers", {})
        object.__setattr__(wrapper, "_modules", {"module": module})
        object.__setattr__(wrapper, "_backward_hooks", {})
        object.__setattr__(wrapper, "_forward_hooks", {})
        object.__setattr__(wrapper, "_forward_pre_hooks", {})
        return wrapper

    def test_custom_attr_proxied(self):
        class M(nn.Module):
            custom_config = {"answer": 42}
            def forward(self, x): return x

        wrapper = self._make_fake_wrapper(M())
        # proxy via __getattr__ explicitly
        result = wrapper.__getattr__("custom_config")
        self.assertEqual(result, {"answer": 42})

    def test_missing_attr_raises(self):
        class M(nn.Module):
            def forward(self, x): return x

        wrapper = self._make_fake_wrapper(M())
        with self.assertRaises(AttributeError):
            wrapper.__getattr__("this_will_never_exist_xyz")

    def test_is_ddp_subclass(self):
        self.assertTrue(issubclass(DDPWrapper, torch.nn.parallel.DistributedDataParallel))


if __name__ == "__main__":
    unittest.main()

