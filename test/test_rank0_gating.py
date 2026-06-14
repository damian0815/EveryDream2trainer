"""
Unit tests for rank-0 gating utilities.

These tests verify that:
- is_main_process() returns correct values for different RANK env vars
- A rank-0-gated function executes only on rank 0
- Barrier is a no-op in non-distributed mode
- all_reduce_mean is a no-op in non-distributed mode (dist not initialized)
"""
import os
import unittest
from unittest.mock import patch, call, MagicMock

import torch

from utils.distributed import is_main_process, barrier, all_reduce_mean


class TestIsMainProcess(unittest.TestCase):

    def test_rank_0_is_main(self):
        with patch.dict(os.environ, {"RANK": "0"}, clear=False):
            self.assertTrue(is_main_process())

    def test_rank_1_not_main(self):
        with patch.dict(os.environ, {"RANK": "1"}, clear=False):
            self.assertFalse(is_main_process())

    def test_rank_2_not_main(self):
        with patch.dict(os.environ, {"RANK": "2"}, clear=False):
            self.assertFalse(is_main_process())

    def test_no_rank_env_defaults_to_main(self):
        env = {k: v for k, v in os.environ.items() if k != "RANK"}
        with patch.dict(os.environ, env, clear=True):
            self.assertTrue(is_main_process())


class TestRank0GatingPattern(unittest.TestCase):
    """Simulate the gating pattern used in train.py."""

    def _run_gated(self, rank: int, side_effect_fn):
        """Run side_effect_fn only when rank==0, mimicking train.py's _is_main guard."""
        call_log = []
        with patch.dict(os.environ, {"RANK": str(rank)}, clear=False):
            if is_main_process():
                side_effect_fn(call_log)
        return call_log

    def test_function_called_on_rank_0(self):
        calls = self._run_gated(0, lambda log: log.append("called"))
        self.assertEqual(calls, ["called"])

    def test_function_not_called_on_rank_1(self):
        calls = self._run_gated(1, lambda log: log.append("called"))
        self.assertEqual(calls, [])

    def test_function_not_called_on_rank_2(self):
        calls = self._run_gated(2, lambda log: log.append("called"))
        self.assertEqual(calls, [])


class TestBarrierNoop(unittest.TestCase):
    """barrier() is a no-op when not distributed (dist not initialized)."""

    def test_barrier_noop_without_pg(self):
        import torch.distributed as dist
        # If the process group is not initialised, barrier() should return without error
        if not dist.is_initialized():
            # Should not raise
            barrier()

    def test_all_reduce_mean_noop_without_pg(self):
        import torch.distributed as dist
        if not dist.is_initialized():
            t = torch.tensor(3.14)
            result = all_reduce_mean(t)
            self.assertAlmostEqual(result.item(), 3.14, places=5)


if __name__ == "__main__":
    unittest.main()

