import unittest
from itertools import permutations

import torch

from core.noise_permutation import (
    _build_cost_matrix,
    _pool_for_assignment,
    do_hungarian_noise_permutation,
    matched_noise_errors,
    quantize_int8,
)
from core.step import nibble_batch


def _reference_cost(clean_latents, noise, pool_size):
    """Recompute the exact int32 cost matrix the Hungarian operates on."""
    z_flat = _pool_for_assignment(clean_latents, pool_size)
    n_flat = _pool_for_assignment(noise, pool_size)
    b = z_flat.shape[0]
    denom = max(z_flat.abs().max().item(), n_flat.abs().max().item())
    if denom == 0:
        return torch.zeros((b, b), dtype=torch.int32)
    scale = 127.0 / denom
    z_q = quantize_int8(z_flat, scale).to(torch.int32)
    n_q = quantize_int8(n_flat, scale).to(torch.int32)
    z_sq = (z_q ** 2).sum(dim=1)
    n_sq = (n_q ** 2).sum(dim=1)
    return z_sq[:, None] + n_sq[None, :] - 2 * (z_q @ n_q.T)


def _brute_force_min_cost(cost):
    b = cost.shape[0]
    best = None
    for perm in permutations(range(b)):
        total = sum(cost[i, perm[i]].item() for i in range(b))
        if best is None or total < best:
            best = total
    return best


class TestQuantizeInt8(unittest.TestCase):
    def test_bounds(self):
        torch.manual_seed(0)
        t = torch.randn(64, 16) * 10
        scale = 127.0 / t.abs().max().item()
        q = quantize_int8(t, scale)
        self.assertEqual(q.dtype, torch.int8)
        self.assertTrue((q >= -127).all().item())
        self.assertTrue((q <= 127).all().item())

    def test_shared_scale_maps_max_to_127(self):
        t = torch.tensor([[0.0, -2.0, 5.0]])
        q = quantize_int8(t, 127.0 / 5.0)
        self.assertEqual(q[0, 2].item(), 127)
        self.assertEqual(q[0, 1].item(), -51)


class TestHungarianNoisePermutation(unittest.TestCase):
    def test_identity_for_batch_one(self):
        z = torch.randn(1, 4, 4, 4)
        n = torch.randn(1, 4, 4, 4)
        perm = do_hungarian_noise_permutation(z, n)
        self.assertEqual(perm.tolist(), [0])

    def test_valid_permutation_4d(self):
        torch.manual_seed(1)
        b, c, h, w = 8, 4, 16, 16
        z = torch.randn(b, c, h, w)
        n = torch.randn(b, c, h, w)
        perm = do_hungarian_noise_permutation(z, n, pool_size=4)
        self.assertEqual(perm.shape, (b,))
        self.assertEqual(perm.dtype, torch.int64)
        self.assertEqual(perm.device.type, "cpu")
        self.assertEqual(sorted(perm.tolist()), list(range(b)))

    def test_valid_permutation_5d_video(self):
        torch.manual_seed(2)
        b, c, f, h, w = 6, 4, 5, 16, 16
        z = torch.randn(b, c, f, h, w)
        n = torch.randn(b, c, f, h, w)
        perm = do_hungarian_noise_permutation(z, n, pool_size=4)
        self.assertEqual(perm.shape, (b,))
        self.assertEqual(sorted(perm.tolist()), list(range(b)))

    def test_optimality_against_brute_force(self):
        torch.manual_seed(3)
        b = 5
        z = torch.randn(b, 4, 4, 4)
        n = torch.randn(b, 4, 4, 4)
        pool_size = 4

        cost = _reference_cost(z, n, pool_size)
        best = _brute_force_min_cost(cost)

        perm = do_hungarian_noise_permutation(z, n, pool_size=pool_size)
        achieved = sum(cost[i, perm[i]].item() for i in range(b))
        self.assertEqual(achieved, best)

    def test_no_permutation_worse_than_identity(self):
        # Hungarian must never increase the sum-of-squared-distance vs. the identity.
        torch.manual_seed(4)
        b = 6
        z = torch.randn(b, 8, 8, 8)
        n = torch.randn(b, 8, 8, 8)
        pool_size = 2

        cost = _reference_cost(z, n, pool_size)
        identity_cost = sum(cost[i, i].item() for i in range(b))
        perm = do_hungarian_noise_permutation(z, n, pool_size=pool_size)
        matched_cost = sum(cost[i, perm[i]].item() for i in range(b))
        self.assertLessEqual(matched_cost, identity_cost)

    def test_all_zero_inputs_return_identity(self):
        z = torch.zeros(4, 4, 8, 8)
        n = torch.zeros(4, 4, 8, 8)
        perm = do_hungarian_noise_permutation(z, n)
        self.assertEqual(perm.tolist(), [0, 1, 2, 3])


class TestMatchedNoiseErrors(unittest.TestCase):
    def test_raw_is_identity_pairing(self):
        torch.manual_seed(6)
        z = torch.randn(4, 4, 8, 8)
        n = torch.randn(4, 4, 8, 8)
        raw, _ = matched_noise_errors(z, n, torch.arange(4))
        expected = torch.mean((z - n) ** 2).item()
        self.assertAlmostEqual(raw, expected)

    def test_opt_never_worse_than_raw(self):
        torch.manual_seed(7)
        b = 6
        z = torch.randn(b, 4, 8, 8)
        n = torch.randn(b, 4, 8, 8)
        perm = do_hungarian_noise_permutation(z, n, pool_size=4)
        raw, opt = matched_noise_errors(z, n, perm)
        self.assertLessEqual(opt, raw)

    def test_opt_matches_permuted_pairing(self):
        torch.manual_seed(8)
        z = torch.randn(4, 4, 8, 8)
        n = torch.randn(4, 4, 8, 8)
        perm = torch.tensor([2, 0, 3, 1])
        _, opt = matched_noise_errors(z, n, perm)
        expected = torch.mean((z - n[perm]) ** 2).item()
        self.assertAlmostEqual(opt, expected)


class TestBuildCostMatrix(unittest.TestCase):
    def test_chunked_matches_unchunked(self):
        torch.manual_seed(5)
        z_q = torch.randint(-127, 128, (10, 32), dtype=torch.int8)
        n_q = torch.randint(-127, 128, (10, 32), dtype=torch.int8)
        chunked = _build_cost_matrix(z_q, n_q, max_cost_bytes=32)
        full = _build_cost_matrix(z_q, n_q, max_cost_bytes=1 << 40)
        self.assertTrue(torch.equal(chunked, full))
        self.assertEqual(chunked.dtype, torch.int32)
        self.assertEqual(chunked.shape, (10, 10))


class TestNibbleSlicing(unittest.TestCase):
    def test_nibble_carries_precomputed_tensors(self):
        full_batch = {
            "image": torch.randn(4, 3, 8, 8),
            "runt_size": 0,
            "pathnames": ["p0", "p1", "p2", "p3"],
            "captions": {"default": ["c0", "c1", "c2", "c3"]},
            "clean_latents": torch.randn(4, 4, 2, 2),
            "noise": torch.randn(4, 4, 2, 2),
        }

        nibble, remainder = nibble_batch(full_batch, 3)

        self.assertTrue(
            torch.equal(nibble["clean_latents"], full_batch["clean_latents"][:3])
        )
        self.assertTrue(torch.equal(nibble["noise"], full_batch["noise"][:3]))
        self.assertTrue(
            torch.equal(remainder["clean_latents"], full_batch["clean_latents"][3:])
        )
        self.assertTrue(torch.equal(remainder["noise"], full_batch["noise"][3:]))

    def test_nibble_slicing_respects_runt(self):
        # With a runt (duplicated tail images), nibbles only cover the unique prefix,
        # so the attached tensors must be sliced the same way.
        full_batch = {
            "image": torch.randn(5, 3, 8, 8),
            "runt_size": 3,
            "pathnames": ["p0", "p1", "p2", "p3", "p4"],
            "clean_latents": torch.randn(5, 4, 2, 2),
            "noise": torch.randn(5, 4, 2, 2),
        }

        nibble, remainder = nibble_batch(full_batch, 2)

        self.assertTrue(
            torch.equal(nibble["clean_latents"], full_batch["clean_latents"][:2])
        )
        self.assertTrue(torch.equal(nibble["noise"], full_batch["noise"][:2]))
        self.assertTrue(
            torch.equal(remainder["clean_latents"], full_batch["clean_latents"][2:3])
        )
        self.assertTrue(torch.equal(remainder["noise"], full_batch["noise"][2:3]))


if __name__ == "__main__":
    unittest.main()
