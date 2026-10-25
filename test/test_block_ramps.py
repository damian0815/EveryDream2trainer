"""
Unit tests for per-block LR ramps -- piecewise-linear alpha keypoints on SANA transformer blocks.

Alpha at each block is linearly interpolated between adjacent keypoints;
blocks outside the keypoint range clamp to the nearest keypoint's alpha.
Param LR = its source group's LR x alpha.
"""
import unittest

import torch

from optimizer.per_unet_layer_scales import (
    _get_block_index_from_key,
    _get_ramp_blockwise_alpha,
    apply_block_ramps,
    parse_lr_block_ramps,
)


def _make_params(names):
    return [(n, torch.nn.Parameter(torch.zeros(1))) for n in names]


def _make_single_group(parameters, lr=1e-4, name="single"):
    return [{"name": name, "params": [p for _, p in parameters], "lr": lr,
             "betas": (0.9, 0.999), "weight_decay": 0.01}]


def _transformer_names(num_blocks=20, params_per_block=1):
    names = []
    for i in range(num_blocks):
        for j in range(params_per_block):
            names.append(f"transformer_blocks.{i}.attn.q.weight")
    return names


def _sana_fixture(num_blocks=6):
    names = _transformer_names(num_blocks=num_blocks, params_per_block=1)
    names.extend([
        "scale_shift_table",
        "caption_projection.linear_1.weight",
        "time_embed.linear_1.weight",
        "patch_embed.proj.weight",
        "proj_out.weight",
    ])
    return names


def _lr_by_name(result, params):
    lrs = {}
    for g in result:
        for p in g["params"]:
            idx = next(i for i, (_, pp) in enumerate(params) if id(p) == id(pp))
            lrs[params[idx][0]] = g["lr"]
    return lrs


class TestParseLrBlockRamps(unittest.TestCase):

    def test_none_returns_empty(self):
        self.assertEqual(parse_lr_block_ramps(None), [])

    def test_pairs(self):
        self.assertEqual(parse_lr_block_ramps([0, 0, 5, 1]), [(0, 0.0), (5, 1.0)])

    def test_unsorted_pairs_sorted(self):
        self.assertEqual(parse_lr_block_ramps([5, 1, 0, 0]), [(0, 0.0), (5, 1.0)])

    def test_odd_count_raises(self):
        with self.assertRaises(ValueError):
            parse_lr_block_ramps([0, 0, 5])

    def test_duplicate_index_raises(self):
        with self.assertRaises(ValueError):
            parse_lr_block_ramps([0, 0, 0, 1])


class TestGetBlockIndexFromKey(unittest.TestCase):

    def test_transformer_block(self):
        self.assertEqual(_get_block_index_from_key("transformer_blocks.5.attn.q.weight", 15), 5)

    def test_scale_shift_table(self):
        self.assertEqual(_get_block_index_from_key("scale_shift_table", 15), 0)

    def test_caption_prefix(self):
        self.assertEqual(_get_block_index_from_key("caption_projection.linear_1.weight", 15), 0)

    def test_time_embed(self):
        self.assertEqual(_get_block_index_from_key("time_embed.linear_1.weight", 15), 0)

    def test_patch_embed(self):
        self.assertEqual(_get_block_index_from_key("patch_embed.proj.weight", 15), 0)

    def test_proj_out(self):
        self.assertEqual(_get_block_index_from_key("proj_out.weight", 15), 15)

    def test_unrecognized_raises(self):
        with self.assertRaises(RuntimeError):
            _get_block_index_from_key("conv_in.weight", 15)


class TestRampBlockwiseAlpha(unittest.TestCase):

    def test_ramp_0_to_5(self):
        ramp = [(0, 0.0), (5, 1.0)]
        for i, expected in enumerate([0.0, 0.2, 0.4, 0.6, 0.8, 1.0]):
            self.assertAlmostEqual(_get_ramp_blockwise_alpha(i, ramp), expected)

    def test_clamp_below_first(self):
        ramp = [(1, 0.0), (4, 1.0), (8, 1.0), (15, 0.0)]
        self.assertAlmostEqual(_get_ramp_blockwise_alpha(0, ramp), 0.0)

    def test_clamp_above_last(self):
        ramp = [(1, 0.0), (4, 1.0), (8, 1.0), (15, 0.0)]
        self.assertAlmostEqual(_get_ramp_blockwise_alpha(19, ramp), 0.0)

    def test_interpolation_midpoints(self):
        ramp = [(1, 0.0), (4, 1.0), (8, 1.0), (15, 0.0)]
        self.assertAlmostEqual(_get_ramp_blockwise_alpha(2, ramp), 1.0 / 3.0)
        self.assertAlmostEqual(_get_ramp_blockwise_alpha(3, ramp), 2.0 / 3.0)
        self.assertAlmostEqual(_get_ramp_blockwise_alpha(9, ramp), 6.0 / 7.0)
        self.assertAlmostEqual(_get_ramp_blockwise_alpha(10, ramp), 5.0 / 7.0)
        self.assertAlmostEqual(_get_ramp_blockwise_alpha(11, ramp), 4.0 / 7.0)
        self.assertAlmostEqual(_get_ramp_blockwise_alpha(12, ramp), 3.0 / 7.0)
        self.assertAlmostEqual(_get_ramp_blockwise_alpha(13, ramp), 2.0 / 7.0)
        self.assertAlmostEqual(_get_ramp_blockwise_alpha(14, ramp), 1.0 / 7.0)

    def test_single_pair_constant(self):
        ramp = [(3, 0.5)]
        for i in range(10):
            self.assertAlmostEqual(_get_ramp_blockwise_alpha(i, ramp), 0.5)


class TestApplyBlockRamps(unittest.TestCase):

    def test_empty_ramp_passthrough(self):
        params = _make_params(_sana_fixture())
        groups = _make_single_group(params)
        self.assertEqual(apply_block_ramps(groups, params, [], (0.9, 0.999), 0.01), groups)

    def test_no_transformer_blocks_passthrough(self):
        params = _make_params(["scale_shift_table", "proj_out.weight"])
        groups = _make_single_group(params)
        result = apply_block_ramps(groups, params, [(0, 0.0), (5, 1.0)], (0.9, 0.999), 0.01)
        self.assertEqual(len(result), 1)
        self.assertEqual(result[0]["lr"], 1e-4)

    def test_empty_param_group_preserved(self):
        params = _make_params(_sana_fixture())
        groups = [{"name": "empty", "params": [], "lr": 1.0,
                   "betas": (0.9, 0.999), "weight_decay": 0.01}]
        result = apply_block_ramps(groups, params, [(0, 0.0), (5, 1.0)], (0.9, 0.999), 0.01)
        self.assertEqual(result, groups)

    def test_ramp_0_to_5_over_sana_fixture(self):
        names = _sana_fixture(num_blocks=6)
        params = _make_params(names)
        groups = _make_single_group(params, lr=1.0)
        result = apply_block_ramps(groups, params, [(0, 0.0), (5, 1.0)], (0.9, 0.999), 0.01)

        lrs = _lr_by_name(result, params)
        for i in range(6):
            self.assertAlmostEqual(lrs[f"transformer_blocks.{i}.attn.q.weight"], 0.2 * i)
        self.assertAlmostEqual(lrs["proj_out.weight"], 1.0)
        self.assertAlmostEqual(lrs["scale_shift_table"], 0.0)
        self.assertAlmostEqual(lrs["caption_projection.linear_1.weight"], 0.0)
        self.assertAlmostEqual(lrs["time_embed.linear_1.weight"], 0.0)
        self.assertAlmostEqual(lrs["patch_embed.proj.weight"], 0.0)

    def test_ramp_with_clamped_extra_block(self):
        names = _sana_fixture(num_blocks=6)
        names.append("transformer_blocks.6.attn.q.weight")
        params = _make_params(names)
        groups = _make_single_group(params, lr=1.0)
        result = apply_block_ramps(groups, params, [(0, 0.0), (5, 1.0)], (0.9, 0.999), 0.01)

        lrs = _lr_by_name(result, params)
        self.assertAlmostEqual(lrs["transformer_blocks.6.attn.q.weight"], 1.0)

    def test_full_ramp_20_blocks(self):
        names = _sana_fixture(num_blocks=20)
        params = _make_params(names)
        groups = _make_single_group(params, lr=1.0)
        ramp = [(1, 0.0), (4, 1.0), (8, 1.0), (15, 0.0)]
        result = apply_block_ramps(groups, params, ramp, (0.9, 0.999), 0.01)

        expected = {0: 0.0, 1: 0.0, 2: 1.0 / 3.0, 3: 2.0 / 3.0}
        for i in range(4, 9):
            expected[i] = 1.0
        for i in range(9, 15):
            expected[i] = (15 - i) / 7.0
        for i in range(15, 20):
            expected[i] = 0.0

        lrs = _lr_by_name(result, params)
        for i, alpha in expected.items():
            self.assertAlmostEqual(lrs[f"transformer_blocks.{i}.attn.q.weight"], alpha)
        self.assertAlmostEqual(lrs["proj_out.weight"], 0.0)

    def test_single_pair_constant_alpha(self):
        names = _sana_fixture(num_blocks=4)
        params = _make_params(names)
        groups = _make_single_group(params, lr=2.0)
        result = apply_block_ramps(groups, params, [(3, 0.5)], (0.9, 0.999), 0.01)

        lrs = _lr_by_name(result, params)
        for name in names:
            self.assertAlmostEqual(lrs[name], 1.0)

    def test_transformer10x_composition(self):
        names = _transformer_names(num_blocks=5, params_per_block=1)
        names.append("scale_shift_table")
        params = _make_params(names)
        transformer_params = [p for n, p in params if "transformer_blocks" in n]
        other_params = [p for n, p in params if "transformer_blocks" not in n]
        groups = [
            {"name": "transformer_blocks", "params": transformer_params,
             "lr": 10.0, "betas": (0.9, 0.999), "weight_decay": 0.01},
            {"name": "non-transformer_blocks", "params": other_params,
             "lr": 1.0, "betas": (0.9, 0.999), "weight_decay": 0.01},
        ]
        result = apply_block_ramps(groups, params, [(0, 0.0), (5, 1.0)], (0.9, 0.999), 0.01)

        lrs = _lr_by_name(result, params)
        self.assertAlmostEqual(lrs["transformer_blocks.1.attn.q.weight"], 2.0)
        self.assertAlmostEqual(lrs["scale_shift_table"], 0.0)


if __name__ == "__main__":
    unittest.main()
