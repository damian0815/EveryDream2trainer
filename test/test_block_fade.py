"""
Unit tests for apply_block_fade -- logarithmic LR fade-in/fade-out on transformer blocks.

Scale formula:
  fadein:  2^(i - k)  for block i < k
  fadeout: 2^((N-1-i) - k)  for block i >= N - k
  overlap: min(fadein_scale, fadeout_scale)
"""
import unittest

import torch

from optimizer.per_unet_layer_scales import apply_block_fade


def _make_params(names):
    """Create fake (name, Parameter) pairs."""
    return [(n, torch.nn.Parameter(torch.zeros(1))) for n in names]


def _make_single_group(parameters, lr=1e-4):
    return [{"name": "single", "params": [p for _, p in parameters], "lr": lr,
             "betas": (0.9, 0.999), "weight_decay": 0.01}]


def _transformer_names(num_blocks=20, params_per_block=2):
    names = []
    for i in range(num_blocks):
        for j in range(params_per_block):
            names.append(f"transformer_blocks.{i}.attn.to_q.weight")
    return names


class TestBlockFadePassthrough(unittest.TestCase):

    def test_k_zero_returns_unchanged(self):
        params = _make_params(_transformer_names(4, 1))
        groups = _make_single_group(params)
        result = apply_block_fade(groups, params, fadein_k=0, fadeout_k=0,
                                  betas=(0.9, 0.999), weight_decay=0.01)
        self.assertEqual(result, groups)


class TestBlockFadeIn(unittest.TestCase):

    def test_fadein_k3_scales(self):
        names = _transformer_names(num_blocks=5, params_per_block=1)
        params = _make_params(names)
        groups = _make_single_group(params, lr=1.0)
        result = apply_block_fade(groups, params, fadein_k=3, fadeout_k=0,
                                  betas=(0.9, 0.999), weight_decay=0.01)

        scale_by_name = {}
        for g in result:
            for p in g['params']:
                idx = next(i for i, (_, pp) in enumerate(params) if id(p) == id(pp))
                scale_by_name[names[idx]] = g['lr']

        self.assertAlmostEqual(scale_by_name["transformer_blocks.0.attn.to_q.weight"], 0.125)
        self.assertAlmostEqual(scale_by_name["transformer_blocks.1.attn.to_q.weight"], 0.25)
        self.assertAlmostEqual(scale_by_name["transformer_blocks.2.attn.to_q.weight"], 0.5)
        self.assertAlmostEqual(scale_by_name["transformer_blocks.3.attn.to_q.weight"], 1.0)
        self.assertAlmostEqual(scale_by_name["transformer_blocks.4.attn.to_q.weight"], 1.0)


class TestBlockFadeOut(unittest.TestCase):

    def test_fadeout_k3_scales(self):
        N = 5
        names = _transformer_names(num_blocks=N, params_per_block=1)
        params = _make_params(names)
        groups = _make_single_group(params, lr=1.0)
        result = apply_block_fade(groups, params, fadein_k=0, fadeout_k=3,
                                  betas=(0.9, 0.999), weight_decay=0.01)

        scale_by_name = {}
        for g in result:
            for p in g['params']:
                idx = next(i for i, (_, pp) in enumerate(params) if id(p) == id(pp))
                scale_by_name[names[idx]] = g['lr']

        self.assertAlmostEqual(scale_by_name["transformer_blocks.0.attn.to_q.weight"], 1.0)
        self.assertAlmostEqual(scale_by_name["transformer_blocks.1.attn.to_q.weight"], 1.0)
        self.assertAlmostEqual(scale_by_name["transformer_blocks.2.attn.to_q.weight"], 0.5)
        self.assertAlmostEqual(scale_by_name["transformer_blocks.3.attn.to_q.weight"], 0.25)
        self.assertAlmostEqual(scale_by_name["transformer_blocks.4.attn.to_q.weight"], 0.125)


class TestOverlapHandling(unittest.TestCase):

    def test_fadein_plus_fadeout_exceeds_N(self):
        N = 4
        names = _transformer_names(num_blocks=N, params_per_block=1)
        params = _make_params(names)
        groups = _make_single_group(params, lr=1.0)
        result = apply_block_fade(groups, params, fadein_k=3, fadeout_k=3,
                                  betas=(0.9, 0.999), weight_decay=0.01)

        lrs = {}
        for g in result:
            for p in g['params']:
                idx = next(i for i, (_, pp) in enumerate(params) if id(p) == id(pp))
                lrs[names[idx]] = g['lr']

        # block 0: fadein=2^(0-3)=0.125, fadeout not applicable (0 < 1), scale=0.125
        self.assertAlmostEqual(lrs["transformer_blocks.0.attn.to_q.weight"], 0.125)
        # block 1: fadein=2^(1-3)=0.25, fadeout: 1>=1 so 2^((3-1)-3)=2^-1=0.5, min=0.25
        self.assertAlmostEqual(lrs["transformer_blocks.1.attn.to_q.weight"], 0.25)
        # block 2: fadein=2^(2-3)=0.5, fadeout: 2>=1 so 2^((3-2)-3)=2^-2=0.25, min=0.25
        self.assertAlmostEqual(lrs["transformer_blocks.2.attn.to_q.weight"], 0.25)
        # block 3: fadein not applicable (3>=3), fadeout: 3>=1 so 2^((3-3)-3)=2^-3=0.125
        self.assertAlmostEqual(lrs["transformer_blocks.3.attn.to_q.weight"], 0.125)


class TestNonTransformerParamsPreserved(unittest.TestCase):

    def test_non_block_params_get_original_lr(self):
        names = _transformer_names(num_blocks=3, params_per_block=1)
        names.append("caption_projection.weight")
        names.append("time_embedding.linear_0.weight")
        params = _make_params(names)
        groups = _make_single_group(params, lr=1.0)
        result = apply_block_fade(groups, params, fadein_k=2, fadeout_k=0,
                                  betas=(0.9, 0.999), weight_decay=0.01)

        lrs = {}
        for g in result:
            for p in g['params']:
                idx = next(i for i, (_, pp) in enumerate(params) if id(p) == id(pp))
                lrs[names[idx]] = g['lr']

        self.assertAlmostEqual(lrs["caption_projection.weight"], 1.0)
        self.assertAlmostEqual(lrs["time_embedding.linear_0.weight"], 1.0)


class TestComposabilityWithTransformer10x(unittest.TestCase):

    def test_10x_base_composes_with_fade(self):
        N = 5
        names = _transformer_names(num_blocks=N, params_per_block=1)
        names.append("conv_in.weight")
        params = _make_params(names)
        transformer_params = [p for n, p in params if 'transformer_blocks' in n]
        other_params = [p for n, p in params if 'transformer_blocks' not in n]
        groups = [
            {"name": "transformer_blocks", "params": transformer_params,
             "lr": 10.0, "betas": (0.9, 0.999), "weight_decay": 0.01},
            {"name": "non-transformer_blocks", "params": other_params,
             "lr": 1.0, "betas": (0.9, 0.999), "weight_decay": 0.01},
        ]
        result = apply_block_fade(groups, params, fadein_k=3, fadeout_k=0,
                                  betas=(0.9, 0.999), weight_decay=0.01)

        lrs = {}
        for g in result:
            for p in g['params']:
                idx = next(i for i, (_, pp) in enumerate(params) if id(p) == id(pp))
                lrs[names[idx]] = g['lr']

        # 10x * 0.125 = 1.25 for block 0
        self.assertAlmostEqual(lrs["transformer_blocks.0.attn.to_q.weight"], 1.25)
        # 10x * 0.25 = 2.5 for block 1
        self.assertAlmostEqual(lrs["transformer_blocks.1.attn.to_q.weight"], 2.5)
        # 10x * 0.5 = 5.0 for block 2
        self.assertAlmostEqual(lrs["transformer_blocks.2.attn.to_q.weight"], 5.0)
        # 10x * 1.0 = 10.0 for blocks 3, 4
        self.assertAlmostEqual(lrs["transformer_blocks.3.attn.to_q.weight"], 10.0)
        self.assertAlmostEqual(lrs["transformer_blocks.4.attn.to_q.weight"], 10.0)
        # Non-transformer stays at 1.0
        self.assertAlmostEqual(lrs["conv_in.weight"], 1.0)


class TestNoTransformerBlocks(unittest.TestCase):

    def test_no_blocks_returns_unchanged(self):
        params = _make_params(["conv_in.weight", "conv_out.weight"])
        groups = _make_single_group(params, lr=1.0)
        result = apply_block_fade(groups, params, fadein_k=3, fadeout_k=3,
                                  betas=(0.9, 0.999), weight_decay=0.01)
        self.assertEqual(len(result), 1)
        self.assertEqual(result[0]['lr'], 1.0)


if __name__ == "__main__":
    unittest.main()
