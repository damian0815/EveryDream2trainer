import json
import os
import tempfile
import unittest

import torch
import safetensors.torch

from scripts.sana_lora_tool import (
    BLOCK_RE,
    bake_lora_into_state_dict,
    compute_delta,
    extract_lora_from_state_dicts,
    get_num_blocks,
    load_lora_adapter,
    resize_adapter,
    save_lora_adapter,
    svd_extract,
)

CORE = ["attn1.to_q", "attn1.to_k", "attn1.to_v", "attn1.to_out.0",
        "attn2.to_q", "attn2.to_k", "attn2.to_v", "attn2.to_out.0"]


def build_synthetic_base(out=16, ind=8, n_blocks=2):
    sd = {}
    for i in range(n_blocks):
        for name in CORE:
            sd[f"transformer_blocks.{i}.{name}.weight"] = torch.randn(out, ind)
    return sd


def build_synthetic_adapter(module_path, rank=4, out=16, ind=8, alpha=8):
    torch.manual_seed(1234)
    A = torch.randn(rank, ind)
    B = torch.randn(out, rank)
    return {module_path: (A, B, rank, alpha)}


def rel_error(a, b):
    return (a.float() - b.float()).norm() / b.float().norm()


class TestSvdExtract(unittest.TestCase):
    def test_reproduces_delta_at_full_rank(self):
        torch.manual_seed(0)
        M = torch.randn(24, 12)
        A, B, used = svd_extract(M, rank=8, scaling=2.0)
        self.assertEqual(used, 8)
        approx = 2.0 * (B @ A)
        self.assertLess(rel_error(approx, M), 1e-4)

    def test_conv_reproduces_delta(self):
        torch.manual_seed(1)
        M = torch.randn(16, 8, 1, 1)
        A, B, used = svd_extract(M, rank=4, scaling=3.0)
        self.assertEqual(used, 4)
        self.assertEqual(A.shape, (4, 8, 1, 1))
        self.assertEqual(B.shape, (16, 4, 1, 1))
        approx = 3.0 * (B[:, :, 0, 0] @ A[:, :, 0, 0])
        self.assertLess(rel_error(approx, M[:, :, 0, 0]), 1e-4)

    def test_truncates_rank(self):
        torch.manual_seed(2)
        M = torch.randn(30, 30)
        A, B, used = svd_extract(M, rank=3, scaling=1.0)
        self.assertEqual(used, 3)
        self.assertEqual(A.shape[0], 3)
        # a rank-3 approx cannot reproduce a full-rank matrix exactly
        approx = B @ A
        self.assertGreater(rel_error(approx, M), 1e-3)


class TestBakeExtractRoundTrip(unittest.TestCase):
    def test_bake_then_extract_recovers_delta(self):
        base_sd = build_synthetic_base()
        module_path = "transformer_blocks.0.attn1.to_q"
        adapter = build_synthetic_adapter(module_path)
        original_delta = compute_delta(adapter[module_path][0], adapter[module_path][1],
                                       8 / 4)

        baked, merged, skipped = bake_lora_into_state_dict(base_sd, adapter)
        self.assertEqual(merged, 1)
        self.assertEqual(skipped, 0)
        self.assertEqual(baked[f"{module_path}.weight"].shape, (16, 8))

        recovered, skipped_keys, used = extract_lora_from_state_dicts(
            base_sd, baked, rank=4, alpha=8)
        self.assertEqual(skipped_keys, [])
        self.assertEqual(used, [4])
        A, B = recovered[module_path]
        approx_delta = compute_delta(A, B, 8 / 4)
        self.assertLess(rel_error(approx_delta, original_delta), 1e-4)

    def test_bake_missing_weight_is_skipped(self):
        base_sd = {"some.other.weight": torch.zeros(2, 2)}
        adapter = build_synthetic_adapter("transformer_blocks.0.attn1.to_q")
        _, merged, skipped = bake_lora_into_state_dict(base_sd, adapter)
        self.assertEqual(merged, 0)
        self.assertEqual(skipped, 1)

    def test_bake_conv(self):
        base_sd = {"transformer_blocks.0.ff.conv_point.weight": torch.randn(16, 8, 1, 1)}
        torch.manual_seed(3)
        A = torch.randn(4, 8, 1, 1)
        B = torch.randn(16, 4, 1, 1)
        adapter = {"transformer_blocks.0.ff.conv_point": (A, B, 4, 8)}
        baked, _, _ = bake_lora_into_state_dict(base_sd, adapter)
        recovered, _, _ = extract_lora_from_state_dicts(
            base_sd, baked, rank=4, alpha=8, include_ffn=True)
        A2, B2 = recovered["transformer_blocks.0.ff.conv_point"]
        approx = compute_delta(A2, B2, 8 / 4)
        target = compute_delta(A, B, 8 / 4)
        self.assertLess(rel_error(approx, target), 1e-4)


class TestResize(unittest.TestCase):
    def _make_adapter(self, rank=8, out=20, ind=10, alpha=16):
        torch.manual_seed(7)
        module_path = "transformer_blocks.1.attn2.to_v"
        A = torch.randn(rank, ind)
        B = torch.randn(out, rank)
        return {module_path: (A, B, rank, alpha)}

    def test_same_rank_round_trip(self):
        adapter = self._make_adapter()
        module_path = next(iter(adapter))
        A, B, r, alpha = adapter[module_path]
        target = compute_delta(A, B, alpha / r)

        resized, used = resize_adapter(adapter, rank=8, alpha=16)
        self.assertEqual(used, [8])
        A2, B2 = resized[module_path]
        approx = compute_delta(A2, B2, 16 / 8)
        self.assertLess(rel_error(approx, target), 1e-4)

    def test_smaller_rank_truncates(self):
        adapter = self._make_adapter()
        module_path = next(iter(adapter))
        resized, used = resize_adapter(adapter, rank=4, alpha=16)
        self.assertEqual(used, [4])
        self.assertEqual(resized[module_path][0].shape[0], 4)
        self.assertEqual(resized[module_path][1].shape[1], 4)

    def test_larger_rank_is_numerically_identical(self):
        adapter = self._make_adapter(rank=8)
        module_path = next(iter(adapter))
        A, B, r, alpha = adapter[module_path]
        target = compute_delta(A, B, alpha / r)

        resized, used = resize_adapter(adapter, rank=16, alpha=16)
        # SVD of an already-rank-8 matrix at rank 16 yields zeros beyond rank 8
        self.assertEqual(used, [8])
        A2, B2 = resized[module_path]
        approx = compute_delta(A2, B2, 16 / 16)
        self.assertLess(rel_error(approx, target), 1e-4)


class TestLoadLoraAdapter(unittest.TestCase):
    STYLES = {
        "diffusers": "transformer.transformer_blocks.0.attn1.to_q.lora.{part}.weight",
        "trainer_unprefixed": "transformer_blocks.0.attn1.to_q.lora.{part}.weight",
        "peft_wrapped": "transformer.base_model.model.transformer_blocks.0.attn1.to_q.lora_{part}.default.weight",
    }
    MODULE = "transformer_blocks.0.attn1.to_q"
    METADATA = {"transformer": {"r": 4, "lora_alpha": 8, "target_modules": ["attn1.to_q"]}}

    def _write_adapter(self, dirpath, style, metadata=None):
        torch.manual_seed(42)
        A = torch.randn(4, 8)
        B = torch.randn(16, 4)
        key = lambda part: self.STYLES[style].format(part=part)
        sd = {key("down" if "down" in key("down") else "A"): A,
              key("up" if "up" in key("up") else "B"): B}
        sd = {k.replace("down", "down") if "down" in k else k: v for k, v in sd.items()}
        file_meta = None if metadata is None else {
            "format": "pt",
            "lora_adapter_metadata": json.dumps(metadata, indent=2, sort_keys=True),
        }
        path = os.path.join(dirpath, "pytorch_lora_weights.safetensors")
        safetensors.torch.save_file(sd, path, metadata=file_meta)
        return path

    def test_normalizes_all_key_styles(self):
        with tempfile.TemporaryDirectory() as tmp:
            results = {}
            for style in self.STYLES:
                path = self._write_adapter(tmp, style, metadata=self.METADATA)
                adapters = load_lora_adapter(path)
                A, B, rank, alpha = adapters[self.MODULE]
                results[style] = (A, B, rank, alpha)
            base = results["diffusers"]
            for style, (A, B, rank, alpha) in results.items():
                self.assertEqual(rank, 4)
                self.assertEqual(alpha, 8)
                self.assertTrue(torch.equal(A, base[0]))
                self.assertTrue(torch.equal(B, base[1]))

    def test_defaults_alpha_to_rank_without_metadata(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = self._write_adapter(tmp, "diffusers", metadata=None)
            adapters = load_lora_adapter(path)
            _, _, rank, alpha = adapters[self.MODULE]
            self.assertEqual(rank, 4)
            self.assertEqual(alpha, 4)

    def test_bias_lora_raises(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "pytorch_lora_weights.safetensors")
            safetensors.torch.save_file({
                "transformer.transformer_blocks.0.attn1.to_q.lora_A.default.weight": torch.zeros(4, 8),
                "transformer.transformer_blocks.0.attn1.to_q.lora_B.default.weight": torch.zeros(16, 4),
                "transformer.transformer_blocks.0.attn1.to_q.lora_B.default.bias": torch.zeros(16),
            })
            with self.assertRaises(ValueError):
                load_lora_adapter(path)

    def test_unrecognized_keys_warn_and_skip(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "pytorch_lora_weights.safetensors")
            safetensors.torch.save_file({
                "transformer.transformer_blocks.0.attn1.to_q.lora.down.weight": torch.zeros(4, 8),
                "transformer.transformer_blocks.0.attn1.to_q.lora.up.weight": torch.zeros(16, 4),
                "transformer.transformer_blocks.0.attn1.to_q.dora_scale": torch.ones(16),
            })
            adapters = load_lora_adapter(path)
            self.assertEqual(len(adapters), 1)

    def test_missing_half_is_hard_error(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "pytorch_lora_weights.safetensors")
            safetensors.torch.save_file({
                "transformer.transformer_blocks.0.attn1.to_q.lora.down.weight": torch.zeros(4, 8),
            })
            with self.assertRaises(ValueError):
                load_lora_adapter(path)


class TestSaveLoraAdapter(unittest.TestCase):
    def test_writes_expected_keys_and_metadata(self):
        with tempfile.TemporaryDirectory() as tmp:
            torch.manual_seed(9)
            A = torch.randn(4, 8)
            B = torch.randn(16, 4)
            adapters = {"transformer_blocks.0.attn1.to_q": (A, B)}
            save_lora_adapter(tmp, adapters, rank=4, alpha=8,
                              target_modules=["attn1.to_q"], base_id="base/model")

            path = os.path.join(tmp, "pytorch_lora_weights.safetensors")
            sd = safetensors.torch.load_file(path)
            self.assertEqual(set(sd.keys()), {
                "transformer.transformer_blocks.0.attn1.to_q.lora.down.weight",
                "transformer.transformer_blocks.0.attn1.to_q.lora.up.weight",
            })
            self.assertTrue(torch.equal(sd["transformer.transformer_blocks.0.attn1.to_q.lora.down.weight"], A))
            self.assertTrue(torch.equal(sd["transformer.transformer_blocks.0.attn1.to_q.lora.up.weight"], B))

            with safetensors.torch.safe_open(path, framework="pt", device="cpu") as f:
                meta = f.metadata()
            parsed = json.loads(meta["lora_adapter_metadata"])
            self.assertEqual(parsed, {
                "transformer": {
                    "r": 4,
                    "lora_alpha": 8,
                    "target_modules": ["attn1.to_q"],
                }
            })

            with open(os.path.join(tmp, "model_id.txt")) as f:
                self.assertEqual(f.read(), "base/model")
            self.assertFalse(os.path.exists(os.path.join(tmp, "adapter_config.json")))

    def test_round_trips_through_save_and_load(self):
        with tempfile.TemporaryDirectory() as tmp:
            torch.manual_seed(11)
            A = torch.randn(4, 8)
            B = torch.randn(16, 4)
            adapters = {"transformer_blocks.1.attn2.to_k": (A, B)}
            save_lora_adapter(tmp, adapters, rank=4, alpha=16,
                              target_modules=["attn2.to_k"])
            loaded = load_lora_adapter(tmp)
            A2, B2, rank, alpha = loaded["transformer_blocks.1.attn2.to_k"]
            self.assertEqual(rank, 4)
            self.assertEqual(alpha, 16)
            self.assertTrue(torch.equal(A, A2))
            self.assertTrue(torch.equal(B, B2))


class TestHelpers(unittest.TestCase):
    def test_get_num_blocks(self):
        sd = {"transformer_blocks.0.a.weight": torch.zeros(1),
              "transformer_blocks.11.b.weight": torch.zeros(1)}
        self.assertEqual(get_num_blocks(sd), 12)

    def test_block_re(self):
        self.assertEqual(BLOCK_RE.sub("", "transformer_blocks.3.attn1.to_q"), "attn1.to_q")


if __name__ == "__main__":
    unittest.main()
