"""
Unit tests for core/lora.py — adapter-params-only EMA.

Runs on CPU, no GPU, no peft import.  Uses nested ``nn.Module``s whose state-dict
keys contain dots (as PEFT produces) and marks the "adapter-ish" params with
``requires_grad_(True)``, mirroring how LoRA training freezes the base weights.
"""
import os
import tempfile
import unittest
from argparse import Namespace
from collections import OrderedDict
from unittest import mock

import torch
import torch.nn as nn
import safetensors.torch

from core.lora import (
    apply_lora_ema_inplace,
    build_lora_ema_state,
    get_adapter_params,
    get_lora_components,
    init_lora_ema,
    lora_ema_inplace_swap,
    save_lora_ema_sidecar,
    update_lora_ema,
)


class LoRAAdapter(nn.Module):
    """Mini stand-in for a PEFT LoRA layer (lora_A / lora_B, trainable)."""

    def __init__(self, out_dim, in_dim, rank=4):
        super().__init__()
        self.lora_A = nn.Parameter(torch.randn(rank, in_dim))
        self.lora_B = nn.Parameter(torch.randn(out_dim, rank))


class Attn(nn.Module):
    """One attention block: frozen base weights + trainable LoRA adapters."""

    def __init__(self):
        super().__init__()
        self.to_q = nn.Linear(8, 8)
        self.to_k = nn.Linear(8, 8)
        self.q_lora = LoRAAdapter(8, 8)
        self.k_lora = LoRAAdapter(8, 8)
        for p in self.parameters():
            p.requires_grad_(False)
        for p in self.q_lora.parameters():
            p.requires_grad_(True)
        for p in self.k_lora.parameters():
            p.requires_grad_(True)


def build_unet_like(n_blocks=2) -> nn.Module:
    net = nn.Module()
    net.down_blocks = nn.ModuleList([Attn() for _ in range(n_blocks)])
    return net


def build_sd_like_model() -> nn.Module:
    model = nn.Module()
    model.unet = build_unet_like()
    model.text_encoder = build_unet_like(n_blocks=1)
    model.text_encoder_2 = None
    model.is_sdxl = False
    return model


def build_sana_like_model() -> nn.Module:
    model = nn.Module()
    model.transformer = build_unet_like()
    return model


def base_args(**overrides) -> Namespace:
    args = Namespace(
        disable_unet_training=False,
        disable_textenc_training=False,
        ema_device="cpu",
        ema_decay_rate=0.99,
        ema_update_interval=1,
    )
    for k, v in overrides.items():
        setattr(args, k, v)
    return args


def trainable_params(model: nn.Module) -> OrderedDict:
    return get_adapter_params(model)


class TestBuildLoraEmaState(unittest.TestCase):

    def test_snapshots_only_trainable_params(self):
        module = build_unet_like()
        state = build_lora_ema_state(module)
        self.assertTrue(state)
        for name in state:
            self.assertTrue(name.endswith("lora_A") or name.endswith("lora_B"),
                            f"unexpected key {name!r}")
        # base weights are frozen → not in the adapter snapshot
        self.assertNotIn("down_blocks.0.to_q.weight", state)

    def test_snapshot_is_detached_copy(self):
        module = build_unet_like()
        state = build_lora_ema_state(module)
        name = next(iter(state))
        live = dict(module.named_parameters())[name]
        before = live.detach().clone()
        # mutating the snapshot must not touch the live param
        state[name].add_(123.0)
        self.assertTrue(torch.equal(live, before))
        # mutating the live param must not touch the snapshot
        live.data.add_(456.0)
        state2 = build_lora_ema_state(module)
        self.assertFalse(torch.equal(state2[name], state[name] - 123.0))
        self.assertTrue(torch.equal(state2[name], before + 456.0))


class TestUpdateLoraEma(unittest.TestCase):

    def test_decay_math_inplace_devices(self):
        module = build_unet_like()
        state = build_lora_ema_state(module)
        name = "down_blocks.0.q_lora.lora_A"
        old = state[name].clone()
        module.down_blocks[0].q_lora.lora_A.data.add_(1.0)
        new = module.down_blocks[0].q_lora.lora_A.detach()
        update_lora_ema(module, state, decay=0.9, default_device="cpu", ema_device="cpu")
        expected = old * 0.9 + new * 0.1
        torch.testing.assert_close(state[name], expected)

    def test_decay_math_cross_device(self):
        # ema lives on cpu, live reported on a different device → copy path.
        module = build_unet_like()
        state = build_lora_ema_state(module)
        name = "down_blocks.0.q_lora.lora_A"
        old = state[name].clone()
        module.down_blocks[0].q_lora.lora_A.data.add_(1.0)
        new = module.down_blocks[0].q_lora.lora_A.detach()
        update_lora_ema(module, state, decay=0.9, default_device="cuda", ema_device="cpu")
        expected = old * 0.9 + new.float() * 0.1
        torch.testing.assert_close(state[name], expected)

    def test_ignores_missing_live_params(self):
        module = build_unet_like()
        state = build_lora_ema_state(module)
        # a stale key that no longer has a live counterpart must not raise
        state["ghost.weight"] = state[next(iter(state))].clone()
        update_lora_ema(module, state, decay=0.9, default_device="cpu", ema_device="cpu")
        self.assertIn("ghost.weight", state)

    def test_disk_ema_device_falls_back_to_cpu(self):
        module = build_unet_like()
        state = build_lora_ema_state(module)
        update_lora_ema(module, state, decay=0.9, default_device="cpu", ema_device="disk")
        self.assertEqual(state["down_blocks.0.q_lora.lora_A"].device.type, "cpu")


class TestApplyLoraEmaInplace(unittest.TestCase):

    def test_writes_then_exactly_restores(self):
        module = build_unet_like()
        state = build_lora_ema_state(module)
        live_before = trainable_params(module)
        with apply_lora_ema_inplace(module, state):
            live_now = trainable_params(module)
            for name, ema in state.items():
                torch.testing.assert_close(live_now[name], ema)
        live_after = trainable_params(module)
        for name, before in live_before.items():
            torch.testing.assert_close(live_after[name], before)

    def test_restores_on_exception(self):
        module = build_unet_like()
        state = build_lora_ema_state(module)
        live_before = trainable_params(module)
        with self.assertRaises(RuntimeError):
            with apply_lora_ema_inplace(module, state):
                raise RuntimeError("boom")
        live_after = trainable_params(module)
        for name, before in live_before.items():
            torch.testing.assert_close(live_after[name], before)


class TestLoraEmaInplaceSwap(unittest.TestCase):

    def test_swaps_all_components_and_restores(self):
        model = build_sd_like_model()
        args = base_args()
        init_lora_ema(model, args, device="cpu", resume_dir="")
        live_before = {comp: trainable_params(getattr(model, comp))
                       for comp in ("unet", "text_encoder")}
        with lora_ema_inplace_swap(model):
            for comp, state in model.lora_ema.items():
                for name, ema in state.items():
                    torch.testing.assert_close(
                        dict(getattr(model, comp).named_parameters())[name], ema
                    )
        for comp, before in live_before.items():
            for name, t in before.items():
                torch.testing.assert_close(
                    dict(getattr(model, comp).named_parameters())[name], t
                )


class TestInitAndSidecarRoundTrip(unittest.TestCase):

    def test_roundtrip(self):
        model = build_sd_like_model()
        args = base_args()
        model.lora_ema = None
        self.assertTrue(init_lora_ema(model, args, device="cpu", resume_dir=""))
        self.assertEqual(set(model.lora_ema.keys()), {"unet", "text_encoder"})

        # advance the EMA a bit so the saved sidecar differs from live weights
        update_lora_ema(model.unet, model.lora_ema["unet"], 0.5, "cpu", "cpu")

        with tempfile.TemporaryDirectory() as tmp:
            save_lora_ema_sidecar(model, tmp)
            sidecar = os.path.join(tmp, "lora_ema.safetensors")
            self.assertTrue(os.path.isfile(sidecar))

            sd = safetensors.torch.load_file(sidecar)
            self.assertTrue(any(k.startswith("unet::") for k in sd))
            self.assertTrue(any(k.startswith("text_encoder::") for k in sd))
            self.assertTrue(all("::" in k for k in sd))

            # fresh model resumes identical EMA state
            model2 = build_sd_like_model()
            init_lora_ema(model2, args, device="cpu", resume_dir=tmp)
            for comp in model.lora_ema:
                for name, t in model.lora_ema[comp].items():
                    self.assertIn(name, model2.lora_ema[comp])
                    torch.testing.assert_close(t, model2.lora_ema[comp][name])

    def test_disk_ema_device_warns_and_uses_cpu(self):
        model = build_sd_like_model()
        args = base_args(ema_device="disk")
        with self.assertLogs(level="WARNING"):
            init_lora_ema(model, args, device="cpu", resume_dir="")
        for state in model.lora_ema.values():
            for t in state.values():
                self.assertEqual(t.device.type, "cpu")

    def test_resume_without_sidecar_blends_toward_identity(self):
        # Adapter was loaded from --lora_resume but has no EMA sidecar: the EMA
        # must start as a blend (0.9 live + 0.1 identity), not a pure snapshot.
        # Mock randn so identity-A is deterministic (zeros).
        model = build_sd_like_model()
        args = base_args()
        live_before = {comp: trainable_params(getattr(model, comp))
                       for comp in ("unet", "text_encoder")}
        with tempfile.TemporaryDirectory() as tmp, \
                mock.patch("core.lora.torch.randn_like",
                           side_effect=lambda t: torch.zeros_like(t)):
            init_lora_ema(model, args, device="cpu", resume_dir=tmp)
        for comp in ("unet", "text_encoder"):
            for name, ema in model.lora_ema[comp].items():
                expected = 0.9 * live_before[comp][name].detach()
                torch.testing.assert_close(ema, expected)

    def test_fresh_init_uses_identity_start(self):
        # No adapter was loaded (no --lora_resume): EMA starts at the pure
        # identity start (A~gaussian, B=zeros), i.e. zero LoRA delta.
        model = build_sd_like_model()
        args = base_args()
        with mock.patch("core.lora.torch.randn_like",
                        side_effect=lambda t: torch.zeros_like(t)):
            init_lora_ema(model, args, device="cpu", resume_dir="")
        for comp in ("unet", "text_encoder"):
            for name, ema in model.lora_ema[comp].items():
                self.assertTrue(torch.all(ema == 0), f"{comp}::{name} not identity")

    def test_save_sidecar_noop_without_lora_ema(self):
        model = build_sd_like_model()
        model.lora_ema = None
        with tempfile.TemporaryDirectory() as tmp:
            save_lora_ema_sidecar(model, tmp)
            self.assertFalse(os.path.exists(os.path.join(tmp, "lora_ema.safetensors")))


class TestGetLoraComponents(unittest.TestCase):

    def test_sd_defaults(self):
        self.assertEqual(get_lora_components(build_sd_like_model(), base_args()),
                         ["unet", "text_encoder"])

    def test_sd_sdxl(self):
        model = build_sd_like_model()
        model.text_encoder_2 = build_unet_like(n_blocks=1)
        model.is_sdxl = True
        self.assertEqual(get_lora_components(model, base_args()),
                         ["unet", "text_encoder", "text_encoder_2"])

    def test_sd_disable_textenc(self):
        self.assertEqual(
            get_lora_components(build_sd_like_model(),
                                base_args(disable_textenc_training=True)),
            ["unet"])

    def test_sd_disable_unet(self):
        self.assertEqual(
            get_lora_components(build_sd_like_model(),
                                base_args(disable_unet_training=True)),
            ["text_encoder"])

    def test_sana(self):
        self.assertEqual(get_lora_components(build_sana_like_model(), base_args()),
                         ["transformer"])


if __name__ == "__main__":
    unittest.main()
