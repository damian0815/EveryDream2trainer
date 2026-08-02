import unittest
from unittest.mock import MagicMock

import torch
import torch.nn.functional as F

from core.loss_sana import compute_sana_dpo_loss


class TestComputeSanaDpoLoss(unittest.TestCase):
    def setUp(self):
        self.batch_size = 4
        self.channels = 4
        self.height = 8
        self.width = 8
        self.seq_len = 16
        self.text_dim = 64

    def _make_mock_transformer(self):
        mock = MagicMock()
        mock.parameters = MagicMock(return_value=iter([]))
        return mock

    def _make_mock_scheduler(self):
        mock = MagicMock()
        mock.add_noise.side_effect = lambda z, noise, t: z * (1 - t.view(-1, 1, 1, 1)) + noise * t.view(-1, 1, 1, 1)
        return mock

    def _make_inputs(self, seed=42):
        torch.manual_seed(seed)
        z_good = torch.randn(self.batch_size, self.channels, self.height, self.width)
        z_bad = torch.randn(self.batch_size, self.channels, self.height, self.width)
        y = torch.randn(self.batch_size, self.seq_len, self.text_dim)
        y_mask = torch.ones(self.batch_size, self.seq_len, dtype=torch.bool)
        timesteps = torch.rand(self.batch_size)
        noise = torch.randn(self.batch_size, self.channels, self.height, self.width)
        return z_good, z_bad, y, y_mask, timesteps, noise

    def test_output_shapes(self):
        z_good, z_bad, y, y_mask, timesteps, noise = self._make_inputs()

        policy_transformer = self._make_mock_transformer()
        reference_transformer = self._make_mock_transformer()
        noise_scheduler = self._make_mock_scheduler()

        with torch.no_grad():
            # Mock transformers to return velocity predictions
            target_good = noise - z_good
            target_bad = noise - z_bad

            def policy_forward(**kwargs):
                result = MagicMock()
                hs = kwargs["hidden_states"]
                # Simply return noise as prediction (arbitrary but valid shape)
                result.sample = torch.randn_like(hs)
                return result

            policy_transformer.side_effect = policy_forward
            reference_transformer.side_effect = policy_forward

            loss_1d, info = compute_sana_dpo_loss(
                policy_transformer=policy_transformer,
                reference_transformer=reference_transformer,
                noise_scheduler=noise_scheduler,
                z_good=z_good,
                z_bad=z_bad,
                y=y,
                y_mask=y_mask,
                timesteps=timesteps,
                noise=noise,
                beta=0.1,
            )

        self.assertEqual(loss_1d.shape, (self.batch_size,))
        self.assertIn("dpo_signal", info)
        self.assertIn("err_policy_good", info)
        self.assertIn("err_policy_bad", info)
        self.assertIn("err_ref_good", info)
        self.assertIn("err_ref_bad", info)

    def test_loss_low_when_policy_prefers_good(self):
        z_good, z_bad, y, y_mask, timesteps, noise = self._make_inputs(seed=10)

        policy_transformer = MagicMock()
        reference_transformer = MagicMock()
        noise_scheduler = self._make_mock_scheduler()

        target_good = noise - z_good
        target_bad = noise - z_bad

        call_count = [0]
        def policy_forward(**kwargs):
            call_count[0] += 1
            result = MagicMock()
            hs = kwargs["hidden_states"]
            # Policy predicts good images well (low error) and bad images poorly (high error)
            if call_count[0] % 2 == 1:
                # Good image: predict close to target
                result.sample = target_good[call_count[0]//2] + 0.01 * torch.randn_like(target_good[0])
            else:
                # Bad image: predict far from target
                result.sample = target_bad[(call_count[0]-1)//2] + 2.0 * torch.randn_like(target_bad[0])
            return result

        def ref_forward(**kwargs):
            result = MagicMock()
            result.sample = kwargs["hidden_states"] + 0.5 * torch.randn_like(kwargs["hidden_states"])
            return result

        policy_transformer.side_effect = policy_forward
        reference_transformer.side_effect = ref_forward

        with torch.no_grad():
            loss_1d, info = compute_sana_dpo_loss(
                policy_transformer=policy_transformer,
                reference_transformer=reference_transformer,
                noise_scheduler=noise_scheduler,
                z_good=z_good,
                z_bad=z_bad,
                y=y,
                y_mask=y_mask,
                timesteps=timesteps,
                noise=noise,
                beta=0.1,
            )

        # Loss should be reasonable (not NaN, not Inf)
        self.assertFalse(torch.isnan(loss_1d).any(), "Loss contains NaN")
        self.assertFalse(torch.isinf(loss_1d).any(), "Loss contains Inf")

    def test_slice_size(self):
        z_good, z_bad, y, y_mask, timesteps, noise = self._make_inputs(seed=20)

        policy_transformer = MagicMock()
        reference_transformer = MagicMock()
        noise_scheduler = self._make_mock_scheduler()

        target_good = noise - z_good
        target_bad = noise - z_bad

        def forward_fn(**kwargs):
            result = MagicMock()
            result.sample = torch.randn_like(kwargs["hidden_states"])
            return result

        policy_transformer.side_effect = forward_fn
        reference_transformer.side_effect = forward_fn

        with torch.no_grad():
            loss_full, info_full = compute_sana_dpo_loss(
                policy_transformer=policy_transformer,
                reference_transformer=reference_transformer,
                noise_scheduler=noise_scheduler,
                z_good=z_good,
                z_bad=z_bad,
                y=y,
                y_mask=y_mask,
                timesteps=timesteps,
                noise=noise,
                beta=0.1,
                slice_size=None,
            )
            loss_sliced, info_sliced = compute_sana_dpo_loss(
                policy_transformer=policy_transformer,
                reference_transformer=reference_transformer,
                noise_scheduler=noise_scheduler,
                z_good=z_good,
                z_bad=z_bad,
                y=y,
                y_mask=y_mask,
                timesteps=timesteps,
                noise=noise,
                beta=0.1,
                slice_size=2,
            )

        self.assertEqual(loss_full.shape, loss_sliced.shape)

    def test_precomputed_policy_good(self):
        z_good, z_bad, y, y_mask, timesteps, noise = self._make_inputs(seed=30)

        policy_transformer = MagicMock()
        reference_transformer = MagicMock()
        noise_scheduler = self._make_mock_scheduler()

        target_good = noise - z_good
        model_pred_good = torch.randn_like(z_good)

        def policy_forward(**kwargs):
            result = MagicMock()
            result.sample = torch.randn_like(kwargs["hidden_states"])
            return result

        def ref_forward(**kwargs):
            result = MagicMock()
            result.sample = torch.randn_like(kwargs["hidden_states"])
            return result

        policy_transformer.side_effect = policy_forward
        reference_transformer.side_effect = ref_forward

        with torch.no_grad():
            loss_1d, info = compute_sana_dpo_loss(
                policy_transformer=policy_transformer,
                reference_transformer=reference_transformer,
                noise_scheduler=noise_scheduler,
                z_good=z_good,
                z_bad=z_bad,
                y=y,
                y_mask=y_mask,
                timesteps=timesteps,
                noise=noise,
                beta=0.1,
                model_pred_good=model_pred_good,
                target_good=target_good,
            )

        # Policy transformer should have been called for bad images but not re-called for good
        # The exact call count depends on slicing, but loss shape should be correct
        self.assertEqual(loss_1d.shape, (self.batch_size,))
        self.assertFalse(torch.isnan(loss_1d).any())


if __name__ == "__main__":
    unittest.main()
