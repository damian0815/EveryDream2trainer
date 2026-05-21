"""
Integration test: teacher lambda rolloff vs SNR
===============================================

Plots teacher_lambda_rolloff(t) against SNR for three scheduler families:

  1. Flow Matching  – SDPipelineInferenceFlowMatchEulerDiscreteScheduler (shift=1)
  2. Shifted FM     – same scheduler with shift=2
  3. DDPM           – linear-beta schedule  (alphas_cumprod)

Because the rolloff is defined in SNR space the shape should be nearly
identical across all three panels when plotted against log-SNR.

All invariants are asserted in **SNR space** so they hold regardless of
which scheduler maps timesteps to sigma / alpha-bar.

Run standalone::

    pytest test/test_teacher_lambda_rolloff.py -v

or to get the plot without pytest overhead::

    python test/test_teacher_lambda_rolloff.py
"""
from __future__ import annotations

import math
import os
import sys

import pytest
import torch

# ── repo root on path ──────────────────────────────────────────────────────────
_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from core.flow_match_model import SDPipelineInferenceFlowMatchEulerDiscreteScheduler
# Import from the lightweight module — no full training-stack deps.
from core.teacher_lambda import (
    TEACHER_LAMBDA_SNR_ZERO_LO,
    TEACHER_LAMBDA_SNR_FULL_LO,
    TEACHER_LAMBDA_SNR_FULL_HI,
    TEACHER_LAMBDA_SNR_ZERO_HI,
    _snr_from_scheduler,
    get_teacher_lambda,
)


# ── helpers ────────────────────────────────────────────────────────────────────

def _make_fm_scheduler(shift: float = 1.0, num_steps: int = 1000):
    sched = SDPipelineInferenceFlowMatchEulerDiscreteScheduler(
        num_train_timesteps=num_steps, shift=shift,
    )
    sched.set_timesteps(num_steps)
    return sched


def _make_ddpm_scheduler(num_steps: int = 1000):
    """Minimal mock DDPM scheduler with alphas_cumprod only."""
    betas = torch.linspace(1e-4, 0.02, num_steps)
    ac    = torch.cumprod(1.0 - betas, dim=0)

    class _MockDDPM:
        alphas_cumprod = ac

    return _MockDDPM()


def _ts_for_snr_linear_fm(snr: float, num_ts: int = 1000) -> int:
    """
    Return the integer timestep whose SNR equals *snr* under the linear-FM
    fallback schedule (σ = t / num_ts).  Exact inverse of SNR = ((1−σ)/σ)²,
    i.e. σ = 1 / (1 + √SNR).
    """
    sigma = 1.0 / (1.0 + math.sqrt(snr))
    return max(0, min(num_ts - 1, round(sigma * num_ts)))


class _FakeArgs:
    teacher_lambda: float = 1.0
    teacher_lambda_falloff: bool = False


# ── fixtures ───────────────────────────────────────────────────────────────────

@pytest.fixture(scope="module")
def fm_shift1():
    return _make_fm_scheduler(shift=1.0)


@pytest.fixture(scope="module")
def fm_shift2():
    return _make_fm_scheduler(shift=2.0)


@pytest.fixture(scope="module")
def ddpm():
    return _make_ddpm_scheduler()


# ── SNR helper tests ───────────────────────────────────────────────────────────

class TestSnrHelper:
    """Verify _snr_from_scheduler for all three scheduler families."""

    def test_linear_fm_fallback(self):
        """σ = t/1000 → SNR = ((1-σ)/σ)²"""
        ts  = torch.tensor([200, 400, 800], dtype=torch.long)
        snr = _snr_from_scheduler(ts, noise_scheduler=None)
        expected = torch.tensor([16.0, 2.25, 0.0625])
        assert torch.allclose(snr, expected, rtol=1e-4), f"got {snr}"

    def test_fm_shift1_snr_order_matches_fallback(self, fm_shift1):
        """FM shift=1 SNR ranking should agree with the linear-FM fallback."""
        ts = torch.arange(1, 999, dtype=torch.long)
        snr_fb  = _snr_from_scheduler(ts, noise_scheduler=None)
        snr_fm1 = _snr_from_scheduler(ts, noise_scheduler=fm_shift1)
        assert (snr_fb.argsort() == snr_fm1.argsort()).all(), \
            "FM shift=1 SNR order should match fallback order"

    def test_ddpm_snr_decreasing_in_t(self, ddpm):
        """DDPM SNR must be monotonically decreasing with timestep."""
        ts  = torch.arange(0, 999, dtype=torch.long)
        snr = _snr_from_scheduler(ts, noise_scheduler=ddpm)
        assert (snr[1:] - snr[:-1] <= 1e-6).all(), \
            f"DDPM SNR not monotone (max delta={(snr[1:]-snr[:-1]).max():.2e})"

    def test_fm_shift2_snr_decreasing_in_t(self, fm_shift2):
        """FM shift=2 SNR must be monotonically decreasing with timestep value."""
        ts  = torch.arange(1, 999, dtype=torch.long)
        snr = _snr_from_scheduler(ts, noise_scheduler=fm_shift2)
        # Allow tiny non-monotonicities from nearest-match discretisation (1e-3)
        assert (snr[1:] - snr[:-1] <= 1e-3).all(), \
            f"FM shift=2 SNR not monotone (max delta={(snr[1:]-snr[:-1]).max():.2e})"


# ── rolloff invariant tests  (SNR-space assertions) ───────────────────────────

class TestRolloffInvariants:
    """
    Hard mathematical invariants for get_teacher_lambda.
    All assertions are expressed in SNR terms and use the linear-FM fallback
    so that t↔SNR is exact and scheduler-independent.
    """

    def _ts(self, snr_val):
        return torch.tensor([_ts_for_snr_linear_fm(snr_val)], dtype=torch.long)

    def test_zero_at_zero_lo_edge(self):
        lam = get_teacher_lambda(self._ts(TEACHER_LAMBDA_SNR_ZERO_LO), _FakeArgs())
        assert lam.item() < 1e-4, f"Expected ≈0 at SNR=ZERO_LO, got {lam.item():.6f}"

    def test_zero_at_zero_hi_edge(self):
        lam = get_teacher_lambda(self._ts(TEACHER_LAMBDA_SNR_ZERO_HI), _FakeArgs())
        assert lam.item() < 1e-4, f"Expected ≈0 at SNR=ZERO_HI, got {lam.item():.6f}"

    def test_full_at_full_lo_edge(self):
        lam = get_teacher_lambda(self._ts(TEACHER_LAMBDA_SNR_FULL_LO), _FakeArgs())
        assert lam.item() > 1.0 - 1e-4, f"Expected ≈1 at SNR=FULL_LO, got {lam.item():.6f}"

    def test_full_at_full_hi_edge(self):
        lam = get_teacher_lambda(self._ts(TEACHER_LAMBDA_SNR_FULL_HI), _FakeArgs())
        assert lam.item() > 1.0 - 1e-4, f"Expected ≈1 at SNR=FULL_HI, got {lam.item():.6f}"

    def test_full_weight_throughout_active_snr_zone(self):
        """Every t whose SNR falls within [FULL_LO, FULL_HI] must give lambda ≥ 0.999."""
        t_lo = _ts_for_snr_linear_fm(TEACHER_LAMBDA_SNR_FULL_HI)  # higher SNR = lower t
        t_hi = _ts_for_snr_linear_fm(TEACHER_LAMBDA_SNR_FULL_LO)  # lower  SNR = higher t
        ts   = torch.arange(t_lo + 1, t_hi, dtype=torch.long)
        if ts.numel() == 0:
            pytest.skip("No interior timesteps in active zone")
        lam = get_teacher_lambda(ts, _FakeArgs())
        assert lam.min().item() > 1.0 - 1e-4, \
            f"Interior dip in active zone: min={lam.min():.6f}"

    def test_zero_below_zero_lo_snr(self):
        """All timesteps with SNR < ZERO_LO must give lambda = 0."""
        t_zero = _ts_for_snr_linear_fm(TEACHER_LAMBDA_SNR_ZERO_LO)
        ts = torch.arange(t_zero + 5, 999, dtype=torch.long)
        if ts.numel() == 0:
            pytest.skip("No timesteps beyond ZERO_LO")
        lam = get_teacher_lambda(ts, _FakeArgs())
        assert lam.max().item() < 1e-4, \
            f"Expected 0 below SNR_ZERO_LO, max={lam.max():.6f}"

    def test_zero_above_zero_hi_snr(self):
        """All timesteps with SNR > ZERO_HI must give lambda = 0."""
        t_zero = _ts_for_snr_linear_fm(TEACHER_LAMBDA_SNR_ZERO_HI)
        ts = torch.arange(1, max(2, t_zero - 5), dtype=torch.long)
        if ts.numel() == 0:
            pytest.skip("No timesteps beyond ZERO_HI")
        lam = get_teacher_lambda(ts, _FakeArgs())
        assert lam.max().item() < 1e-4, \
            f"Expected 0 above SNR_ZERO_HI, max={lam.max():.6f}"

    def test_monotone_lo_ramp(self):
        """Lambda must be non-increasing as t increases through the lo-SNR ramp."""
        t_zero = _ts_for_snr_linear_fm(TEACHER_LAMBDA_SNR_ZERO_LO)
        t_full = _ts_for_snr_linear_fm(TEACHER_LAMBDA_SNR_FULL_LO)
        ts  = torch.arange(t_full, t_zero + 1, dtype=torch.long)
        lam = get_teacher_lambda(ts, _FakeArgs())
        assert (lam[1:] - lam[:-1] <= 1e-5).all(), \
            f"Lo-SNR ramp not monotone (max increase={( lam[1:]-lam[:-1]).max():.2e})"

    def test_monotone_hi_ramp(self):
        """Lambda must be non-decreasing as t increases through the hi-SNR ramp."""
        t_full_hi = _ts_for_snr_linear_fm(TEACHER_LAMBDA_SNR_FULL_HI)
        t_zero_hi = _ts_for_snr_linear_fm(TEACHER_LAMBDA_SNR_ZERO_HI)
        ts  = torch.arange(max(1, t_zero_hi), t_full_hi + 1, dtype=torch.long)
        lam = get_teacher_lambda(ts, _FakeArgs())
        assert (lam[1:] - lam[:-1] >= -1e-5).all(), \
            f"Hi-SNR ramp not monotone (min delta={( lam[1:]-lam[:-1]).min():.2e})"

    def test_half_weight_at_log_snr_midpoints(self):
        """Cosine midpoints in log-SNR space must yield lambda ≈ 0.5."""
        for lo, hi in [
            (TEACHER_LAMBDA_SNR_ZERO_LO, TEACHER_LAMBDA_SNR_FULL_LO),
            (TEACHER_LAMBDA_SNR_FULL_HI, TEACHER_LAMBDA_SNR_ZERO_HI),
        ]:
            snr_mid = math.exp(0.5 * (math.log(lo) + math.log(hi)))
            ts  = torch.tensor([_ts_for_snr_linear_fm(snr_mid)], dtype=torch.long)
            lam = get_teacher_lambda(ts, _FakeArgs()).item()
            assert abs(lam - 0.5) < 0.05, \
                f"Expected λ≈0.5 at log-SNR midpoint (lo={lo}, hi={hi}, SNR={snr_mid:.4f}), got {lam:.4f}"

    def test_scales_with_base_lambda(self):
        ts = torch.arange(1, 999, dtype=torch.long)
        a1 = _FakeArgs(); a1.teacher_lambda = 1.0
        a2 = _FakeArgs(); a2.teacher_lambda = 0.5
        l1 = get_teacher_lambda(ts, a1)
        l2 = get_teacher_lambda(ts, a2)
        mask = l1 > 1e-6
        ratio = l1[mask] / l2[mask].clamp(min=1e-12)
        assert (ratio - 2.0).abs().max() < 1e-4

    def test_scheduler_invariance_fm_vs_ddpm_at_matching_snr(self, fm_shift1, ddpm):
        """
        At comparable SNR (not the same integer t), FM and DDPM should give
        the same lambda — the rolloff is defined in SNR space.
        """
        target_snr = 0.5
        sigma_fm   = 1.0 / (1.0 + math.sqrt(target_snr))
        fm_sigmas  = fm_shift1.sigmas[:-1].flip(0)
        idx_fm     = (fm_sigmas - sigma_fm).abs().argmin().item()
        ts_fm      = torch.tensor([idx_fm + 1], dtype=torch.long)

        target_ac = target_snr / (1.0 + target_snr)
        betas = torch.linspace(1e-4, 0.02, 1000)
        ac    = torch.cumprod(1.0 - betas, dim=0)
        idx_d = (ac - target_ac).abs().argmin().item()
        ts_d  = torch.tensor([idx_d], dtype=torch.long)

        lam_fm = get_teacher_lambda(ts_fm, _FakeArgs(), fm_shift1).item()
        lam_dd = get_teacher_lambda(ts_d,  _FakeArgs(), ddpm).item()
        assert abs(lam_fm - lam_dd) < 0.05, \
            f"FM({lam_fm:.4f}) vs DDPM({lam_dd:.4f}) mismatch at SNR≈{target_snr}"

    def test_legacy_falloff_path_still_works(self):
        class LegacyArgs:
            teacher_lambda         = 1.0
            teacher_lambda_falloff = True
            teacher_lambda_falloff_tmin = 200
            teacher_lambda_falloff_tmax = 600
        ts  = torch.tensor([200, 400, 600, 800], dtype=torch.long)
        lam = get_teacher_lambda(ts, LegacyArgs())
        assert lam[0].item() < 1e-6,            "Expected 0 at tmin"
        assert abs(lam[1].item() - 0.5) < 1e-4, "Expected 0.5 at midpoint"
        assert abs(lam[2].item() - 1.0) < 1e-4, "Expected 1.0 at tmax"
        assert abs(lam[3].item() - 1.0) < 1e-4, "Expected 1.0 above tmax"


# ── SNR-annotated visual test (generates PNG) ──────────────────────────────────

class TestRolloffVsSNRPlot:
    """
    Generates /tmp/teacher_lambda_rolloff.png.
    The three panels should produce nearly identical lambda-vs-log_SNR curves,
    demonstrating scheduler independence.
    """

    OUTPUT_PATH = "/tmp/teacher_lambda_rolloff.png"

    def test_curves_nearly_identical_across_schedulers(self, fm_shift1, fm_shift2, ddpm):
        """
        At matched-SNR points firmly inside the active zone, lambda must be 1.0
        for all three schedulers.  We find the matching timestep by scanning each
        scheduler's actual SNR curve (so shift-2's non-linear sigma mapping is
        handled correctly).
        """
        all_ts = torch.arange(1, 999, dtype=torch.long)

        test_snrs = [0.5, TEACHER_LAMBDA_SNR_FULL_LO * 3, TEACHER_LAMBDA_SNR_FULL_HI * 0.4]

        for target_snr in test_snrs:
            for sched, name in [(fm_shift1, "FM-shift1"), (fm_shift2, "FM-shift2"), (ddpm, "DDPM")]:
                snr_all = _snr_from_scheduler(all_ts, noise_scheduler=sched)
                idx     = (snr_all - target_snr).abs().argmin().item()
                ts      = all_ts[idx:idx+1]
                lam     = get_teacher_lambda(ts, _FakeArgs(), noise_scheduler=sched).item()
                assert abs(lam - 1.0) < 0.02, (
                    f"{name}: expected λ=1.0 inside active zone at SNR≈{target_snr:.3f} "
                    f"(nearest SNR={snr_all[idx].item():.4f}), got λ={lam:.4f}"
                )

    def test_generate_plot(self, fm_shift1, fm_shift2, ddpm):
        try:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt
        except ImportError:
            pytest.skip("matplotlib not installed")

        N  = 1000
        ts = torch.arange(1, N, dtype=torch.long)
        fig, axes = plt.subplots(1, 3, figsize=(18, 6))
        fig.suptitle(
            "Teacher λ rolloff vs log-SNR  (SNR-domain boundaries)\n"
            f"full-weight SNR ∈ [{TEACHER_LAMBDA_SNR_FULL_LO}, {TEACHER_LAMBDA_SNR_FULL_HI}]  │  "
            f"zero outside [{TEACHER_LAMBDA_SNR_ZERO_LO}, {TEACHER_LAMBDA_SNR_ZERO_HI}]",
            fontsize=12,
        )
        for ax, sched, title, color in [
            (axes[0], fm_shift1, "FM  shift=1  (σ = t/1000)",      "#1f77b4"),
            (axes[1], fm_shift2, "FM  shift=2  (compressed σ→t)",  "#ff7f0e"),
            (axes[2], ddpm,      "DDPM  (linear-β schedule)",       "#2ca02c"),
        ]:
            snr = _snr_from_scheduler(ts, sched)
            lam = get_teacher_lambda(ts, _FakeArgs(), noise_scheduler=sched).numpy()
            _plot_rolloff_vs_snr(ax, snr.numpy(), lam, color, ts.numpy())
            ax.set_title(title, fontsize=10)

        plt.tight_layout()
        plt.savefig(self.OUTPUT_PATH, dpi=120, bbox_inches="tight")
        plt.close(fig)
        print(f"\n  ✓  Plot written to {self.OUTPUT_PATH}")
        assert os.path.isfile(self.OUTPUT_PATH)


def _plot_rolloff_vs_snr(ax, snr, lam, color, t_values):
    import numpy as np
    log_snr = np.log10(np.clip(snr, 1e-5, None))
    ax.plot(log_snr, lam, color=color, linewidth=2)
    for snr_val, ls, col, desc in [
        (TEACHER_LAMBDA_SNR_ZERO_LO, ":",  "red",   f"ZERO_LO={TEACHER_LAMBDA_SNR_ZERO_LO}"),
        (TEACHER_LAMBDA_SNR_FULL_LO, "--", "red",   f"FULL_LO={TEACHER_LAMBDA_SNR_FULL_LO}"),
        (TEACHER_LAMBDA_SNR_FULL_HI, "--", "green", f"FULL_HI={TEACHER_LAMBDA_SNR_FULL_HI}"),
        (TEACHER_LAMBDA_SNR_ZERO_HI, ":",  "green", f"ZERO_HI={TEACHER_LAMBDA_SNR_ZERO_HI}"),
    ]:
        ax.axvline(math.log10(snr_val), color=col, linestyle=ls, linewidth=1.1, label=desc)
    ax.axvspan(math.log10(TEACHER_LAMBDA_SNR_FULL_LO),
               math.log10(TEACHER_LAMBDA_SNR_FULL_HI),
               alpha=0.07, color="green", label="active zone")
    for snr_mark in [TEACHER_LAMBDA_SNR_ZERO_LO, TEACHER_LAMBDA_SNR_FULL_LO,
                     0.5, TEACHER_LAMBDA_SNR_FULL_HI, TEACHER_LAMBDA_SNR_ZERO_HI]:
        idx = int(np.argmin(np.abs(snr - snr_mark)))
        x = math.log10(max(float(snr[idx]), 1e-5))
        ax.annotate(f" t≈{int(t_values[idx])}", xy=(x, float(lam[idx])), fontsize=7, color="#444")
    ax.set_xlabel("log₁₀(SNR)   [← high noise | low noise →]", fontsize=9)
    ax.set_ylabel("teacher λ", fontsize=9)
    ax.set_ylim(-0.05, 1.15)
    ax.legend(fontsize=7, loc="upper left")
    ax.grid(True, alpha=0.3)
    ax2 = ax.twiny()
    ax2.set_xlim(ax.get_xlim())
    ticks = np.array([-3, -2, -1, 0, 1, 2])
    ax2.set_xticks(ticks)
    ax2.set_xticklabels([f"10^{int(v)}" for v in ticks], fontsize=7)
    ax2.set_xlabel("SNR", fontsize=8)


# ── Main entry-point for direct execution ─────────────────────────────────────

if __name__ == "__main__":
    print("Building schedulers …")
    fm1    = _make_fm_scheduler(shift=1.0)
    fm2    = _make_fm_scheduler(shift=2.0)
    ddpm_s = _make_ddpm_scheduler()
    N  = 1000
    ts = torch.arange(1, N, dtype=torch.long)

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("pip install matplotlib")
        sys.exit(1)

    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    fig.suptitle(
        f"Teacher λ rolloff vs log-SNR  "
        f"(active SNR∈[{TEACHER_LAMBDA_SNR_FULL_LO},{TEACHER_LAMBDA_SNR_FULL_HI}]  "
        f"zero outside [{TEACHER_LAMBDA_SNR_ZERO_LO},{TEACHER_LAMBDA_SNR_ZERO_HI}])",
        fontsize=12,
    )
    for ax, sched, title, color in [
        (axes[0], fm1,    "FM  shift=1",   "#1f77b4"),
        (axes[1], fm2,    "FM  shift=2",   "#ff7f0e"),
        (axes[2], ddpm_s, "DDPM linear-β", "#2ca02c"),
    ]:
        snr = _snr_from_scheduler(ts, sched)
        lam = get_teacher_lambda(ts, _FakeArgs(), noise_scheduler=sched).numpy()
        _plot_rolloff_vs_snr(ax, snr.numpy(), lam, color, ts.numpy())
        ax.set_title(title)
    plt.tight_layout()
    out = "/tmp/teacher_lambda_rolloff.png"
    plt.savefig(out, dpi=120, bbox_inches="tight")
    print(f"Saved → {out}")




