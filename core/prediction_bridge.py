"""
Cross-objective teacher distillation bridges.

See TEACHER_BRIDGE_DESIGN.md for design rationale and full documentation.

Public API
----------
  get_prediction_bridge(teacher_type, student_type) -> PredictionBridge
  PredictionBridge  (ABC)
  IdentityBridge, VPredToFMBridge, EpsilonToFMBridge,
  FMToVPredBridge, FMToEpsilonBridge,
  EpsilonToVPredBridge, VPredToEpsilonBridge

Internal helpers (exported for tests)
--------------------------------------
  _snr_from_flowmatch_sigma, _snr_from_ddpm_alpha_bar
  _ddpm_timesteps_matching_fm_sigma, _fm_timesteps_matching_ddpm_timestep
  _recover_x0_from_epsilon, _recover_x0_from_vpred, _recover_x0_from_fm_velocity
  _x0_to_epsilon, _x0_to_vpred, _x0_to_fm_velocity
"""
from __future__ import annotations

from abc import ABC, abstractmethod

import torch


# ─── Noisy-latent construction ────────────────────────────────────────────────
# Minimal copy that avoids importing from core.loss (would create a circular
# dependency once loss.py imports from this module).

def _build_noisy_latents(
    clean_latents: torch.Tensor,
    noise: torch.Tensor,
    scheduler,
    timesteps: torch.Tensor,
) -> torch.Tensor:
    """
    Build x_t from clean latents and noise using *scheduler*.
    latents_perturbation is always 0 for teacher-target construction.
    """
    if hasattr(scheduler, "add_noise"):
        return scheduler.add_noise(clean_latents, noise, timesteps)
    elif hasattr(scheduler, "scale_noise"):
        return scheduler.scale_noise(clean_latents, timesteps, noise)
    else:
        raise RuntimeError(
            "Scheduler has no method to build noisy latents "
            "(tried .add_noise() and .scale_noise())"
        )


# ─── Shared SNR helpers ───────────────────────────────────────────────────────

def _snr_from_flowmatch_sigma(sigma: torch.Tensor) -> torch.Tensor:
    """SNR for FM interpolant x_t = (1-σ)x₁ + σε  →  SNR = (1-σ)²/σ²"""
    return ((1.0 - sigma) / sigma.clamp(min=1e-8)) ** 2


def _snr_from_ddpm_alpha_bar(alpha_bar: torch.Tensor) -> torch.Tensor:
    """SNR for DDPM interpolant x_t = √ᾱ·x₀ + √(1-ᾱ)·ε  →  SNR = ᾱ/(1-ᾱ)"""
    return alpha_bar / (1.0 - alpha_bar).clamp(min=1e-8)


def _ddpm_timesteps_matching_fm_sigma(
    fm_sigmas: torch.Tensor,
    ddpm_scheduler,
) -> torch.Tensor:
    """
    SNR-match FM sigmas → nearest DDPM integer timestep.
    Shared by VPredToFMBridge and EpsilonToFMBridge.
    """
    snr_fm = _snr_from_flowmatch_sigma(fm_sigmas.float())
    alpha_bar_target = snr_fm / (1.0 + snr_fm)             # [B]
    # [1000] vs [B, 1]  →  [B, 1000]
    ddpm_ts = torch.argmin(
        (ddpm_scheduler.alphas_cumprod.cpu()
         - alpha_bar_target.cpu().unsqueeze(-1)).abs(),
        dim=-1,
    )
    return ddpm_ts.to(fm_sigmas.device)


def _fm_timesteps_matching_ddpm_timestep(
    ddpm_timesteps: torch.Tensor,
    ddpm_scheduler,
    fm_scheduler,
) -> torch.Tensor:
    """
    SNR-match DDPM integer timestep → nearest FM shifted-float timestep.
    Shared by FMToVPredBridge and FMToEpsilonBridge.
    Derived from: σ = 1 / (1 + √SNR_DDPM).
    """
    ac = ddpm_scheduler.alphas_cumprod.to(ddpm_timesteps.device)
    alpha_bar = ac[ddpm_timesteps].float()
    snr_ddpm = _snr_from_ddpm_alpha_bar(alpha_bar)
    sigma_target = 1.0 / (1.0 + snr_ddpm.sqrt().clamp(min=1e-8))   # [B], in (0,1)

    # Convert sigma ∈ (0,1) to FM timestep scale (sigmas * 1000 ≈ timestep indices)
    fm_sigmas = fm_scheduler.get_sigmas_for_timesteps(
        fm_scheduler.timesteps.to(fm_scheduler.timesteps.device)
    ).float()   # [N]
    # [B, 1] - [1, N]  →  [B, N]
    nearest_idx = (
        fm_sigmas.cpu().unsqueeze(0) - sigma_target.cpu().unsqueeze(1)
    ).abs().argmin(dim=-1)
    return fm_scheduler.timesteps[nearest_idx].to(ddpm_timesteps.device)


# ─── x₀ recovery helpers ─────────────────────────────────────────────────────

def _recover_x0_from_epsilon(
    x_t: torch.Tensor,
    epsilon: torch.Tensor,
    alpha_bar: torch.Tensor,
) -> torch.Tensor:
    """x₀ = (x_t − √(1-ᾱ)·ε) / √ᾱ"""
    sqrt_ab   = alpha_bar.sqrt().view(-1, 1, 1, 1)
    sqrt_1mab = (1.0 - alpha_bar).sqrt().view(-1, 1, 1, 1)
    return (x_t - sqrt_1mab * epsilon) / sqrt_ab.clamp(min=1e-8)


def _recover_x0_from_vpred(
    x_t: torch.Tensor,
    v: torch.Tensor,
    alpha_bar: torch.Tensor,
) -> torch.Tensor:
    """x₀ = √ᾱ·x_t − √(1-ᾱ)·v"""
    sqrt_ab   = alpha_bar.sqrt().view(-1, 1, 1, 1)
    sqrt_1mab = (1.0 - alpha_bar).sqrt().view(-1, 1, 1, 1)
    return sqrt_ab * x_t - sqrt_1mab * v


def _recover_x0_from_fm_velocity(
    x_t: torch.Tensor,
    v: torch.Tensor,
    sigma: torch.Tensor,
) -> torch.Tensor:
    """FM code convention v = ε − x₁  →  x₁ = x_t − σ·v"""
    return x_t - sigma.view(-1, 1, 1, 1) * v


# ─── Target conversion helpers ────────────────────────────────────────────────

def _x0_to_epsilon(
    x_t: torch.Tensor,
    x0: torch.Tensor,
    alpha_bar: torch.Tensor,
) -> torch.Tensor:
    """ε = (x_t − √ᾱ·x₀) / √(1-ᾱ)"""
    sqrt_ab   = alpha_bar.sqrt().view(-1, 1, 1, 1)
    sqrt_1mab = (1.0 - alpha_bar).sqrt().view(-1, 1, 1, 1)
    return (x_t - sqrt_ab * x0) / sqrt_1mab.clamp(min=1e-8)


def _x0_to_vpred(
    epsilon: torch.Tensor,
    x0: torch.Tensor,
    alpha_bar: torch.Tensor,
) -> torch.Tensor:
    """v = √ᾱ·ε − √(1-ᾱ)·x₀"""
    sqrt_ab   = alpha_bar.sqrt().view(-1, 1, 1, 1)
    sqrt_1mab = (1.0 - alpha_bar).sqrt().view(-1, 1, 1, 1)
    return sqrt_ab * epsilon - sqrt_1mab * x0


def _x0_to_fm_velocity(
    noise: torch.Tensor,
    x0: torch.Tensor,
) -> torch.Tensor:
    """FM code convention: target = ε − x₁"""
    return noise - x0


# ─── Abstract base ────────────────────────────────────────────────────────────

class PredictionBridge(ABC):
    """
    Bridges one noise-prediction objective to another for teacher distillation.

    The three steps of a bridge, in calling order:
      1. remap_timesteps      – student t → teacher t at the same SNR level
      2. build_noisy_latents  – (x₁, ε, t_teacher) → x_t in teacher schedule
      3. convert_output       – teacher raw output → student target tensor

    Both teacher_timesteps and student_timesteps are passed to convert_output
    so that every bridge can look up the alpha_bar it needs from its own
    scheduler without external state.  Bridges that only need one set simply
    ignore the other.

    All tensors are full-batch [B, ...]; implementations should be vectorised.
    """

    @abstractmethod
    def remap_timesteps(
        self,
        student_timesteps: torch.Tensor,   # [B]
        student_scheduler,
        teacher_scheduler,
    ) -> torch.Tensor:
        """
        Return teacher-domain timesteps whose SNR matches the student timesteps.
        Identity for same-schedule crossings; SNR-lookup for cross-schedule ones.
        """

    @abstractmethod
    def build_noisy_latents(
        self,
        clean_latents: torch.Tensor,       # [B, C, H, W]
        noise: torch.Tensor,               # [B, C, H, W]
        teacher_timesteps: torch.Tensor,   # [B], teacher-domain
        teacher_scheduler,
    ) -> torch.Tensor:
        """Return x_t constructed with the teacher's noising schedule."""

    @abstractmethod
    def convert_output(
        self,
        teacher_output: torch.Tensor,          # [B, C, H, W] raw UNet .sample
        teacher_noisy_latents: torch.Tensor,   # [B, C, H, W]
        teacher_timesteps: torch.Tensor,       # [B], teacher-domain
        student_timesteps: torch.Tensor,       # [B], student-domain
        noise: torch.Tensor,                   # [B, C, H, W]  (shared ε)
        teacher_sched,
        student_sched,
    ) -> torch.Tensor:
        """Convert the teacher's raw UNet output into the student's target tensor."""


# ─── IdentityBridge ───────────────────────────────────────────────────────────

class IdentityBridge(PredictionBridge):
    """Teacher and student share the same prediction type — no conversion needed."""

    def remap_timesteps(self, student_ts, student_sched, teacher_sched):
        return student_ts

    def build_noisy_latents(self, clean, noise, teacher_ts, teacher_sched):
        return _build_noisy_latents(clean, noise, teacher_sched, teacher_ts)

    def convert_output(self, teacher_output, teacher_noisy_latents, teacher_timesteps,
                       student_timesteps, noise, teacher_sched, student_sched):
        return teacher_output


# ─── DDPM → FM crossings (SNR-match integer ts → FM float ts) ─────────────────

class VPredToFMBridge(PredictionBridge):
    """v-prediction teacher  →  flow-matching student"""

    def remap_timesteps(self, student_ts, student_sched, teacher_sched):
        fm_sigmas = student_sched.get_sigmas_for_timesteps(
            student_ts.to(student_sched.timesteps.device)
        )
        return _ddpm_timesteps_matching_fm_sigma(fm_sigmas, teacher_sched)

    def build_noisy_latents(self, clean, noise, teacher_ts, teacher_sched):
        return _build_noisy_latents(clean, noise, teacher_sched, teacher_ts)

    def convert_output(self, teacher_output, teacher_noisy_latents, teacher_timesteps,
                       student_timesteps, noise, teacher_sched, student_sched):
        ac = teacher_sched.alphas_cumprod.to(teacher_timesteps.device)
        x0 = _recover_x0_from_vpred(
            teacher_noisy_latents.float(), teacher_output.float(), ac[teacher_timesteps].float()
        )
        return _x0_to_fm_velocity(noise.float(), x0)


class EpsilonToFMBridge(PredictionBridge):
    """epsilon teacher  →  flow-matching student"""

    # Epsilon models are unreliable below this DDPM timestep — x₀ recovery
    # blows up because √(1-ᾱ) → 0 makes the division numerically unstable.
    #_MIN_TEACHER_TIMESTEP = 50  # tune: 20–100 depending on your teacher
    _MIN_TEACHER_TIMESTEP = 0

    def remap_timesteps(self, student_ts, student_sched, teacher_sched):
        fm_sigmas = student_sched.get_sigmas_for_timesteps(
            student_ts.to(student_sched.timesteps.device)
        )
        ts = _ddpm_timesteps_matching_fm_sigma(fm_sigmas, teacher_sched)
        return ts.clamp(min=self._MIN_TEACHER_TIMESTEP)  # clamp to avoid unstable timesteps

    def build_noisy_latents(self, clean, noise, teacher_ts, teacher_sched):
        return _build_noisy_latents(clean, noise, teacher_sched, teacher_ts)

    def convert_output(self, teacher_output, teacher_noisy_latents, teacher_timesteps,
                       student_timesteps, noise, teacher_sched, student_sched):
        ac = teacher_sched.alphas_cumprod.to(teacher_timesteps.device)
        ab = ac[teacher_timesteps].float()

        # Guard: at very low t, √(1-ᾱ) is tiny and amplifies error catastrophically.
        # Fall back to a scaled noise target rather than blowing up x₀ recovery.
        sqrt_1mab = (1.0 - ab).sqrt().view(-1, 1, 1, 1)
        safe_mask = (sqrt_1mab > 0.1).squeeze()  # [B] bool

        x0 = _recover_x0_from_epsilon(
            teacher_noisy_latents.float(), teacher_output.float(), ab
        )
        fm_vel = _x0_to_fm_velocity(noise.float(), x0)

        # For unsafe (very low noise) samples, substitute clean-latent target
        # (velocity ≈ ε − x₀_true, use teacher x₀ directly without amplification)
        if not safe_mask.all():
            # x₀ ≈ teacher_noisy_latents at very low t — use it directly
            fallback = _x0_to_fm_velocity(noise.float(), teacher_noisy_latents.float())
            fm_vel[~safe_mask] = fallback[~safe_mask]

        return fm_vel


# ─── FM → DDPM crossings (SNR-match FM float ts → integer ts) ─────────────────

class FMToVPredBridge(PredictionBridge):
    """flow-matching teacher  →  v-prediction student"""

    def remap_timesteps(self, student_ts, student_sched, teacher_sched):
        # student_ts are DDPM integer indices; find FM timestep with matching SNR
        return _fm_timesteps_matching_ddpm_timestep(student_ts, student_sched, teacher_sched)

    def build_noisy_latents(self, clean, noise, teacher_ts, teacher_sched):
        # teacher_ts are FM-shifted floats after remap_timesteps
        return _build_noisy_latents(clean, noise, teacher_sched, teacher_ts)

    def convert_output(self, teacher_output, teacher_noisy_latents, teacher_timesteps,
                       student_timesteps, noise, teacher_sched, student_sched):
        fm_sigmas = teacher_sched.get_sigmas_for_timesteps(
            teacher_timesteps.to(teacher_sched.timesteps.device)
        ).float()
        x0 = _recover_x0_from_fm_velocity(
            teacher_noisy_latents.float(), teacher_output.float(), fm_sigmas
        )
        # student_timesteps are DDPM integer indices → look up student alpha_bar
        ac = student_sched.alphas_cumprod.to(student_timesteps.device)
        return _x0_to_vpred(noise.float(), x0, ac[student_timesteps].float())


class FMToEpsilonBridge(PredictionBridge):
    """flow-matching teacher  →  epsilon student"""

    def remap_timesteps(self, student_ts, student_sched, teacher_sched):
        return _fm_timesteps_matching_ddpm_timestep(student_ts, student_sched, teacher_sched)

    def build_noisy_latents(self, clean, noise, teacher_ts, teacher_sched):
        return _build_noisy_latents(clean, noise, teacher_sched, teacher_ts)

    def convert_output(self, teacher_output, teacher_noisy_latents, teacher_timesteps,
                       student_timesteps, noise, teacher_sched, student_sched):
        fm_sigmas = teacher_sched.get_sigmas_for_timesteps(
            teacher_timesteps.to(teacher_sched.timesteps.device)
        ).float()
        x0 = _recover_x0_from_fm_velocity(
            teacher_noisy_latents.float(), teacher_output.float(), fm_sigmas
        )
        ac = student_sched.alphas_cumprod.to(student_timesteps.device)
        ab = ac[student_timesteps].float()
        sqrt_ab   = ab.sqrt().view(-1, 1, 1, 1)
        sqrt_1mab = (1.0 - ab).sqrt().view(-1, 1, 1, 1)
        x_t_student = sqrt_ab * x0.detach() + sqrt_1mab * noise.float()
        return _x0_to_epsilon(x_t_student, x0, ab)


# ─── Pure DDPM crossings (same integer timestep scale, no SNR mapping) ─────────

class EpsilonToVPredBridge(PredictionBridge):
    """epsilon teacher  →  v-prediction student  (same DDPM schedule)"""

    def remap_timesteps(self, student_ts, student_sched, teacher_sched):
        return student_ts

    def build_noisy_latents(self, clean, noise, teacher_ts, teacher_sched):
        return _build_noisy_latents(clean, noise, teacher_sched, teacher_ts)

    def convert_output(self, teacher_output, teacher_noisy_latents, teacher_timesteps,
                       student_timesteps, noise, teacher_sched, student_sched):
        ac = teacher_sched.alphas_cumprod.to(teacher_timesteps.device)
        x0 = _recover_x0_from_epsilon(
            teacher_noisy_latents.float(), teacher_output.float(), ac[teacher_timesteps].float()
        )
        return _x0_to_vpred(noise.float(), x0, ac[teacher_timesteps].float())


class VPredToEpsilonBridge(PredictionBridge):
    """v-prediction teacher  →  epsilon student  (same DDPM schedule)"""

    def remap_timesteps(self, student_ts, student_sched, teacher_sched):
        return student_ts

    def build_noisy_latents(self, clean, noise, teacher_ts, teacher_sched):
        return _build_noisy_latents(clean, noise, teacher_sched, teacher_ts)

    def convert_output(self, teacher_output, teacher_noisy_latents, teacher_timesteps,
                       student_timesteps, noise, teacher_sched, student_sched):
        ac = teacher_sched.alphas_cumprod.to(teacher_timesteps.device)
        x0 = _recover_x0_from_vpred(
            teacher_noisy_latents.float(), teacher_output.float(), ac[teacher_timesteps].float()
        )
        return _x0_to_epsilon(teacher_noisy_latents.float(), x0, ac[teacher_timesteps].float())


# ─── Factory ──────────────────────────────────────────────────────────────────

def get_prediction_bridge(
    teacher_prediction_type: str,
    student_prediction_type: str,
) -> PredictionBridge:
    """
    Return the appropriate PredictionBridge for the given objective crossing.
    Raises ValueError for unsupported or unrecognised prediction types.
    """
    teacher = _normalise_pred_type(teacher_prediction_type)
    student = _normalise_pred_type(student_prediction_type)

    if teacher == student:
        return IdentityBridge()

    _BRIDGE_MAP = {
        ("v_prediction",    "flow_prediction"): VPredToFMBridge,
        ("flow_prediction", "v_prediction"):    FMToVPredBridge,
        ("epsilon",         "flow_prediction"): EpsilonToFMBridge,
        ("flow_prediction", "epsilon"):         FMToEpsilonBridge,
        ("epsilon",         "v_prediction"):    EpsilonToVPredBridge,
        ("v_prediction",    "epsilon"):         VPredToEpsilonBridge,
    }
    key = (teacher, student)
    if key not in _BRIDGE_MAP:
        raise ValueError(
            f"Unsupported teacher→student prediction crossing: {teacher} → {student}"
        )
    return _BRIDGE_MAP[key]()

