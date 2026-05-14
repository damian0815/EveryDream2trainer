# Cross-Objective Teacher Distillation — Design Plan

## 1. Problem Statement

The current teacher pipeline in `core/loss.py` (`get_teacher_target`) mixes three
distinct concerns in a single function:

| Concern | What it does | Current home |
|---|---|---|
| Timestep/SNR mapping | Find teacher t such that SNR(t_teacher) ≈ SNR(t_student) | inline in `get_teacher_target` / `_remap_noise_v_pred_to_flow_matching` |
| Noisy-latent construction | Build x_t in the teacher's coordinate system | `_get_noisy_latents` (shared) |
| Output-space conversion | Map teacher's raw UNet output → student target | `_convert_model_output` |

This works for the single implemented crossing (v-pred teacher → FM student) but makes
adding new crossings messy and error-prone.

### Currently supported crossings

| Teacher | Student | Works? | Path |
|---|---|---|---|
| FM | FM | ✅ | identity |
| ε | ε | ✅ | identity |
| v | v | ✅ | identity |
| v → FM | ✅ | `_remap_noise_v_pred_to_flow_matching` + `_convert_model_output` |
| v → ε | ✅ | `_convert_model_output` (same DDPM ts) |
| ε → v | ✅ | `_convert_model_output` (same DDPM ts) |
| **ε → FM** | ❌ | No SNR-match for DDPM→FM; `_convert_model_output` only covers v→FM |
| **FM → ε** | ❌ | No bridge at all |
| **FM → v** | ❌ | No bridge at all |

### Known bugs in the current code

1. **Disjoint-check is inverted** (`get_teacher_target`, line ~1143):
   ```python
   if supported_prediction_interconversion_types.isdisjoint(present_prediction_types):
       raise ValueError(...)
   ```
   `isdisjoint` returns `True` when the sets share *no* elements, so `{FM, FM}`
   (disjoint from the `{ε, v}` set) would incorrectly raise, while `{FM, ε}` (which
   should raise because FM isn't in the supported set) would pass silently and crash
   later.  The guard should use
   `not present_prediction_types.issubset(supported_prediction_interconversion_types)`.
   After the refactor this guard is removed entirely — the factory raises cleanly.

2. **SNR-match missing for ε → FM**: `_remap_noise_v_pred_to_flow_matching` only
   handles v-pred → FM; epsilon uses a different `add_noise` API (needs `alphas_cumprod`
   lookup, not a special API call).

3. **Hard assert in `_convert_model_output`** for ε paths:
   ```python
   assert student_timesteps == teacher_unet_timesteps
   ```
   This works when types are the same (DDPM ↔ DDPM) but will always fail for any
   cross-schedule conversion.

---

## 2. Design: `PredictionBridge` abstraction

### 2.1 Shared pure helpers (module-level, no class)

All conversion maths lives in stateless functions so every step is independently
unit-testable:

```python
# core/prediction_bridge.py

def _snr_from_flowmatch_sigma(sigma: torch.Tensor) -> torch.Tensor:
    """SNR for FM interpolant x_t = (1-σ)x₁ + σε  →  SNR = (1-σ)²/σ²"""
    return ((1 - sigma) / sigma.clamp(min=1e-8)) ** 2


def _snr_from_ddpm_alpha_bar(alpha_bar: torch.Tensor) -> torch.Tensor:
    """SNR for DDPM interpolant x_t = √ᾱ·x₀ + √(1-ᾱ)·ε  →  SNR = ᾱ/(1-ᾱ)"""
    return alpha_bar / (1 - alpha_bar).clamp(min=1e-8)


def _ddpm_timesteps_matching_fm_sigma(
    fm_sigmas: torch.Tensor,
    ddpm_scheduler: SchedulerMixin,
) -> torch.Tensor:
    """
    SNR-match FM sigmas → nearest DDPM integer timestep.
    Shared by VPredToFMBridge and EpsilonToFMBridge.
    """
    snr_fm = _snr_from_flowmatch_sigma(fm_sigmas)
    alpha_bar_target = snr_fm / (1 + snr_fm)
    ddpm_ts = torch.argmin(
        (ddpm_scheduler.alphas_cumprod.cpu() - alpha_bar_target.unsqueeze(-1).cpu()).abs(),
        dim=-1,
    )
    return ddpm_ts.to(fm_sigmas.device)


def _fm_timesteps_matching_ddpm_timestep(
    ddpm_timesteps: torch.Tensor,
    ddpm_scheduler: SchedulerMixin,
    fm_scheduler,                # TrainFlowMatchEulerDiscreteScheduler
) -> torch.Tensor:
    """
    SNR-match DDPM integer timestep → nearest FM shifted-float timestep.
    Shared by FMToVPredBridge and FMToEpsilonBridge.
    Derived from: σ = 1 / (1 + √SNR_DDPM).
    """
    ac = ddpm_scheduler.alphas_cumprod.to(ddpm_timesteps.device)
    alpha_bar = ac[ddpm_timesteps]
    snr_ddpm = _snr_from_ddpm_alpha_bar(alpha_bar)
    sigma_target = 1.0 / (1.0 + snr_ddpm.sqrt().clamp(min=1e-8))
    fm_sigmas = fm_scheduler.get_sigmas_for_timesteps(fm_scheduler.timesteps)
    nearest_idx = (
        fm_sigmas.cpu().unsqueeze(0) - sigma_target.cpu().unsqueeze(1)
    ).abs().argmin(dim=-1)
    return fm_scheduler.timesteps[nearest_idx].to(ddpm_timesteps.device)


def _recover_x0_from_epsilon(
    x_t: torch.Tensor, epsilon: torch.Tensor, alpha_bar: torch.Tensor,
) -> torch.Tensor:
    """x₀ = (x_t − √(1-ᾱ)·ε) / √ᾱ"""
    sqrt_ab   = alpha_bar.sqrt().view(-1, 1, 1, 1)
    sqrt_1mab = (1 - alpha_bar).sqrt().view(-1, 1, 1, 1)
    return (x_t - sqrt_1mab * epsilon) / sqrt_ab.clamp(min=1e-8)


def _recover_x0_from_vpred(
    x_t: torch.Tensor, v: torch.Tensor, alpha_bar: torch.Tensor,
) -> torch.Tensor:
    """x₀ = √ᾱ·x_t − √(1-ᾱ)·v"""
    sqrt_ab   = alpha_bar.sqrt().view(-1, 1, 1, 1)
    sqrt_1mab = (1 - alpha_bar).sqrt().view(-1, 1, 1, 1)
    return sqrt_ab * x_t - sqrt_1mab * v


def _recover_x0_from_fm_velocity(
    x_t: torch.Tensor, v: torch.Tensor, sigma: torch.Tensor,
) -> torch.Tensor:
    """FM code convention v = ε − x₁  →  x₁ = x_t − σ·v"""
    return x_t - sigma.view(-1, 1, 1, 1) * v


def _x0_to_epsilon(
    x_t: torch.Tensor, x0: torch.Tensor, alpha_bar: torch.Tensor,
) -> torch.Tensor:
    """ε = (x_t − √ᾱ·x₀) / √(1-ᾱ)"""
    sqrt_ab   = alpha_bar.sqrt().view(-1, 1, 1, 1)
    sqrt_1mab = (1 - alpha_bar).sqrt().view(-1, 1, 1, 1)
    return (x_t - sqrt_ab * x0) / sqrt_1mab.clamp(min=1e-8)


def _x0_to_vpred(
    epsilon: torch.Tensor, x0: torch.Tensor, alpha_bar: torch.Tensor,
) -> torch.Tensor:
    """v = √ᾱ·ε − √(1-ᾱ)·x₀"""
    sqrt_ab   = alpha_bar.sqrt().view(-1, 1, 1, 1)
    sqrt_1mab = (1 - alpha_bar).sqrt().view(-1, 1, 1, 1)
    return sqrt_ab * epsilon - sqrt_1mab * x0


def _x0_to_fm_velocity(noise: torch.Tensor, x0: torch.Tensor) -> torch.Tensor:
    """FM code convention: target = ε − x₁"""
    return noise - x0
```

### 2.2 Interface

`convert_output` receives **both** `teacher_timesteps` and `student_timesteps`.
Bridges that use only one set simply ignore the other, but the contract is uniform so
no bridge ever needs to reach outside its own arguments for schedule lookups.

```python
# core/prediction_bridge.py  (continued)

from abc import ABC, abstractmethod

class PredictionBridge(ABC):
    """
    Bridges one noise-prediction objective to another for teacher distillation.

    The three steps of a bridge, in calling order:
      1. remap_timesteps      – student t → teacher t at the same SNR level
      2. build_noisy_latents  – (x₁, ε, t_teacher) → x_t in teacher schedule
      3. convert_output       – teacher raw output → student target tensor

    All tensors are full-batch [B, ...]; implementations should be vectorised.
    """

    @abstractmethod
    def remap_timesteps(
        self,
        student_timesteps: torch.Tensor,   # [B]
        student_scheduler: SchedulerMixin,
        teacher_scheduler: SchedulerMixin,
    ) -> torch.Tensor:
        """
        Return teacher-domain timesteps whose SNR matches the student timesteps.
        Identity for same-schedule crossings; SNR-lookup for cross-schedule crossings.
        """

    @abstractmethod
    def build_noisy_latents(
        self,
        clean_latents: torch.Tensor,       # [B, C, H, W]
        noise: torch.Tensor,               # [B, C, H, W]
        teacher_timesteps: torch.Tensor,   # [B], teacher-domain
        teacher_scheduler: SchedulerMixin,
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
        teacher_scheduler: SchedulerMixin,
        student_scheduler: SchedulerMixin,
    ) -> torch.Tensor:
        """
        Convert the teacher's raw UNet output into the student's target tensor.

        Both teacher_timesteps and student_timesteps are provided so that bridges
        spanning different schedules (e.g. FM→v, FM→ε) can look up alpha_bar from the
        student scheduler without needing any external state or extra arguments.
        """
```

### 2.3 Factory / registry

```python
# core/prediction_bridge.py  (continued)

def get_prediction_bridge(
    teacher_prediction_type: str,   # "epsilon" | "v_prediction" | "flow_prediction"
    student_prediction_type: str,
) -> PredictionBridge:
    """
    Return the appropriate bridge for the given objective crossing.

    Raises ValueError for unsupported or unrecognised prediction types.
    """
    teacher = _normalise_pred_type(teacher_prediction_type)
    student = _normalise_pred_type(student_prediction_type)

    if teacher == student:
        return IdentityBridge()

    bridge_map = {
        ("v_prediction",   "flow_prediction"): VPredToFMBridge,
        ("flow_prediction","v_prediction"):    FMToVPredBridge,
        ("epsilon",        "flow_prediction"): EpsilonToFMBridge,
        ("flow_prediction","epsilon"):         FMToEpsilonBridge,
        ("epsilon",        "v_prediction"):    EpsilonToVPredBridge,
        ("v_prediction",   "epsilon"):         VPredToEpsilonBridge,
    }
    key = (teacher, student)
    if key not in bridge_map:
        raise ValueError(
            f"Unsupported teacher→student prediction crossing: {teacher} → {student}"
        )
    return bridge_map[key]()


def _normalise_pred_type(t: str) -> str:
    if t in ("v-prediction", "v_prediction"):
        return "v_prediction"
    if t in ("flow-matching", "flow_prediction", "flow_match"):
        return "flow_prediction"
    if t == "epsilon":
        return "epsilon"
    raise ValueError(f"Unrecognised prediction type: {t!r}")
```

### 2.4 Concrete bridge implementations

All implementations are thin wrappers around the shared helpers in §2.1.

```python
# core/prediction_bridge.py  (continued)

class IdentityBridge(PredictionBridge):
    """Teacher and student share the same prediction type — no conversion needed."""

    def remap_timesteps(self, student_ts, student_sched, teacher_sched):
        return student_ts

    def build_noisy_latents(self, clean, noise, teacher_ts, teacher_sched):
        return _get_noisy_latents(clean, noise, teacher_sched, teacher_ts, latents_perturbation=0)

    def convert_output(self, teacher_out, teacher_noisy, teacher_ts, student_ts,
                       noise, teacher_sched, student_sched):
        return teacher_out


# ─── DDPM → FM crossings (SNR-match integer ts → FM float ts) ─────────────────

class VPredToFMBridge(PredictionBridge):
    """v-prediction teacher  →  flow-matching student"""

    def remap_timesteps(self, student_ts, student_sched, teacher_sched):
        fm_sigmas = student_sched.get_sigmas_for_timesteps(
            student_ts.to(student_sched.timesteps.device)
        )
        return _ddpm_timesteps_matching_fm_sigma(fm_sigmas, teacher_sched)

    def build_noisy_latents(self, clean, noise, teacher_ts, teacher_sched):
        return _get_noisy_latents(clean, noise, teacher_sched, teacher_ts, latents_perturbation=0)

    def convert_output(self, teacher_out, teacher_noisy, teacher_ts, student_ts,
                       noise, teacher_sched, student_sched):
        ac = teacher_sched.alphas_cumprod.to(teacher_ts.device)
        x0 = _recover_x0_from_vpred(teacher_noisy, teacher_out, ac[teacher_ts])
        return _x0_to_fm_velocity(noise, x0)


class EpsilonToFMBridge(PredictionBridge):
    """epsilon teacher  →  flow-matching student"""

    def remap_timesteps(self, student_ts, student_sched, teacher_sched):
        fm_sigmas = student_sched.get_sigmas_for_timesteps(
            student_ts.to(student_sched.timesteps.device)
        )
        return _ddpm_timesteps_matching_fm_sigma(fm_sigmas, teacher_sched)

    def build_noisy_latents(self, clean, noise, teacher_ts, teacher_sched):
        return _get_noisy_latents(clean, noise, teacher_sched, teacher_ts, latents_perturbation=0)

    def convert_output(self, teacher_out, teacher_noisy, teacher_ts, student_ts,
                       noise, teacher_sched, student_sched):
        ac = teacher_sched.alphas_cumprod.to(teacher_ts.device)
        x0 = _recover_x0_from_epsilon(teacher_noisy, teacher_out, ac[teacher_ts])
        return _x0_to_fm_velocity(noise, x0)


# ─── FM → DDPM crossings (SNR-match FM float ts → integer ts) ─────────────────

class FMToVPredBridge(PredictionBridge):
    """flow-matching teacher  →  v-prediction student"""

    def remap_timesteps(self, student_ts, student_sched, teacher_sched):
        return _fm_timesteps_matching_ddpm_timestep(student_ts, student_sched, teacher_sched)

    def build_noisy_latents(self, clean, noise, teacher_ts, teacher_sched):
        # teacher_ts are FM-shifted floats after remap_timesteps
        return _get_noisy_latents(clean, noise, teacher_sched, teacher_ts, latents_perturbation=0)

    def convert_output(self, teacher_out, teacher_noisy, teacher_ts, student_ts,
                       noise, teacher_sched, student_sched):
        fm_sigmas = teacher_sched.get_sigmas_for_timesteps(
            teacher_ts.to(teacher_sched.timesteps.device)
        )
        x0 = _recover_x0_from_fm_velocity(teacher_noisy, teacher_out, fm_sigmas)
        # student_ts are DDPM integer indices → look up student alpha_bar
        ac = student_sched.alphas_cumprod.to(student_ts.device)
        return _x0_to_vpred(noise, x0, ac[student_ts])


class FMToEpsilonBridge(PredictionBridge):
    """flow-matching teacher  →  epsilon student"""

    def remap_timesteps(self, student_ts, student_sched, teacher_sched):
        return _fm_timesteps_matching_ddpm_timestep(student_ts, student_sched, teacher_sched)

    def build_noisy_latents(self, clean, noise, teacher_ts, teacher_sched):
        return _get_noisy_latents(clean, noise, teacher_sched, teacher_ts, latents_perturbation=0)

    def convert_output(self, teacher_out, teacher_noisy, teacher_ts, student_ts,
                       noise, teacher_sched, student_sched):
        fm_sigmas = teacher_sched.get_sigmas_for_timesteps(
            teacher_ts.to(teacher_sched.timesteps.device)
        )
        x0 = _recover_x0_from_fm_velocity(teacher_noisy, teacher_out, fm_sigmas)
        # Reconstruct the student's noisy latent from x0 and ε, then solve for ε:
        #   x_t_student = √ᾱ·x₀ + √(1-ᾱ)·ε
        #   ε = (x_t_student − √ᾱ·x₀) / √(1-ᾱ)
        # (simplifies to 'noise', but expressed via _x0_to_epsilon for consistency)
        ac = student_sched.alphas_cumprod.to(student_ts.device)
        ab = ac[student_ts]
        sqrt_ab   = ab.sqrt().view(-1, 1, 1, 1)
        sqrt_1mab = (1 - ab).sqrt().view(-1, 1, 1, 1)
        x_t_student = sqrt_ab * x0.detach() + sqrt_1mab * noise
        return _x0_to_epsilon(x_t_student, x0, ab)


# ─── Pure DDPM crossings (same integer timestep scale, no SNR mapping) ─────────

class EpsilonToVPredBridge(PredictionBridge):
    """epsilon teacher  →  v-prediction student  (same DDPM schedule)"""

    def remap_timesteps(self, student_ts, student_sched, teacher_sched):
        return student_ts

    def build_noisy_latents(self, clean, noise, teacher_ts, teacher_sched):
        return _get_noisy_latents(clean, noise, teacher_sched, teacher_ts, latents_perturbation=0)

    def convert_output(self, teacher_out, teacher_noisy, teacher_ts, student_ts,
                       noise, teacher_sched, student_sched):
        ac = teacher_sched.alphas_cumprod.to(teacher_ts.device)
        x0 = _recover_x0_from_epsilon(teacher_noisy, teacher_out, ac[teacher_ts])
        return _x0_to_vpred(noise, x0, ac[teacher_ts])


class VPredToEpsilonBridge(PredictionBridge):
    """v-prediction teacher  →  epsilon student  (same DDPM schedule)"""

    def remap_timesteps(self, student_ts, student_sched, teacher_sched):
        return student_ts

    def build_noisy_latents(self, clean, noise, teacher_ts, teacher_sched):
        return _get_noisy_latents(clean, noise, teacher_sched, teacher_ts, latents_perturbation=0)

    def convert_output(self, teacher_out, teacher_noisy, teacher_ts, student_ts,
                       noise, teacher_sched, student_sched):
        ac = teacher_sched.alphas_cumprod.to(teacher_ts.device)
        x0 = _recover_x0_from_vpred(teacher_noisy, teacher_out, ac[teacher_ts])
        return _x0_to_epsilon(teacher_noisy, x0, ac[teacher_ts])
```

---

## 3. Refactoring `get_teacher_target`

The new `get_teacher_target` is a clean four-step sequence with no per-crossing
branching.  The bridge encapsulates every difference.

```python
# core/loss.py  (replaces the existing get_teacher_target)

def get_teacher_target(
    teacher_model: TrainingModel,
    teacher_conditioning: Conditioning,
    student_model: TrainingModel,
    student_timesteps: torch.Tensor,    # [B], student-domain
    clean_image_latents: torch.Tensor,  # [B, C, H, W]
    noise: torch.Tensor,                # [B, C, H, W]
) -> torch.Tensor:
    """
    Unified teacher-target function.  Handles all 9 objective crossings via
    PredictionBridge, plus the cross-VAE interposer path (unchanged).
    """
    # ── Cross-VAE interposer (highest priority, unchanged) ─────────────────────
    latent_interposer = getattr(student_model, 'latent_interposer', None)
    if latent_interposer is not None and student_model.is_flow_matching:
        from core.latent_interposer import infer_latent_space_type
        src = infer_latent_space_type(student_model)
        dst = infer_latent_space_type(teacher_model)
        if src is not None and dst is not None and src != dst:
            return _teacher_target_via_interposer(...)   # unchanged

    # ── Unified same-VAE-space distillation ────────────────────────────────────
    teacher_pred_type = _get_prediction_type(teacher_model)
    student_pred_type = _get_prediction_type(student_model)

    bridge = get_prediction_bridge(teacher_pred_type, student_pred_type)

    # Step 1: map student timesteps → teacher timesteps (SNR-matched or identity)
    teacher_timesteps = bridge.remap_timesteps(
        student_timesteps,
        student_sched=student_model.noise_scheduler,
        teacher_sched=teacher_model.noise_scheduler,
    ).to(teacher_model.device)

    # Step 2: build teacher noisy latents
    teacher_noisy = bridge.build_noisy_latents(
        clean_image_latents, noise, teacher_timesteps,
        teacher_scheduler=teacher_model.noise_scheduler,
    )

    # Step 3: run teacher UNet (single call site, no per-crossing branching)
    with torch.no_grad():
        teacher_output = teacher_model.unet(
            teacher_noisy.to(teacher_model.device, dtype=teacher_model.unet.dtype),
            teacher_timesteps.to(teacher_model.device, dtype=teacher_model.unet.dtype),
            teacher_conditioning.prompt_embeds.to(
                teacher_model.device, dtype=teacher_model.unet.dtype
            ),
            added_cond_kwargs=(
                teacher_conditioning.get_added_cond_kwargs(dtype=teacher_model.unet.dtype)
                if teacher_model.is_sdxl else None
            ),
        ).sample.float()

    # Step 4: convert to student target space
    student_target = bridge.convert_output(
        teacher_output=teacher_output,
        teacher_noisy_latents=teacher_noisy,
        teacher_timesteps=teacher_timesteps,
        student_timesteps=student_timesteps,
        noise=noise,
        teacher_scheduler=teacher_model.noise_scheduler,
        student_scheduler=student_model.noise_scheduler,
    )
    return student_target.to(dtype=student_model.unet.dtype)
```

---

## 4. File layout and separation of concerns

```
core/
  prediction_bridge.py        ← NEW
    PredictionBridge            (ABC)
    IdentityBridge
    VPredToFMBridge
    EpsilonToFMBridge
    FMToVPredBridge
    FMToEpsilonBridge
    EpsilonToVPredBridge
    VPredToEpsilonBridge
    get_prediction_bridge()     (factory)
    _normalise_pred_type()
    _snr_from_*()               (shared SNR helpers)
    _ddpm_timesteps_matching_fm_sigma()
    _fm_timesteps_matching_ddpm_timestep()
    _recover_x0_from_*()        (shared x₀ recovery)
    _x0_to_*()                  (shared target conversion)

  loss.py
    get_teacher_target()        ← refactored to 4-step bridge dispatch
    _teacher_target_via_interposer()  ← unchanged (cross-VAE path)
    _get_noisy_latents()        ← unchanged (called by bridge.build_noisy_latents)
    [remove] _remap_noise_v_pred_to_flow_matching()
    [remove] _convert_model_output()
    [remove] _get_ddpm_timesteps_for_flowmatch_timesteps()
```

Each file has a single responsibility:
- `prediction_bridge.py` — all objective-conversion maths and the bridge abstraction.
- `loss.py` — training loss computation and teacher-target orchestration.
- `teacher.py` — model loading and scheduler configuration.
- `step.py` — per-step training logic (`get_teacher_lambda`, masking, accumulation).

---

## 5. Testing strategy

```
test/
  test_prediction_bridge.py
    test_identity_bridge_{fm,v,eps}()
    test_vpred_to_fm_snr_roundtrip()
    test_epsilon_to_fm_x0_roundtrip()
    test_fm_to_vpred_x0_roundtrip()
    test_fm_to_epsilon_x0_roundtrip()
    test_epsilon_to_vpred_x0_roundtrip()
    test_vpred_to_epsilon_x0_roundtrip()
    test_get_prediction_bridge_all_pairs()
    test_normalise_pred_type()
    test_snr_helpers_invertible()
```

Each crossing test follows the same pattern:
1. Fix x₁ and ε analytically.
2. Run the full teacher-simulation path: `remap_timesteps → build_noisy_latents →
   convert_output` with a synthetic teacher-output computed from the known x₁/ε.
3. Assert that the returned student target matches the ground-truth target computed
   directly from x₁ and ε for the student's objective.

---

## 6. Migration path (no-op for existing users)

1. Create `core/prediction_bridge.py` with the ABC + all eight bridge classes.
2. Add and unit-test the three currently missing crossings
   (`EpsilonToFMBridge`, `FMToVPredBridge`, `FMToEpsilonBridge`).
3. Refactor `get_teacher_target` in `loss.py` to the four-step dispatch (§3).
   Existing v-pred→FM users are routed through `VPredToFMBridge` — same SNR maths,
   just cleaner home.
4. Delete `_remap_noise_v_pred_to_flow_matching`, `_convert_model_output`, and
   `_get_ddpm_timesteps_for_flowmatch_timesteps` from `loss.py`.
5. No CLI changes required; `--teacher_prediction_type` already accepts all three
   objective names.

---

## 7. Summary table of all nine crossings

| Teacher | Student | Bridge | Timestep mapping | x₀ recovery | Final conversion |
|---|---|---|---|---|---|
| FM | FM | `IdentityBridge` | identity | — | identity |
| ε | ε | `IdentityBridge` | identity | — | identity |
| v | v | `IdentityBridge` | identity | — | identity |
| v → FM | `VPredToFMBridge` | DDPM→FM (SNR) | `_recover_x0_from_vpred` | `_x0_to_fm_velocity` |
| ε → FM | `EpsilonToFMBridge` | DDPM→FM (SNR) | `_recover_x0_from_epsilon` | `_x0_to_fm_velocity` |
| FM → v | `FMToVPredBridge` | FM→DDPM (SNR) | `_recover_x0_from_fm_velocity` | `_x0_to_vpred` (student ᾱ) |
| FM → ε | `FMToEpsilonBridge` | FM→DDPM (SNR) | `_recover_x0_from_fm_velocity` | `_x0_to_epsilon` (student ᾱ) |
| ε → v | `EpsilonToVPredBridge` | identity (same DDPM ts) | `_recover_x0_from_epsilon` | `_x0_to_vpred` |
| v → ε | `VPredToEpsilonBridge` | identity (same DDPM ts) | `_recover_x0_from_vpred` | `_x0_to_epsilon` |

All bridges share:
- A single `_get_noisy_latents` call for noisy-latent construction.
- A single teacher UNet forward call in `get_teacher_target`.
- Pure stateless helper functions for all maths, each independently unit-testable.

