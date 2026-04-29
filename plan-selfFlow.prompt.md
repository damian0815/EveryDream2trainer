# Plan: Self-Flow Representation Learning for Flow-Matching U-Net

Self-Flow adds a self-distillation loop on top of the existing Flow Matching training: the student U-Net (forward-passed with **heterogeneously noised** latents) is taught to mimic the intermediate features of an EMA-copy teacher U-Net (forward-passed with a **uniformly cleaner** version). The student's early features (`down_blocks[1]`, ~30% depth) are projected to match the teacher's late features (`up_blocks[0]`, ~70% depth), using cosine similarity loss. Since both extraction points are at the same spatial resolution (H/4 × W/4), no spatial interpolation is needed — only a 1×1 channel projection (640 → 1280).

**Shape confirmation (512px input, 64×64 latent):**
- `down_blocks[1]` output → `[B, 640, 16, 16]` (after its internal downsampler)
- `up_blocks[0]` output → `[B, 1280, 16, 16]` (after its internal upsampler)
- ✅ Spatial dims match. Projection head = one `Conv2d(640, 1280, kernel_size=1)`.

### Steps

1. **Create [`core/self_flow.py`](core/self_flow.py)** with `SelfFlowProjectionHead` (a single `Conv2d(640, 1280, 1)`) and a `compute_self_flow_loss()` function: builds heterogeneous noisy latents (binary spatial mask M at 25% ratio, two timesteps `t` and `s`, pixel-wise mix → `x_τ`; teacher gets uniform `τ_min = min(t, s)`), then computes `L_rep = -cosine_similarity(proj(H_student), F_teacher, dim=1).mean()`.

2. **Add Self-Flow EMA teacher in [`train.py`](train.py)** alongside the existing `unet_ema` block (~line 706): when `args.self_flow_p > 0`, `deepcopy` the student UNet into `self_flow_teacher_unet` (frozen, no grad); update it each step with the existing `update_ema()` at decay `args.self_flow_ema_decay` (default 0.9999). Instantiate `SelfFlowProjectionHead` here, add its parameters to the optimizer.

3. **Add forward hooks in [`core/loss.py`](core/loss.py) `get_model_prediction_and_target()`** following the existing `lcf_mask` / `midblock_out` hook pattern: when Self-Flow is active, register a hook on `model.unet.down_blocks[1]` to capture `H_student`, and a second hook on `self_flow_teacher_unet.up_blocks[0]` to capture `F_teacher`. Extend `ModelPredictionAndTargetReturnType` with `self_flow_student_features` and `self_flow_teacher_features`. Pass the base scalar timestep `t` (the primary timestep, not `s`) to the student U-Net's time embedding, while applying the full spatial τ mix to the student's input latents. The teacher receives `τ_min = min(t, s)` for both its latents and its time embedding.

4. **Add `self_flow_*` args** to argument parsing in [`train.py`](train.py) and add defaults to [`train_flow_matching.json`](train_flow_matching.json): `self_flow_p` (float, 0 = off), `self_flow_gamma` (default 0.8), `self_flow_mask_ratio` (default 0.25), `self_flow_ema_decay` (default 0.9999), `self_flow_ema_update_interval` (default matches `ema_update_interval`).

5. **Integrate `L_rep` in `_do_loss()` in [`core/step.py`](core/step.py)**: when `self_flow_student_features` is not None, call `compute_self_flow_loss()` from `core/self_flow.py` and add `args.self_flow_gamma * L_rep` to the per-sample loss before scaling/accumulation.

6. **Ensure projection head is not saved with the U-Net** in [`model/training_model.py`](model/training_model.py) `save_model()`: the proj head is a standalone module; either skip it on save or optionally save it to a sidecar `.safetensors` file in the log dir.

### Further Considerations

1. **Interaction with existing `teacher_model` and `unet_ema`**: the Self-Flow EMA teacher is a *separate* `deepcopy` from the existing `unet_ema` (used for EMA sampling/saving). They can share the same `update_ema()` call pattern but must remain independent copies to avoid coupling — confirm at implementation time whether a single EMA copy can serve both purposes or if two copies are needed.
2. **Gradient-checkpointing + hooks**: the existing `gradient_checkpointing` flag in the U-Net may interfere with forward hooks on `down_blocks[1]` because checkpointing re-runs the forward pass. Verify hook behaviour with gradient checkpointing enabled, and consider disabling it for self-flow passes only, or hooking the module after checkpointing boundaries.
3. **Memory cost**: two additional U-Net forward passes per step (student + teacher) plus the EMA teacher copy on GPU is expensive. A `self_flow_p < 1.0` probability gate (already planned) lets users subsample Self-Flow steps to manage VRAM/throughput cost. Consider documenting recommended starting values (e.g., `self_flow_p: 0.5`).

