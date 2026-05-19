"""
Teacher model loading for EveryDream2trainer cross-space FM distillation.
"""
from __future__ import annotations

import logging
from typing import Optional

import torch
from diffusers import AutoPipelineForText2Image, FlowMatchEulerDiscreteScheduler

from core.flow_match_model import TrainFlowMatchEulerDiscreteScheduler
from model.training_model import TrainingModel, get_training_noise_scheduler


def load_teacher_model(
    args,
    device,
    student_model: TrainingModel,
) -> Optional[TrainingModel]:
    """
    Load a teacher pipeline and wrap it as a :class:`TrainingModel`.

    Returns ``None`` when ``--teacher`` is not set or ``--teacher_p <= 0``.

    Covers:
    * BUG 1 — calls ``set_noise_scheduler_shift`` so that ``self.timesteps``
      and ``self.sigmas`` are populated before any bridge call.
    * BUG 2 — uses ``AutoPipelineForText2Image`` to support SD1/SD2/SDXL
      teachers; respects ``--teacher_prediction_type`` override.
    * IMPL 4 — extracted from ``train.py`` inline block.
    """
    if getattr(args, 'teacher', None) is None or getattr(args, 'teacher_p', 0) <= 0:
        return None

    logging.info(
        f"* Loading teacher model @ p={args.teacher_p} from {args.teacher}"
    )

    # ── Load pipeline (auto-detects SD1/SD2/SDXL) ───────────────────────────
    try:
        teacher_pipeline = AutoPipelineForText2Image.from_pretrained(
            args.teacher, torch_dtype=torch.float16
        ).to(device)
    except Exception as e:
        logging.warning(
            f"AutoPipelineForText2Image failed ({e}); falling back to "
            "StableDiffusionPipeline."
        )
        from diffusers import StableDiffusionPipeline
        teacher_pipeline = StableDiffusionPipeline.from_pretrained(
            args.teacher, torch_dtype=torch.float16
        ).to(device)

    # ── Scheduler: force FM when requested or already FM ────────────────────
    teacher_prediction_type = getattr(args, 'teacher_prediction_type', 'auto')

    # Resolve the teacher's own shift.  Priority:
    #  1. Explicit --teacher_flow_match_shift arg
    #  2. The shift already baked into the teacher's saved scheduler config
    #  3. Fall back to the student's --flow_match_shift
    teacher_shift: float = float(
        getattr(args, 'teacher_flow_match_shift', None)
        or teacher_pipeline.scheduler.config.get('shift', None)
        or getattr(args, 'flow_match_shift', 1.0)
    )

    if (
        teacher_prediction_type == 'flow_prediction'
        or isinstance(teacher_pipeline.scheduler, FlowMatchEulerDiscreteScheduler)
    ):
        teacher_pipeline.scheduler = TrainFlowMatchEulerDiscreteScheduler.from_config(
            teacher_pipeline.scheduler.config,
            use_dynamic_shifting=False,   # teacher uses fixed shift, not dynamic
            time_shift_type='linear',
            shift=teacher_shift,          # teacher's OWN shift
        )
        teacher_pipeline.scheduler.config.prediction_type = 'flow_prediction'

    if type(teacher_pipeline.scheduler) != type(student_model.noise_scheduler):
        logging.warning(
            f" * Teacher and student schedulers differ — "
            f"teacher={type(teacher_pipeline.scheduler).__name__}, "
            f"student={type(student_model.noise_scheduler).__name__}"
        )
    if (
        teacher_pipeline.scheduler.config.get('prediction_type')
        != student_model.noise_scheduler.config.get('prediction_type')
    ):
        logging.warning(
            f" * Teacher and student use different prediction types — "
            f"teacher={teacher_pipeline.scheduler.config.get('prediction_type')} ({type(student_model.noise_scheduler)}), "
            f"student={student_model.noise_scheduler.config.get('prediction_type')} ({type(student_model.noise_scheduler)})"
        )

    teacher_unet = teacher_pipeline.unet
    teacher_unet.eval()

    # ── Text encoder: share with student when identical ──────────────────────
    teacher_te_sd = teacher_pipeline.text_encoder.state_dict()
    base_te_sd = student_model.text_encoder.state_dict()
    delta = 0.0
    epsilon = 1e-2
    with torch.no_grad():
        for k in teacher_te_sd.keys():
            if k == 'text_model.embeddings.token_embedding.weight':
                if k in base_te_sd and base_te_sd[k].shape != teacher_te_sd[k].shape:
                    min_len = min(base_te_sd[k].shape[0], teacher_te_sd[k].shape[0])
                    teacher_te_sd[k] = teacher_te_sd[k][:min_len]
                    base_te_sd[k] = base_te_sd[k][:min_len]
            if k not in base_te_sd or base_te_sd[k].shape != teacher_te_sd[k].shape:
                delta = 1.0 + epsilon
                break
            delta += torch.mean(
                (base_te_sd[k].cpu().float() - teacher_te_sd[k].cpu().float()) ** 2
            ).item()

    if delta > epsilon:
        logging.info(
            "* Teacher text encoder differs from student → using teacher text encoder"
        )
        teacher_text_encoder = teacher_pipeline.text_encoder
        teacher_text_encoder.eval()
        teacher_text_encoder_2 = getattr(teacher_pipeline, 'text_encoder_2', None)
        if teacher_text_encoder_2 is not None:
            teacher_text_encoder_2.eval()
    else:
        logging.info(
            "* Teacher text encoder identical to student → sharing student text encoder"
        )
        teacher_text_encoder = None
        teacher_text_encoder_2 = None

    del teacher_te_sd, base_te_sd

    teacher_sampler = 'flow-matching' if isinstance(teacher_pipeline.scheduler, TrainFlowMatchEulerDiscreteScheduler) else 'ddpm'
    teacher_scheduler = get_training_noise_scheduler(
        teacher_pipeline.scheduler, train_sampler=teacher_sampler
    )
    teacher_model = TrainingModel(
        noise_scheduler=teacher_scheduler,
        unet=teacher_unet,
        text_encoder=teacher_text_encoder,
        text_encoder_2=teacher_text_encoder_2,
        tokenizer=teacher_pipeline.tokenizer,
        tokenizer_2=getattr(teacher_pipeline, 'tokenizer_2', None),
        vae=None,
        compel=None,
        yaml=None,
    )
    del teacher_pipeline

    # ── BUG 1: initialise scheduler timestep table ───────────────────────────
    # `get_sigmas_for_timesteps` (used by the bridge) needs self.sigmas and
    # self.timesteps, which are populated by set_timesteps() / set_noise_scheduler_shift().
    # Use the teacher's OWN shift (resolved above), not the student's.
    if isinstance(teacher_model.noise_scheduler, TrainFlowMatchEulerDiscreteScheduler):
        teacher_model.set_noise_scheduler_shift(teacher_shift)

    return teacher_model
