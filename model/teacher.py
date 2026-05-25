"""
Teacher model loading for EveryDream2trainer cross-space FM distillation.
"""
from __future__ import annotations

import logging
from typing import Optional

import torch
from diffusers import AutoPipelineForText2Image, FlowMatchEulerDiscreteScheduler, StableDiffusionXLPipeline

from core.flow_match_model import TrainFlowMatchEulerDiscreteScheduler
from model.training_model import TrainingModel, get_training_noise_scheduler


def load_teacher_model(
    args,
    device,
    student_model: TrainingModel,
    flow_match_shift_dynamic: bool = False,
    flow_match_shift: int = 1,
) -> Optional[TrainingModel]:
    """
    Load a teacher pipeline and wrap it as a :class:`TrainingModel`.
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

    teacher_is_sdxl = isinstance(teacher_pipeline, StableDiffusionXLPipeline)

    # ── Scheduler: force FM when requested or already FM ────────────────────
    teacher_prediction_type = getattr(args, 'teacher_prediction_type', 'auto')

    teacher_sampler = 'flow-matching' if isinstance(teacher_pipeline.scheduler, FlowMatchEulerDiscreteScheduler) else 'ddpm'
    teacher_scheduler = get_training_noise_scheduler(
        teacher_pipeline.scheduler, train_sampler=teacher_sampler, flow_match_shift=flow_match_shift, flow_match_shift_dynamic=flow_match_shift_dynamic
    )
    if type(teacher_scheduler) != type(student_model.noise_scheduler):
        logging.warning(
            f" * Teacher and student schedulers differ — "
            f"teacher={type(teacher_scheduler).__name__}, "
            f"student={type(student_model.noise_scheduler).__name__}"
        )
    if (
        teacher_scheduler.config.get('prediction_type')
        != student_model.noise_scheduler.config.get('prediction_type')
    ):
        logging.warning(
            f" * Teacher and student use different prediction types — "
            f"teacher={teacher_scheduler.config.get('prediction_type')} ({type(student_model.noise_scheduler)}), "
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

    if teacher_is_sdxl != student_model.is_sdxl:
        teacher_vae = teacher_pipeline.vae
    else:
        teacher_vae = None

    teacher_model = TrainingModel(
        noise_scheduler=teacher_scheduler,
        unet=teacher_unet,
        text_encoder=teacher_text_encoder,
        text_encoder_2=teacher_text_encoder_2,
        tokenizer=teacher_pipeline.tokenizer,
        tokenizer_2=getattr(teacher_pipeline, 'tokenizer_2', None),
        vae=teacher_vae,
        compel=None,
        yaml=None,
    )
    del teacher_pipeline
    if isinstance(teacher_scheduler, TrainFlowMatchEulerDiscreteScheduler):
        teacher_model.set_noise_scheduler_shift(flow_match_shift)

    return teacher_model
