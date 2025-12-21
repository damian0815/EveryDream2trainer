import torch
from diffusers import SchedulerMixin, ConfigMixin

from loss import _get_noisy_latents
from model.training_model import TrainingModel, Conditioning


def get_teacher_target(
    teacher_model: TrainingModel,
    teacher_conditioning: Conditioning,
    student_model: TrainingModel,
    student_timesteps: torch.Tensor,
    clean_image_latents: torch.Tensor,
    noise: torch.Tensor
):

    teacher_prediction_type = teacher_model.noise_scheduler.config.prediction_type
    student_prediction_type = student_model.noise_scheduler.config.prediction_type
    if teacher_prediction_type in ['v_prediction', 'v-prediction'] and student_prediction_type == 'flow_prediction':
        teacher_timesteps, teacher_noisy_latents = _remap_noise_v_pred_to_flow_matching(
            teacher_model, student_model, student_timesteps, clean_image_latents, noise
        )
    else:
        supported_prediction_types = {'epsilon', 'v_prediction', 'v-prediction'}
        present_prediction_types = {teacher_prediction_type, student_prediction_type}
        if supported_prediction_types.isdisjoint(present_prediction_types):
            raise ValueError(f"Unsupported prediction type conversion: {teacher_prediction_type} to {student_prediction_type}")
        teacher_timesteps = student_timesteps
        teacher_noisy_latents = _get_noisy_latents(
            clean_image_latents, noise, teacher_model.noise_scheduler, student_timesteps, latents_perturbation=0
        )

    teacher_model_output = teacher_model.unet(
        teacher_noisy_latents.half(), teacher_timesteps, teacher_conditioning.prompt_embeds.half()
    ).sample.float()

    if teacher_prediction_type == student_prediction_type:
        return teacher_model_output

    return _convert_model_output(
        noise=noise,
        teacher_input=teacher_noisy_latents,
        teacher_output=teacher_model_output,
        teacher_scheduler=teacher_model.noise_scheduler,
        teacher_timesteps=teacher_timesteps,
        student_prediction_type=student_prediction_type,
        student_timesteps=student_timesteps,
    )


def _convert_model_output(
    noise: torch.Tensor,
    teacher_input: torch.Tensor,
    teacher_output: torch.Tensor,
    teacher_scheduler: SchedulerMixin | ConfigMixin,
    teacher_timesteps,
    student_prediction_type,
    student_timesteps,
):
    source_prediction_type = teacher_scheduler.config.prediction_type
    if source_prediction_type in ['v_prediction', 'v-prediction']:
        alpha_t = teacher_scheduler.alphas_cumprod[teacher_timesteps].view(-1, 1, 1, 1).sqrt()
        sigma_t = (1 - alpha_t ** 2).sqrt().to(teacher_output.device)
        # Solve for x_0 from v-prediction
        x_0_pred = (alpha_t * teacher_input - sigma_t * teacher_output) / (alpha_t ** 2 + sigma_t ** 2).sqrt() # note that this denominator == 1 typically

        if student_prediction_type == 'epsilon':
            assert student_timesteps == teacher_timesteps
            # Epsilon: ε = (x_t - α_t·x_0) / σ_t
            return (teacher_input - alpha_t * x_0_pred) / sigma_t
        elif student_prediction_type == 'flow_prediction':
            return noise - x_0_pred

    elif source_prediction_type == 'epsilon' and student_prediction_type in ['v_prediction', 'v-prediction']:
        assert student_timesteps == teacher_timesteps
        # Convert epsilon to v-prediction
        alpha_t = teacher_scheduler.alphas_cumprod[teacher_timesteps].view(-1, 1, 1, 1).sqrt()
        sigma_t = (1 - alpha_t ** 2).sqrt().to(teacher_output.device)
        # First get x_0 from epsilon: x_0 = (x_t - σ_t·ε) / α_t
        x_0_pred = (teacher_input - sigma_t * teacher_output) / alpha_t
        # Then compute v: v = α_t·ε - σ_t·x_0
        return alpha_t * teacher_output - sigma_t * x_0_pred
    else:
        raise ValueError(
            f"Cannot convert between teacher model prediction type {source_prediction_type} and training model prediction type {student_prediction_type}")




def _remap_noise_v_pred_to_flow_matching(
    teacher_model, # v-pred
    student_model, # flow matching
    student_timesteps: torch.Tensor, # integer
    latents: torch.Tensor,
    noise: torch.Tensor,
):
    if teacher_model.noise_scheduler.config.prediction_type not in ['v_prediction', 'v-prediction']:
        raise ValueError("Teacher model must use v-prediction for SNR-based timestep remapping.")
    if student_model.noise_scheduler.config.prediction_type != 'flow_prediction':
        raise ValueError("Student model must use flow-matching for SNR-based timestep remapping.")

    student_timesteps = student_timesteps.to(student_model.device)
    snr = _get_snr_schedule(student_model.noise_scheduler).to(student_model.device)[student_timesteps]
    teacher_timesteps = _find_best_timestep_for_snr(snr, _get_snr_schedule(teacher_model.noise_scheduler)).to(teacher_model.device)

    teacher_noisy_latents = _get_noisy_latents(latents, noise, teacher_model.noise_scheduler, teacher_timesteps, latents_perturbation=0.0)

    return teacher_timesteps, teacher_noisy_latents



def _get_snr_schedule(noise_scheduler) -> torch.Tensor:
    if hasattr(noise_scheduler, 'alphas_cumprod'):
        alphas_cumprod = noise_scheduler.alphas_cumprod
        return alphas_cumprod / (1 - alphas_cumprod)
    else:
        sigmas = noise_scheduler.sigmas
        assert len(sigmas) == noise_scheduler.config.num_train_timesteps
        # Avoid division by zero at sigma=0
        # Rectified Flow: signal_scale = (1 - t), noise_scale = t
        # SNR = (1-t)^2 / t^2
        return (1 - sigmas) ** 2 / (sigmas**2 + 1e-8)


def _find_best_timestep_for_snr(target_snr: torch.Tensor, snr_schedule: torch.Tensor) -> torch.Tensor:
    # Flip for searchsorted (assuming strictly decreasing SNR in teacher)
    snr_flipped = torch.flip(snr_schedule, dims=[0])
    idx_flipped = torch.searchsorted(snr_flipped, target_snr.view(-1))

    # Convert back to indices
    max_timestep_idx = snr_schedule.shape[0] - 1
    timesteps = max_timestep_idx - idx_flipped
    timesteps = torch.clamp(timesteps, 0, max_timestep_idx)
    return timesteps
