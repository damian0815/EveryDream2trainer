import os
from typing import Callable, Optional, Union
from typing_extensions import Self

import torch
from diffusers import SchedulerMixin, ConfigMixin, FlowMatchEulerDiscreteScheduler
from diffusers.configuration_utils import register_to_config




class TrainFlowMatchEulerDiscreteScheduler(FlowMatchEulerDiscreteScheduler):

    @classmethod
    def from_pretrained(
        cls,
        pretrained_model_name_or_path: Optional[Union[str, os.PathLike]] = None,
        subfolder: Optional[str] = None,
        return_unused_kwargs=False,
        **kwargs,
    ) -> Self:
        scheduler_ref = FlowMatchEulerDiscreteScheduler.from_pretrained(pretrained_model_name_or_path=pretrained_model_name_or_path,
                                                                    subfolder=subfolder,
                                                                    return_unused_kwargs=False,
                                                                    **kwargs)
        print("TrainFlowMatchEulerDiscreteScheduler.from_pretrained: shift=", scheduler_ref.shift)
        (scheduler, unused_kwargs) = cls.from_config(scheduler_ref.config,
                                                     return_unused_kwargs=True,
                                                     **kwargs)
        scheduler.config.prediction_type = 'flow_prediction'
        if return_unused_kwargs:
            return (scheduler, unused_kwargs)
        else:
            return scheduler

    def __init__(self, **kwargs):
        # 'exponential' time shift matches SD3
        super().__init__(**kwargs, shift=1, use_dynamic_shifting=True, time_shift_type='exponential')
        print('__init__ shift saved:', super().shift)
        self.config.prediction_type = "flow_prediction"

    @staticmethod
    def get_shifted_timesteps(timestep_indices, timestep_values):
        """ For incoming timestep indices (from 0 to num_train_timesteps-1), get the exact timesteps incorporating any shift """
        assert timestep_indices.min() >= 0 and timestep_indices.max() < len(timestep_values), \
            f"Timestep indices should be >=0, <{len(timestep_values)} but got min {timestep_indices.min()} max {timestep_indices.max()}"

        # timestep indices goes from 0 to 999 but self.timesteps goes from 1000 to 1
        # so we need to reverse the indices
        indices_reversed = len(timestep_values) - 1 - timestep_indices.cpu()
        return timestep_values[indices_reversed]

    @staticmethod
    def get_shifted_sigmas(timestep_indices, timestep_values):
        """ For incoming timestep indices (from 0 to num_train_timesteps-1), get the sigmas incorporating any shift """
        assert timestep_indices.min() >= 0 and timestep_indices.max() < len(timestep_values), \
            f"Timestep indices should be >=0, <{len(timestep_values)} but got min {timestep_indices.min()} max {timestep_indices.max()}"

        indices_reversed = len(timestep_values) - 1 - timestep_indices.cpu()
        return timestep_values[indices_reversed] / len(timestep_values)

    def get_best_timestep_for_sigma(self, sigma: float):
        delta = (self.sigmas - sigma).abs()
        index_reversed = torch.argmin(delta).item()
        return self.config.num_train_timesteps - 1 - index_reversed

    def get_timestep_indices(self, exact_timesteps: torch.Tensor):
        """ For incoming exact timesteps, get the corresponding timestep indices (from 0 to num_train_timesteps-1) """
        assert exact_timesteps.min() >= self.timesteps.min() and exact_timesteps.max() <= self.timesteps.max(), \
            f"Exact timesteps should be in [{self.timesteps.min()}, {self.timesteps.max()}] but got {exact_timesteps.min()} to {exact_timesteps.max()}"
        # timestep indices goes from 0 to 999 but self.timesteps goes from 1000 to 1
        # so we need to reverse the indices
        exact_timesteps_cpu = exact_timesteps.cpu()
        indices_reversed = torch.nonzero(self.timesteps[:, None] == exact_timesteps_cpu[None, :], as_tuple=False)[:, 0]
        timestep_indices = self.config.num_train_timesteps - 1 - indices_reversed
        return timestep_indices.to(exact_timesteps.device)

    @property
    def shift(self):
        return super().shift

    def add_noise(self, latents: torch.Tensor, noise: torch.Tensor, timesteps: torch.Tensor) -> torch.Tensor:
        timesteps_shifted = type(self).get_shifted_timesteps(timestep_indices=timesteps, timestep_values=self.timesteps)
        #print("shifted: ", timesteps.cpu().tolist(), "->", timesteps_shifted.cpu().tolist())
        return self.scale_noise(latents, timesteps_shifted, noise)
        #alpha = (timesteps / self.config.num_train_timesteps).view(-1, 1, 1, 1)
        #x_t = alpha*noise + (1-alpha)*latents
        #return x_t



class SDPipelineInferenceFlowMatchEulerDiscreteScheduler(FlowMatchEulerDiscreteScheduler):

    def __init__(self, **kwargs):
        kwargs.pop('shift', None)
        kwargs.pop('use_dynamic_shifting', None)
        kwargs.pop('time_shift_type', None)
        time_shift_type = 'exponential' # don't change without matching training
        super().__init__(**kwargs, shift=1, use_dynamic_shifting=True, time_shift_type=time_shift_type)
        self.shift_to_hack_in = 1
        self.timesteps_no_shift = None   # populated in set_timesteps
        self.timesteps_with_shift = None # populated in set_timesteps
        self.set_timesteps(self.config.num_train_timesteps, device='cpu')

    @property
    def init_noise_sigma(self):
        return 1

    def scale_model_input(self, x, t):
        return x

    def set_shift(self, shift):
        # override super behaviour
        self.shift_to_hack_in = shift

    def add_noise(self, latents, noise, timesteps):
        print("timesteps input: ", timesteps.cpu().tolist())
        print("schedule for unet: ", self.timesteps_no_shift)
        indices_reversed = torch.nonzero(self.timesteps_no_shift.cpu() == timesteps.cpu(), as_tuple=False)[:, 0]
        print("indices reversed: ", indices_reversed.tolist())
        timesteps_shifted = self.timesteps[indices_reversed]

        # timesteps_shifted = type(self).get_shifted_timesteps(timestep_indices=timesteps, timestep_values=self.timesteps)
        print("timesteps input: ", timesteps.cpu().tolist(), "-> shifted", timesteps_shifted.cpu().tolist())
        return self.scale_noise(latents, timesteps_shifted, noise)

    def set_timesteps(self, num_inference_steps, device, **kwargs):
        if 'mu' in kwargs:
            del kwargs['mu']
        # construct timesteps with and without shift.
        # during inference, we need to use the shifted timesteps in the step and add_noise functions
        # however the unet forward must be passed the unshifted timesteps.
        super().set_timesteps(num_inference_steps, device, mu=self.shift_to_hack_in, **kwargs)
        self.timesteps_with_shift = self.timesteps.clone()
        super().set_timesteps(num_inference_steps, device, mu=0, **kwargs)
        self.timesteps_no_shift = self.timesteps.clone()

    def step(self,
             model_output: torch.FloatTensor,
             timestep: Union[float, torch.FloatTensor],
             sample: torch.FloatTensor, **kwargs
    ):
        # During inference, apply shift
        print('in overridden step(), timestep=', timestep, 'kwargs', kwargs)
        if not isinstance(timestep, torch.Tensor):
            timestep = torch.tensor(timestep)
        if timestep.ndim == 0:
            timestep = timestep.unsqueeze(0)

        # timesteps should be unshifted
        assert (self.timesteps - self.timesteps_no_shift).abs().sum() == 0, "timesteps should be unshifted at the start of step()"
        if self.config.time_shift_type != 'exponential':
            raise NotImplementedError('todo: mu=0 must be changed to=1 for "linear" shift type')
        timestep_index = torch.nonzero(self.timesteps == timestep, as_tuple=False)[:, 0]
        # shift timesteps
        super().set_timesteps(len(self.timesteps), device=sample.device, mu=self.shift_to_hack_in)
        # adjust input timestep
        shifted_timestep = self.timesteps[timestep_index].squeeze(0)
        print('shifted from', timestep, 'via', timestep_index, 'to', shifted_timestep)

        # do step with shifted timesteps
        result = super().step(model_output, timestep=shifted_timestep, sample=sample, **kwargs)
        print(' -> next step index', self.step_index, "next sigma", self.sigmas[self.step_index])

        # revert timesteps to unshifted
        super().set_timesteps(len(self.timesteps), device=sample.device, mu=0)
        return result
