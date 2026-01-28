import logging
import os
from typing import Callable, Optional, Union, Dict, Any, Tuple
from typing_extensions import Self

import torch
from diffusers import SchedulerMixin, ConfigMixin, FlowMatchEulerDiscreteScheduler
from diffusers.configuration_utils import register_to_config, FrozenDict


class TrainFlowMatchEulerDiscreteScheduler(FlowMatchEulerDiscreteScheduler):

    #@classmethod
    #def from_pretrained(
    #    cls,
    #    pretrained_model_name_or_path: Optional[Union[str, os.PathLike]] = None,
    #    subfolder: Optional[str] = None,
    #    return_unused_kwargs=False,
    #    **kwargs,
    #) -> Self:
    #    scheduler_ref = FlowMatchEulerDiscreteScheduler.from_pretrained(#pretrained_model_name_or_path=pretrained_model_name_or_path,
    #                                                                subfolder=subfolder,
    #                                                                return_unused_kwargs=False,
    #                                                                **kwargs)
    #    print("TrainFlowMatchEulerDiscreteScheduler.from_pretrained: shift=", scheduler_ref.shift)
    #    (scheduler, unused_kwargs) = cls.from_config(scheduler_ref.config,
    #                                                 return_unused_kwargs=True,
    #                                                 **kwargs)
    #    scheduler.config.prediction_type = 'flow_prediction'
    #    if return_unused_kwargs:
    #        return (scheduler, unused_kwargs)
    #    else:
    #        return scheduler

    #@classmethod
    #def from_config(
    #    cls, config: Union[FrozenDict, Dict[str, Any]] = None, return_unused_kwargs=False, **kwargs
    #) -> Union[Self, Tuple[Self, Dict[str, Any]]]:
    #    model, unused_kwargs = ConfigMixin.from_config.__func__(cls, config, return_unused_kwargs=True, **kwargs)
    #    model.config.prediction_type = 'flow_prediction'
    #    print(f'from_config shift saved: {super().shift}, dynamic: {model.config.use_dynamic_shifting}, #type: {model.config.time_shift_type}')
    #    if return_unused_kwargs:
    #        return model, unused_kwargs
    #    else:
    #        return model


    #def __init__(self, **kwargs):
    #    super().__init__(**kwargs)
    #    print(f'__init__ shift saved: {super().shift}, dynamic: {self.config.use_dynamic_shifting}, type: {self.config.time_shift_type}')
    #   self.config.prediction_type = "flow_prediction"


    @staticmethod
    def get_shifted_timesteps(timestep_indices, timestep_values):
        """ For incoming timestep indices (from 0 to num_train_timesteps-1), get the exact timesteps incorporating any shift """
        assert timestep_indices.min() >= 0 and timestep_indices.max() < len(timestep_values), \
            f"Timestep indices should be >=0, <{len(timestep_values)} but got min {timestep_indices.min()} max {timestep_indices.max()}"

        # timestep indices goes from 0 to 999 but self.timesteps goes from 1000 to 1
        # so we need to reverse the indices
        indices_reversed = len(timestep_values) - 1 - timestep_indices.cpu()
        return timestep_values[indices_reversed]


    def get_sigmas_for_timesteps(self, timesteps: torch.Tensor):
        """ For the given timesteps, get the corresponding sigmas """
        indices = (timesteps[:, None] == self.timesteps[None, :]).nonzero(as_tuple=True)[1]
        return self.sigmas[indices]


    def get_best_timestep_for_sigma(self, sigma: float):
        delta = (self.sigmas - sigma).abs()
        index = torch.argmin(delta).item()
        return self.timesteps[index]


    @property
    def shift(self):
        return super().shift

    def add_noise(self, latents: torch.Tensor, noise: torch.Tensor, timesteps: torch.Tensor) -> torch.Tensor:
        return self.scale_noise(latents, timesteps, noise)


class SDPipelineInferenceFlowMatchEulerDiscreteScheduler(FlowMatchEulerDiscreteScheduler):

    @property
    def init_noise_sigma(self):
        return 1

    def scale_model_input(self, x, t):
        return x

    def add_noise(self, latents, noise, timesteps):
        return self.scale_noise(latents, timesteps, noise)

    def set_timesteps(
        self,
        num_inference_steps: Optional[int]=None,
        device: Union[str, torch.device] = None,
        **kwargs
    ):
        if self.config.use_dynamic_shifting and 'mu' not in kwargs:
            kwargs['mu'] = self.shift
        super().set_timesteps(num_inference_steps=num_inference_steps, device=device, **kwargs)
        print(f'timesteps ({num_inference_steps}):', self.timesteps)
