import os
from typing import Callable, Optional, Union
from typing_extensions import Self

import torch
from diffusers import SchedulerMixin, ConfigMixin, FlowMatchEulerDiscreteScheduler
from diffusers.configuration_utils import register_to_config


class TrainFlowMatchScheduler(FlowMatchEulerDiscreteScheduler):

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
        print(scheduler_ref.shift)
        (scheduler, unused_kwargs) = cls.from_config(scheduler_ref.config,
                                                     return_unused_kwargs=True,
                                                     **kwargs)
        scheduler.config.prediction_type = 'flow_prediction'
        if return_unused_kwargs:
            return (scheduler, unused_kwargs)
        else:
            return scheduler

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        print('__init__ shift saved:', super().shift)
        self.config.prediction_type = "flow_prediction"

    def get_exact_timesteps(self, timestep_indices):
        """ For incoming timestep indices (from 0 to num_train_timesteps-1), get the exact timesteps incorporating any shift """
        assert timestep_indices.min() >= 0 and timestep_indices.max() < self.config.num_train_timesteps, \
            f"Timestep indices should be in [0, {self.config.num_train_timesteps - 1}] but got {timestep_indices.min()} to {timestep_indices.max()}"

        # timestep indices goes from 0 to 999 but self.timesteps goes from 1000 to 1
        # so we need to reverse the indices
        indices_reversed = self.config.num_train_timesteps - 1 - timestep_indices.cpu()
        return self.timesteps[indices_reversed]

    @property
    def shift(self):
        return super().shift

    def add_noise(self, latents: torch.Tensor, noise: torch.Tensor, timesteps: torch.Tensor) -> torch.Tensor:
        return self.scale_noise(latents, timesteps, noise)
        #alpha = (timesteps / self.config.num_train_timesteps).view(-1, 1, 1, 1)
        #x_t = alpha*noise + (1-alpha)*latents
        #return x_t


