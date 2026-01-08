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

    @staticmethod
    def get_shifted_timesteps(timestep_indices, timestep_values):
        """ For incoming timestep indices (from 0 to num_train_timesteps-1), get the exact timesteps incorporating any shift """
        assert timestep_indices.min() >= 0 and timestep_indices.max() < len(timestep_values), \
            f"Timestep indices should be >=0, <{len(timestep_values)} but got min {timestep_indices.min()} max {timestep_indices.max()}"

        # timestep indices goes from 0 to 999 but self.timesteps goes from 1000 to 1
        # so we need to reverse the indices
        indices_reversed = len(timestep_values) - 1 - timestep_indices.cpu()
        return timestep_values[indices_reversed]

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


