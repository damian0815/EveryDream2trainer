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
        (scheduler, unused_kwargs) = cls.from_config(scheduler_ref.config,
                                                     return_unused_kwargs=True,
                                                     **kwargs)
        scheduler.config.prediction_type = 'flow-matching'
        if return_unused_kwargs:
            return (scheduler, unused_kwargs)
        else:
            return scheduler

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.config.prediction_type = "flow-matching"



    def add_noise(self, latents: torch.Tensor, noise: torch.Tensor, timesteps: torch.Tensor) -> torch.Tensor:
        alpha = (timesteps / self.config.num_train_timesteps).view(-1, 1, 1, 1)
        x_t = alpha*noise + (1-alpha)*latents
        return x_t


