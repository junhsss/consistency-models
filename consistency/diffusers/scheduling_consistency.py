import math
from dataclasses import dataclass
from typing import Optional, Tuple, Union

import torch
from diffusers.configuration_utils import ConfigMixin, register_to_config
from diffusers.schedulers.scheduling_utils import SchedulerMixin
from diffusers.utils import BaseOutput, randn_tensor


@dataclass
class ConsistencySchedulerOutput(BaseOutput):
    """
    Output class for the scheduler's step function output.
    Args:
        prev_sample (`torch.FloatTensor` of shape `(batch_size, num_channels, height, width)` for images):
            Computed sample (x_{t-1}) of previous timestep. `prev_sample` should be used as next model input in the
            denoising loop.
        prev_time (`float`): Searched
        pred_original_sample (`torch.FloatTensor` of shape `(batch_size, num_channels, height, width)` for images):
            The predicted denoised sample (x_{0}) based on the model output from the current timestep.
            `pred_original_sample` can be used to preview progress or for guidance.
    """

    prev_sample: torch.FloatTensor


class ConsistencyScheduler(SchedulerMixin, ConfigMixin):
    @register_to_config
    def __init__(
        self,
        time_min: float = 0.002,
        time_max: float = 80,
        data_std: float = 0.5,
    ):
        # standard deviation of the initial noise distribution
        self.init_noise_sigma = time_max

        # setable values
        self.num_inference_steps: int = None
        self.timesteps: torch.LongTensor = None
        self.schedule: torch.FloatTensor = None  # sigma(t_i)

    def scale_model_input(
        self, sample: torch.FloatTensor, timestep: Optional[int] = None
    ) -> torch.FloatTensor:
        """
        Ensures interchangeability with schedulers that need to scale the denoising model input depending on the
        current timestep.
        Args:
            sample (`torch.FloatTensor`): input sample
            timestep (`int`, optional): current timestep
        Returns:
            `torch.FloatTensor`: scaled input sample
        """
        return sample

    def add_noise_to_input(
        self,
        sample: torch.FloatTensor,
        time: float,
        generator: Optional[torch.Generator] = None,
    ) -> Tuple[torch.FloatTensor, float]:
        if self.config.time_min <= time <= self.config.time_max:
            sigma = math.sqrt(time**2 - self.config.time_min**2)
        else:
            sigma = 0

        eps = sigma * randn_tensor(
            sample.shape, device=sample.device, generator=generator
        )

        sample_hat = sample + eps

        return sample_hat

    def search_previous_time(self, time):
        time = (time + self.config.time_min) / 2
        return time

    def step(
        self,
        model_output: torch.FloatTensor,
        time: float,
        sample: torch.FloatTensor,
        return_dict: bool = True,
    ) -> Union[ConsistencySchedulerOutput, Tuple]:
        skip_coef = self.config.data_std**2 / (
            (time - self.config.time_min) ** 2 + self.config.data_std**2
        )
        out_coef = (
            self.config.data_std
            * time
            / (time**2 + self.config.data_std**2) ** (0.5)
        )

        output = sample * skip_coef + model_output * out_coef

        self.time = time

        return ConsistencySchedulerOutput(prev_sample=output.clamp(-1.0, 1.0))

    @staticmethod
    def image_time_product(images: torch.Tensor, times: torch.Tensor):
        return torch.einsum("b c h w, b -> b c h w", images, times)
