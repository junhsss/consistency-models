import math
from typing import List, Optional, Tuple, Union

import torch
from diffusers import DiffusionPipeline, ImagePipelineOutput, UNet2DModel
from diffusers.utils import randn_tensor


class ConsistencyPipeline(DiffusionPipeline):
    unet: UNet2DModel

    def __init__(
        self,
        unet: UNet2DModel,
    ) -> None:
        super().__init__()
        self.register_modules(unet=unet)

    @torch.no_grad()
    def __call__(
        self,
        steps: int = 1,
        generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None,
        time_min: float = 0.002,
        time_max: float = 80.0,
        data_std: float = 0.5,
        output_type: Optional[str] = "pil",
        return_dict: bool = True,
        **kwargs,
    ) -> Union[Tuple, ImagePipelineOutput]:
        img_size = self.unet.config.sample_size
        shape = (1, 3, img_size, img_size)

        model = self.unet

        time: float = time_max

        sample = randn_tensor(shape, generator=generator, device=self.device) * time

        for step in self.progress_bar(range(steps)):
            if step > 0:
                time = self.search_previous_time(time)
                sigma = math.sqrt(time**2 - time_min**2 + 1e-6)
                sample = sample + sigma * randn_tensor(sample.shape, device=self.device, generator=generator)

            out = model(sample, torch.tensor([time], device=self.device)).sample

            skip_coef = data_std**2 / ((time - time_min) ** 2 + data_std**2)
            out_coef = data_std * time / (time**2 + data_std**2) ** (0.5)

            sample = (sample * skip_coef + out * out_coef).clamp(-1.0, 1.0)

        sample = (sample / 2 + 0.5).clamp(0, 1)
        image = sample.cpu().permute(0, 2, 3, 1).numpy()

        if output_type == "pil":
            image = self.numpy_to_pil(image)

        if not return_dict:
            return (image,)

        return ImagePipelineOutput(images=image)

    def search_previous_time(self, time, time_min: float = 0.002, time_max: float = 80.0):
        return (2 * time + time_min) / 3

    def cuda(self):
        self.to("cuda")
