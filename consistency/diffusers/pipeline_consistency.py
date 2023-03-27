from typing import List, Optional, Tuple, Union

import torch
from diffusers import DiffusionPipeline, ImagePipelineOutput, UNet2DModel
from diffusers.utils import randn_tensor

from .scheduling_consistency import ConsistencyScheduler


class ConsistencyPipeline(DiffusionPipeline):
    unet: UNet2DModel
    scheduler: ConsistencyScheduler

    def __init__(
        self,
        unet: UNet2DModel,
        scheduler: ConsistencyScheduler,
    ) -> None:
        super().__init__()
        self.register_modules(unet=unet, scheduler=scheduler)

    @torch.no_grad()
    def __call__(
        self,
        batch_size: int = 1,
        num_inference_steps: int = 1,
        generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None,
        output_type: Optional[str] = "pil",
        return_dict: bool = True,
        **kwargs,
    ) -> Union[Tuple, ImagePipelineOutput]:
        r"""
        Args:
            batch_size (`int`, *optional*, defaults to 1):
                The number of images to generate.
            generator (`torch.Generator`, *optional*):
                One or a list of [torch generator(s)](https://pytorch.org/docs/stable/generated/torch.Generator.html)
                to make generation deterministic.
            num_inference_steps (`int`, *optional*, defaults to 50):
                The number of denoising steps. More denoising steps usually lead to a higher quality image at the
                expense of slower inference.
            output_type (`str`, *optional*, defaults to `"pil"`):
                The output format of the generate image. Choose between
                [PIL](https://pillow.readthedocs.io/en/stable/): `PIL.Image.Image` or `np.array`.
            return_dict (`bool`, *optional*, defaults to `True`):
                Whether or not to return a [`~pipelines.ImagePipelineOutput`] instead of a plain tuple.
        Returns:
            [`~pipelines.ImagePipelineOutput`] or `tuple`: [`~pipelines.utils.ImagePipelineOutput`] if `return_dict` is
            True, otherwise a `tuple. When returning a tuple, the first element is a list with the generated images.
        """

        img_size = self.unet.config.sample_size
        shape = (batch_size, 3, img_size, img_size)

        model = self.unet

        sample = randn_tensor(shape, generator=generator)

        self.scheduler.set_timesteps(num_inference_steps)

        time: float = self.scheduler.config.time_max

        for step in self.progress_bar(range(num_inference_steps)):
            if step > 0:
                time = self.scheduler.search_for_previous_time(time)

            sample = self.scheduler.add_noise_to_input(
                sample, time, generator=generator
            )
            model_output = model(
                sample, torch.tensor([time], device=sample.device)
            ).sample

            sample = self.scheduler.step(model_output, time, sample).prev_sample

        sample = (sample / 2 + 0.5).clamp(0, 1)
        image = sample.cpu().permute(0, 2, 3, 1).numpy()
        if output_type == "pil":
            image = self.numpy_to_pil(image)

        if not return_dict:
            return (image,)

        return ImagePipelineOutput(images=image)
