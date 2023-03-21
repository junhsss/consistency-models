import copy
import math
from contextlib import suppress
from pathlib import Path
from typing import Optional, Type

import torch
from diffusers import UNet2DModel
from diffusers.models.unet_2d import UNet2DOutput
from pytorch_lightning import LightningModule
from pytorch_lightning.loggers.wandb import WandbLogger
from pytorch_lightning.utilities import rank_zero_only
from torch import nn, optim
from torchmetrics import MeanMetric
from torchvision.utils import make_grid, save_image

with suppress(ImportError):
    import wandb


class DiffusersWrapper(nn.Module):
    def __init__(self, unet: UNet2DModel):
        super().__init__()
        self.unet = unet

    def forward(
        self,
        images: torch.Tensor,
        times: torch.Tensor,
    ):
        out: UNet2DOutput = self.unet(images, times)
        return out.sample


class Consistency(LightningModule):
    def __init__(
        self,
        model: nn.Module,
        loss_fn: nn.Module = nn.MSELoss(),
        learning_rate: float = 1e-4,
        image_size: Optional[int] = None,
        channels: Optional[int] = None,
        data_std: float = 0.5,
        time_min: float = 0.002,
        time_max: float = 80.0,
        bins_min: int = 2,
        bins_max: int = 150,
        bins_rho: float = 7,
        initial_ema_decay: float = 0.9,
        optimizer_type: Type[optim.Optimizer] = optim.AdamW,
        samples_path: str = "samples/",
    ) -> None:
        super().__init__()

        if isinstance(model, UNet2DModel):
            if image_size:
                raise TypeError(
                    "'image_size' is not supported for UNet2DModel"
                )
            if channels:
                raise TypeError("'channels' is not supported for UNet2DModel")

            self.model = DiffusersWrapper(model)
            self.image_size = model.sample_size
            self.channels = model.in_channels

        else:
            if not image_size:
                raise TypeError("'image_size' should be provided.")
            if not channels:
                raise TypeError("'channels' should be provided.")

            self.model = model
            self.image_size = image_size
            self.channels = channels

        self.model_ema = copy.deepcopy(model)
        self.model_ema.requires_grad_(False)

        self.loss_fn = loss_fn
        self.optimizer_type = optimizer_type

        self.learning_rate = learning_rate
        self.initial_ema_decay = initial_ema_decay

        self.data_std = data_std
        self.time_min = time_min
        self.time_max = time_max
        self.bins_min = bins_min
        self.bins_max = bins_max
        self.bins_rho = bins_rho

        self._loss_tracker = MeanMetric()
        self._bins_tracker = MeanMetric()
        self._ema_decay_tracker = MeanMetric()

        Path(samples_path).mkdir(exist_ok=True, parents=True)

        self.samples_path = samples_path

    def forward(
        self,
        images: torch.Tensor,
        times: torch.Tensor,
    ):
        skip_coef = self.data_std**2 / (
            (times - self.time_min).pow(2) + self.data_std**2
        )
        out_coef = (
            self.data_std
            * times
            / (times.pow(2) + self.data_std**2).pow(0.5)
        )

        return self.image_time_product(
            images,
            skip_coef,
        ) + self.image_time_product(
            self.model(images, times),
            out_coef,
        )

    def training_step(self, images: torch.Tensor, *args, **kwargs):
        noise = torch.randn(images.shape, device=images.device)
        timesteps = torch.randint(
            0,
            self.bins - 1,
            (images.shape[0],),
            device=images.device,
        ).long()

        current_times = self.timesteps_to_times(timesteps, self.bins)
        next_times = self.timesteps_to_times(timesteps + 1, self.bins)

        current_noise_image = images + self.image_time_product(
            noise,
            current_times,
        )

        next_noise_image = images + self.image_time_product(
            noise,
            next_times,
        )

        with torch.no_grad():
            target = self(current_noise_image, current_times)

        loss = self.loss_fn(self(next_noise_image, next_times), target)

        self._loss_tracker(loss)
        self.log(
            "loss",
            self._loss_tracker,
            on_step=True,
            on_epoch=False,
            logger=True,
        )

        self._bins_tracker(self.bins)
        self.log(
            "bins",
            self._bins_tracker,
            on_step=True,
            on_epoch=False,
            logger=True,
        )

        return loss

    def configure_optimizers(self):
        return self.optimizer_type(self.parameters(), lr=self.learning_rate)

    def optimizer_step(self, *args, **kwargs) -> None:
        super().optimizer_step(*args, **kwargs)
        self.ema_update()

    @torch.no_grad()
    def ema_update(self):
        param = [p.data for p in self.model.parameters()]
        param_ema = [p.data for p in self.model_ema.parameters()]

        torch._foreach_mul_(param_ema, self.ema_decay)
        torch._foreach_add_(param_ema, param, alpha=1 - self.ema_decay)

        self._ema_decay_tracker(self.ema_decay)
        self.log(
            "ema_decay",
            self._ema_decay_tracker,
            on_step=True,
            on_epoch=False,
            logger=True,
        )

    @property
    def ema_decay(self):
        return math.exp(
            self.bins_min * math.log(self.initial_ema_decay) / self.bins
        )

    @property
    def bins(self) -> int:
        return (
            math.ceil(
                math.sqrt(
                    self.trainer.global_step
                    / self.trainer.estimated_stepping_batches
                    * (self.bins_max**2 - self.bins_min**2)
                    + self.bins_min**2
                )
                - 1
            )
            + 1
        )

    def timesteps_to_times(self, timesteps: torch.LongTensor, bins: int):
        return (
            (
                self.time_min ** (1 / self.bins_rho)
                + timesteps
                / (bins - 1)
                * (
                    self.time_max ** (1 / self.bins_rho)
                    - self.time_min ** (1 / self.bins_rho)
                )
            )
            .pow(self.bins_rho)
            .clamp(0, self.time_max)
        )

    @rank_zero_only
    def on_train_start(self) -> None:
        self.save_samples(f"{0:05}")

    @rank_zero_only
    def on_train_epoch_end(self) -> None:
        self.save_samples(f"{(self.current_epoch+1):05}")

    @torch.no_grad()
    def sample(
        self,
        num_samples: int = 16,
        steps: int = 1,
        seed: int = 0,
    ) -> torch.Tensor:
        generator = torch.Generator(device=self.device).manual_seed(seed)
        time = torch.tensor([self.time_max], device=self.device)
        images: torch.Tensor = self(
            torch.randn(
                num_samples,
                self.channels,
                self.image_size,
                self.image_size,
                device=self.device,
                generator=generator,
            )
            * time,
            time,
        )

        if steps <= 1:
            return images

        times = list(
            reversed(range(0, self.bins_max, self.bins_max // steps - 1))
        )[1:]
        times = [t + self.bins_max // ((steps - 1) * 2) for t in times]

        for time in times:
            noise = torch.randn(
                num_samples,
                self.channels,
                self.image_size,
                self.image_size,
                device=self.device,
                generator=generator,
            )
            images = images + math.sqrt(time**2 - self.time_min**2) * noise
            images = self(images, torch.tensor([time], device=self.device))

        return images

    def save_samples(self, filename: str):
        samples = self.sample()
        samples.mul_(0.5).add_(0.5)
        grid = make_grid(
            samples,
            nrow=math.ceil(math.sqrt(samples.size(0))),
            padding=self.image_size // 16,
        )

        save_image(
            grid,
            f"{self.samples_path}/{filename}.png",
            "png",
        )

        if isinstance(self.trainer.logger, WandbLogger):
            wandb.log(
                {
                    "samples": wandb.Image(
                        grid.permute(1, 2, 0).cpu().numpy()
                    ),
                },
                commit=False,
            )

    @staticmethod
    def image_time_product(images: torch.Tensor, times: torch.Tensor):
        return torch.einsum("b c h w, b -> b c h w", images, times)
