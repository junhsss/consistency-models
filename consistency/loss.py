from collections.abc import Sequence
from typing import Tuple, Union

import torch.nn.functional as F
from torch import nn
from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity


class PerceptualLoss(nn.Module):
    def __init__(
        self,
        *,
        net_type: Union[str, Tuple[str, ...]] = "vgg",
        l1_weight: float = 1.0,
        **kwargs,
    ):
        super().__init__()

        available_net_types = ("vgg", "alex", "squeeze")

        def _append_net_type(net_type: str):
            if net_type in available_net_types:
                self.lpips_losses.append(
                    LearnedPerceptualImagePatchSimilarity(net_type)
                )
            else:
                raise TypeError(
                    f"'net_type' should be on of {available_net_types}, got {net_type}"
                )

        self.lpips_losses = nn.ModuleList()

        if isinstance(net_type, str) and net_type in available_net_types:
            _append_net_type(net_type)

        elif isinstance(net_type, Sequence):
            for _net_type in sorted(net_type):
                _append_net_type(_net_type)

        self.lpips_losses.requires_grad_(False)

        self.l1_weight = l1_weight

    def forward(self, input, target):
        upscaled_input = F.interpolate(input, (224, 224), mode="bilinear")
        upscaled_target = F.interpolate(target, (224, 224), mode="bilinear")

        lpips_loss = sum(
            _lpips_loss(upscaled_input, upscaled_target)
            for _lpips_loss in self.lpips_losses
        )

        return lpips_loss + self.l1_weight * F.l1_loss(input, target)


LPIPSLoss = PerceptualLoss
