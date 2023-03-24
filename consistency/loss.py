from collections.abc import Sequence
from typing import Tuple, Union

import torch.nn.functional as F
from torch import nn
from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity


class PerceptualLoss(nn.Module):
    def __init__(
        self,
        net_type: Union[str, Tuple[str, ...]] = "vgg",
        overflow_weight: float = 5.0,
        l1_weight: float = 1.0,
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

        self.overflow_weight = overflow_weight
        self.l1_weight = l1_weight

    @staticmethod
    def clamp(x):
        return x.clamp(-1, 1)

    def forward(self, input, target):
        clampped_input = self.clamp(input)
        clampped_target = self.clamp(target)

        lpips_loss = sum(
            _lpips_loss(clampped_input, clampped_target)
            for _lpips_loss in self.lpips_losses
        )

        return (
            lpips_loss
            + self.overflow_weight * F.l1_loss(input, self.clamp(input))
            + self.l1_weight * F.l1_loss(input, target)
        )


LPIPSLoss = PerceptualLoss
