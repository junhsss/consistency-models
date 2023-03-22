import sys

if sys.version_info < (3, 7):
    from typing_extensions import Literal
else:
    from typing import Literal

from torch import nn
from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity


class LPIPSLoss(nn.Module):
    def __init__(
        self,
        net_type: Literal["vgg", "alex", "squeeze"] = "vgg",
        overflow_weight: float = 1.0,
    ):
        super().__init__()
        self.lpips = LearnedPerceptualImagePatchSimilarity(net_type=net_type)
        self.lpips.requires_grad_(False)

        self.overflow_weight = overflow_weight

    @staticmethod
    def clamp(x):
        return x.clamp(-1, 1)

    def forward(self, input, target):
        lpips_loss = self.lpips(self.clamp(input), self.clamp(target))
        overflow_loss = nn.functional.l1_loss(input, self.clamp(input))
        return lpips_loss + self.overflow_weight * overflow_loss
