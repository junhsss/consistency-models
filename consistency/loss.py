from typing import Literal

from torch import nn
from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity


class LPIPSLoss(nn.Module):
    def __init__(self, net_type: Literal["vgg", "alex", "squeeze"] = "vgg"):
        super().__init__()
        self.lpips = LearnedPerceptualImagePatchSimilarity(net_type=net_type)
        self.lpips.requires_grad_(False)

    @staticmethod
    def clamp(x):
        return x.clamp(-1, 1)

    def forward(self, input, target):
        lpips_loss = self.lpips(self.clamp(input), self.clamp(target))
        return lpips_loss
