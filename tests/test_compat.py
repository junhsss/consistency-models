import numpy as np
import torch
from diffusers import UNet2DModel

from consistency import Consistency
from consistency.diffusers import ConsistencyPipeline, ConsistencyScheduler


def test_diffusers_pipeline():
    unet = UNet2DModel(sample_size=32)
    pipe = ConsistencyPipeline(
        unet=unet,
        scheduler=ConsistencyScheduler(time_min=0.002, time_max=80, data_std=0.5),
    )
    a = pipe(
        num_inference_steps=1,
        generator=torch.Generator().manual_seed(0),
        output_type=None,
    ).images
    b = (
        Consistency(model=unet)
        .sample(
            num_samples=1,
            steps=1,
            use_ema=False,
            generator=torch.Generator().manual_seed(0),
        )
        .cpu()
        .permute(0, 2, 3, 1)
        .numpy()
    )
    assert np.allclose(a, b)


test_diffusers_pipeline()
