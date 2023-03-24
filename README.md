# **Consistency Models** 🌃

**Single-step** image generation with [Consistency Models](https://arxiv.org/abs/2303.01469).

<br />

<img src="./assets/training.gif" />

<br />

**Consistency Models** are a new family of generative models that achieve high sample quality without adversarial training. They support _fast one-step generation_ by design, while still allowing for few-step sampling to trade compute for sample quality.

## Installation

```sh
$ pip install consistency
```

## Usage

Just wrap your favorite _U-Net_ with `Consistency`. 😊

```python
import torch
from diffusers import UNet2DModel
from consistency import Consistency
from consistency.loss import PerceptualLoss

consistency = Consistency(
    model=UNet2DModel(sample_size=224),
    loss_fn=PerceptualLoss(net_type=("vgg", "squeeze"))
    learning_rate=1e-4,
)

samples = consistency.sample(16)

# multi-step sampling, sample from the ema model
samples = consistency.sample(16, steps=5, use_ema=True)
```

`Consistency` is self-contained with the training logic and all necessary schedules.

You can train it with **PyTorch Lightning**'s `Trainer` 🚀

```python
from pytorch_lightning import Trainer

trainer = Trainer(max_epochs=8000, accelerator="auto")
trainer.fit(consistency, some_dataloader)
```

A complete example can be found in [this **script**](https://github.com/junhsss/consistency-models/blob/main/examples/train.py) or in [this **colab notebook**](https://colab.research.google.com/github/junhsss/consistency-models/blob/main/examples/consistency_models.ipynb).

Checkout [this **Wandb workspace**](https://wandb.ai/junhsss/consistency?workspace=user-junhsss) for some experiment results. (still training... 🏗)

## Reference

```bibtex
@misc{https://doi.org/10.48550/arxiv.2303.01469,
  doi       = {10.48550/ARXIV.2303.01469},
  url       = {https://arxiv.org/abs/2303.01469},
  author    = {Song, Yang and Dhariwal, Prafulla and Chen, Mark and Sutskever, Ilya},
  keywords  = {Machine Learning (cs.LG), Computer Vision and Pattern Recognition (cs.CV), Machine Learning (stat.ML), FOS: Computer and information sciences, FOS: Computer and information sciences},
  title     = {Consistency Models},
  publisher = {arXiv},
  year      = {2023},
  copyright = {arXiv.org perpetual, non-exclusive license}
}
```
