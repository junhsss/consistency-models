# **Consistency Models** 🌃

**Single-step** image generation with [Consistency Models](https://arxiv.org/abs/2303.01469).

<br />

<img src="./assets/training.gif" />

<br />

**Consistency Models** are a new family of generative models that achieve high sample quality without adversarial training. They support _fast one-step generation_ by design, while still allowing for few-step sampling to trade compute for sample quality.

<br />

## Installation

```sh
$ pip install consistency
```

### Note

You **don't need to install** `consistency` for just trying things out:

```python
from diffusers import DiffusionPipeline

pipeline = DiffusionPipeline.from_pretrained(
    "consistency/cifar10-32-demo",
    custom_pipeline="consistency/pipeline",
)

pipeline().images[0]  # Super Fast Generation! 🤯
```

<br />

## Quickstart

### Basic

Just wrap your favorite _U-Net_ with `Consistency`.

```python
import torch
from diffusers import UNet2DModel
from consistency import Consistency
from consistency.loss import PerceptualLoss

consistency = Consistency(
    model=UNet2DModel(sample_size=224),
    loss_fn=PerceptualLoss(net_type=("vgg", "squeeze"))
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

### Push to HF Hub

Provide your `model_id` and `token` to `Consistency`.

```python
consistency = Consistency(
    model=UNet2DModel(sample_size=224),
    loss_fn=PerceptualLoss(net_type=("vgg", "squeeze"))
    model_id="your_model_id",
    token="your_token"  # Not needed if logged in via huggingface-cli
    push_every_n_steps=10000,
)
```

You can safely drop `consistency` afterwards. Good luck! 🤞

```python
from diffusers import DiffusionPipeline

pipeline = DiffusionPipeline.from_pretrained(
    "your_model_id",
    custom_pipeline="consistency/pipeline",
)

pipeline().images[0]
```

A complete example can be found in [this **script**](https://github.com/junhsss/consistency-models/blob/main/examples/train.py) or in [this **colab notebook**](https://colab.research.google.com/github/junhsss/consistency-models/blob/main/examples/consistency_models.ipynb).

Checkout [this **Wandb workspace**](https://wandb.ai/junhsss/consistency?workspace=user-junhsss) for some experiment results.

<br />

## Available Models

| model_id                                                                                                                  | dataset                                                                                    |
| ------------------------------------------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------ |
| <a href="https://huggingface.co/consistency/cifar10-32-demo" target="_blank"><code>consistency/cifar10-32-demo</code></a> | <a href="https://huggingface.co/datasets/cifar10" target="_blank"><code>cifar10</code></a> |

If you've trained some checkpoints using `consistency`, **share with us! 🤗**

<br />

## Documentation

In progress... 🛠

<br />

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
