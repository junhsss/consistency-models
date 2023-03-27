# **Consistency Models** 🌃

**Single-step** image generation with [Consistency Models](https://arxiv.org/abs/2303.01469).

<br />

<img src="./assets/training.gif" />

<br />

**Consistency Models** are a new family of generative models that achieve high sample quality without adversarial training. They support _fast one-step generation_ by design, while still allowing for few-step sampling to trade compute for sample quality.

### Note

If you just want to try things out, just do:

```python
from diffusers import DiffusionPipeline

pipeline = DiffusionPipeline("consistency/cifar10-32-demo", custom_pipeline="consistency/pipeline")

pipeline().images[0]  # Super Fast Generation! 🤯
```

## Installation

```sh
$ pip install consistency
```

## Quickstart

### Basic

Just wrap your favorite _U-Net_ with `Consistency`. 😊

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

You can train `Consistency` with **PyTorch Lightning**'s `Trainer` 🚀

```python
from pytorch_lightning import Trainer

trainer = Trainer(max_epochs=8000, accelerator="auto")
trainer.fit(consistency, some_dataloader)
```

<br />

A complete example can be found in [this **script**](https://github.com/junhsss/consistency-models/blob/main/examples/train.py) or in [this **colab notebook**](https://colab.research.google.com/github/junhsss/consistency-models/blob/main/examples/consistency_models.ipynb).

Checkout [this **Wandb workspace**](https://wandb.ai/junhsss/consistency?workspace=user-junhsss) for some experiment results.

### Push to HF Hub

Just provide your `model_id` and `token`!

```python
consistency = Consistency(
    model=UNet2DModel(sample_size=224),
    loss_fn=PerceptualLoss(net_type=("vgg", "squeeze"))
    model_id="your_model_id",
    token="your_token"  # Not needed if logged in via huggingface-cli
    push_every_n_steps=10000,
)
```

You can safely uninstall `consistency` afterwards. Good luck! 🤞:

```python
from diffusers import DiffusionPipeline

pipeline = DiffusionPipeline("your_model_id", custom_pipeline="consistency/pipeline")

pipeline().images[0]
```

## Available Models

| model_id                    | sample_size |
| --------------------------- | ----------- |
| consistency/cifar10-32-demo | 32          |

If you've trained some checkpoints. **Share with us! 🤗**

## Documentation

In progress... 🛠

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

## Todo

- [ ] `diffusers` integration. (`ConsistencyPipeline` + `ConsistencyScheduler`)
