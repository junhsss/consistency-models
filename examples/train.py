import argparse
import math

import torch
from datasets import load_dataset
from diffusers import UNet2DModel
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers.wandb import WandbLogger
from torch.utils.data import DataLoader
from torchvision import transforms

from consistency import Consistency
from consistency.loss import PerceptualLoss


def parse_args():
    parser = argparse.ArgumentParser(
        description="A simple training script for consistency models."
    )
    parser.add_argument(
        "--dataset-name",
        type=str,
        default="cifar10",
    )
    parser.add_argument(
        "--dataset-config-name",
        type=str,
        default=None,
        help="The config of the Dataset, leave as None if there's only one config.",
    )
    parser.add_argument(
        "--resolution",
        type=int,
        default=32,
        help="The resolution for input images.",
    )
    parser.add_argument(
        "--train-batch-size",
        type=int,
        default=16,
        help="Batch size (per device) for the training dataloader.",
    )
    parser.add_argument(
        "--dataloader-num-workers",
        type=int,
        default=8,
    )
    parser.add_argument("--max-epochs", type=int, default=200)
    parser.add_argument(
        "--learning-rate",
        type=float,
        default=1e-4,
        help="Initial learning rate (after the potential warmup period) to use.",
    )
    parser.add_argument(
        "--data-std",
        type=int,
        default=0.5,
        help="Standard deviation of the dataset",
    )
    parser.add_argument(
        "--time-min",
        type=float,
        default=0.002,
    )
    parser.add_argument(
        "--time-max",
        type=float,
        default=80,
    )
    parser.add_argument(
        "--bins-min",
        type=float,
        default=2,
    )
    parser.add_argument(
        "--bins-max",
        type=float,
        default=150,
    )
    parser.add_argument(
        "--bins-rho",
        type=float,
        default=7,
    )
    parser.add_argument(
        "--initial-ema-decay",
        type=float,
        default=0.9,
    )
    parser.add_argument(
        "--log-every-n-steps",
        type=int,
        default=100,
    )
    parser.add_argument(
        "--sample-path",
        type=str,
        default="samples/",
    )
    parser.add_argument(
        "--save-samples-every-n-epoch",
        type=int,
        default=10,
    )
    parser.add_argument(
        "--num-samples",
        type=int,
        default=16,
        help="The number of images to generate for evaluation.",
    )
    parser.add_argument(
        "--sample-steps",
        type=int,
        default=5,
    )
    parser.add_argument("--sample-ema", action="store_true")
    parser.add_argument(
        "--sample-seed",
        type=int,
        default=0,
    )
    parser.add_argument("--wandb-project", type=str, default="consistency")
    parser.add_argument("--ckpt-path", type=str, default="ckpt")
    parser.add_argument("--unet-config-path", type=str)

    parser.add_argument("--resume-ckpt-path", type=str)
    parser.add_argument("--resume-wandb-id", type=str)
    args = parser.parse_args()
    return args


def main(args):
    augmentations = transforms.Compose(
        [
            transforms.Resize(
                args.resolution,
                interpolation=transforms.InterpolationMode.BILINEAR,
            ),
            transforms.CenterCrop(args.resolution),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize([0.5], [0.5]),
        ]
    )

    class Dataset(torch.utils.data.Dataset):
        def __init__(self, dataset_name: str, dataset_config_name=None):
            self.dataset = load_dataset(
                dataset_name,
                dataset_config_name,
                split="train",
            )
            self.image_key = [
                key for key in ("image", "img") if key in self.dataset[0]
            ][0]

        def __len__(self):
            return len(self.dataset)

        def __getitem__(self, index: int) -> torch.Tensor:
            return augmentations(self.dataset[index][self.image_key].convert("RGB"))

    dataloader = DataLoader(
        Dataset(
            args.dataset_name,
            args.dataset_config_name,
        ),
        batch_size=args.train_batch_size,
        shuffle=True,
        num_workers=args.dataloader_num_workers,
        pin_memory=True,
        persistent_workers=True,
    )

    if args.unet_config_path:
        import json

        with open(args.unet_config_path) as f:
            config = json.load(f)

        UNet2DModel.from_config(config)
    else:
        # Simplified NCSN++ Architecture
        # See https://huggingface.co/google/ncsnpp-ffhq-1024/blob/main/config.json
        unet = UNet2DModel(
            sample_size=args.resolution,
            in_channels=3,
            out_channels=3,
            layers_per_block=4,
            attention_head_dim=8,
            block_out_channels=(128, 256, 256, 256),
            down_block_types=(
                "SkipDownBlock2D",
                "AttnSkipDownBlock2D",
                "SkipDownBlock2D",
                "SkipDownBlock2D",
            ),
            downsample_padding=1,
            act_fn="silu",
            center_input_sample=True,
            mid_block_scale_factor=math.sqrt(2),
            up_block_types=(
                "SkipUpBlock2D",
                "SkipUpBlock2D",
                "AttnSkipUpBlock2D",
                "SkipUpBlock2D",
            ),
        )
    # Use both VGG and SqueezeNet as loss
    loss_fn = PerceptualLoss(net_type=("vgg", "squeeze"))

    if args.resume_ckpt_path:
        consistency = Consistency.load_from_checkpoint(
            checkpoint_path=args.resume_ckpt_path,
            model=unet,
            loss_fn=loss_fn,
            learning_rate=args.learning_rate,
            data_std=args.data_std,
            time_min=args.time_min,
            time_max=args.time_max,
            bins_min=args.bins_min,
            bins_max=args.bins_max,
            bins_rho=args.bins_rho,
            initial_ema_decay=args.initial_ema_decay,
            samples_path=args.sample_path,
            save_samples_every_n_epoch=args.save_samples_every_n_epoch,
            num_samples=args.num_samples,
            sample_steps=args.sample_steps,
            sample_ema=args.sample_ema,
            sample_seed=args.sample_seed,
        )

        trainer = Trainer(
            accelerator="auto",
            logger=WandbLogger(
                project=args.wandb_project,
                log_model=True,
                id=args.resume_wandb_id,
                resume="must",
            )
            if args.wandb_id
            else WandbLogger(
                project=args.wandb_project,
                log_model=True,
            ),
            callbacks=[
                ModelCheckpoint(
                    dirpath="ckpt",
                    save_top_k=3,
                    monitor="loss",
                )
            ],
            max_epochs=args.max_epochs,
            precision=32,
            log_every_n_steps=args.log_every_n_steps,
            gradient_clip_algorithm="norm",
            gradient_clip_val=1.0,
        )
        trainer.fit(consistency, dataloader, ckpt_path=args.ckpt_path)

    else:
        consistency = Consistency(
            model=unet,
            loss_fn=loss_fn,
            learning_rate=args.learning_rate,
            data_std=args.data_std,
            time_min=args.time_min,
            time_max=args.time_max,
            bins_min=args.bins_min,
            bins_max=args.bins_max,
            bins_rho=args.bins_rho,
            initial_ema_decay=args.initial_ema_decay,
            samples_path=args.sample_path,
            save_samples_every_n_epoch=args.save_samples_every_n_epoch,
            num_samples=args.num_samples,
            sample_steps=args.sample_steps,
            sample_ema=args.sample_ema,
            sample_seed=args.sample_seed,
        )

        trainer = Trainer(
            accelerator="auto",
            logger=WandbLogger(project=args.wandb_project, log_model=True),
            callbacks=[
                ModelCheckpoint(
                    dirpath="ckpt",
                    save_top_k=3,
                    monitor="loss",
                )
            ],
            max_epochs=args.max_epochs,
            precision=16,
            log_every_n_steps=args.log_every_n_steps,
            gradient_clip_algorithm="norm",
            gradient_clip_val=1.0,
        )

        trainer.fit(consistency, dataloader)


if __name__ == "__main__":
    args = parse_args()
    main(args)
