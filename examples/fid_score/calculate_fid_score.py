import argparse
import contextlib
import io
import shutil
from pathlib import Path

import torch
from cleanfid import fid
from datasets import load_dataset
from diffusers import DiffusionPipeline
from PIL import Image
from tqdm import tqdm

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def parse_args():
    parser = argparse.ArgumentParser(description="Calculate FID score for Consistency Models")
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
        "--split",
        type=str,
        default="test",
    )
    parser.add_argument(
        "-p",
        "--pretrained_model_name_or_path",
        type=str,
        default="consistency/cifar10-32-demo",
    )
    parser.add_argument(
        "-n",
        "--fid-num-samples",
        type=int,
        default=10000,
    )

    args = parser.parse_args()
    return args


def main(args):
    pipeline = DiffusionPipeline.from_pretrained(
        args.pretrained_model_name_or_path,
        custom_pipeline="consistency/pipeline",
    ).to(DEVICE)

    dataset = load_dataset(args.dataset_name, args.dataset_config_name, split=args.split)

    shutil.rmtree("evaluation", ignore_errors=True)
    Path("evaluation/real").mkdir(parents=True, exist_ok=True)
    Path("evaluation/generated").mkdir(parents=True, exist_ok=True)

    for i in range(min(args.fid_num_samples, len(dataset))):
        image: Image.Image = dataset[i]["img"]
        image.save("evaluation/real/" + f"{i:06}.jpeg")

    for i in tqdm(range(args.fid_num_samples), desc="generating images..."):
        with contextlib.redirect_stdout(io.StringIO()):
            image: Image.Image = pipeline().images[0]
        image.save("evaluation/generated/" + f"{i:06}.jpeg")  # TODO: async

    score = fid.compute_fid(
        fdir1="evaluation/real",
        fdir2="evaluation/generated",
        mode="clean",
        device=DEVICE,
    )

    print(f"FID on {args.dataset_name} for {args.pretrained_model_name_or_path}: {score}")


if __name__ == "__main__":
    args = parse_args()
    main(args)
