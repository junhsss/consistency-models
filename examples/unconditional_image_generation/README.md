## Training examples

### Installing the dependencies

Before running the scripts, make sure to install the library's training dependencies:

**Important**

To make sure you can successfully run the latest versions of the example scripts, we highly recommend **installing from source** and keeping the install up to date as we update the example scripts frequently and install some example-specific requirements. To do this, execute the following steps in a new virtual environment:

```bash
git clone https://github.com/junhsss/consistency-models
cd consistency-models
pip install .
```

Then cd in the example folder and run

```bash
pip install -r requirements.txt
```

### Unconditional CIFAR-10

The command to train a UNet model on the CIFAR-10 dataset:

```bash
python train_unconditional.py \
  --dataset-name="cifar10" \
  --resolution=32 \
  --model-id="cm-cifar10-32" \
  --train-batch-size=32 \
  --max-epochs=100 \
  --use-ema \
  --learning-rate=1e-4 \
  --push-to-hub
```

In practice, you might want a larger batch size and a deeper architecture like [NCSN++](https://arxiv.org/abs/2011.13456) or [its variants](https://arxiv.org/abs/2105.05233). The proper configurations can be found in [HF hub](https://huggingface.co/google/ncsnpp-ffhq-1024).
