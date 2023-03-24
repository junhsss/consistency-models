from setuptools import find_packages, setup

__version__ = "0.2.4"

setup(
    name="consistency",
    packages=find_packages(),
    version=__version__,
    license="MIT",
    description="Consistency Model - Pytorch",
    author="junhsss",
    author_email="junhsssr@gmail.com",
    url="https://github.com/junhsss/consistency-models",
    long_description_content_type="text/markdown",
    keywords=["artificial intelligence", "diffusion models"],
    install_requires=[
        "pillow",
        "torch",
        "torchvision",
        "pytorch-lightning",
        "diffusers",
        "torchmetrics",
        "lpips",
    ],
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Developers",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3.8",
    ],
)
