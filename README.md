## Installation

gpu-dryrun works with Python 3.11+, and [PyTorch](https://pytorch.org/get-started/locally/) 2.7+.

Create and activate a virtual environment with [conda](https://docs.conda.io/en/latest/).

```py
# conda
conda create -n gpu-dryrun python=3.11
conda activate gpu-dryrun

# install cuda-related package
conda install cuda-version=12.1 cudnn=8.9.7.29

# install torch-related package
pip install torch==2.5.1+cu121 --index-url https://mirror.sjtu.edu.cn/pytorch-wheels/cu121
```

Install relevant pip library in your virtual environment.

```py
# pip
pip install nvidia-ml-py
pip install numpy
```
