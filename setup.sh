#!/bin/bash

wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
bash Miniconda3-latest-Linux-x86_64.sh
source ~/.bashrc

conda create -n gnn_env python=3.10 -y

conda activate gnn_env

pip install "numpy<2.0"
# Install PyTorch with CUDA support
pip install torch==2.0.1+cu118 torchvision==0.15.2+cu118 torchaudio==2.0.2+cu118 --extra-index-url https://download.pytorch.org/whl/cu118

# Install PyTorch Geometric dependencies
pip install torch-scatter -f https://data.pyg.org/whl/torch-2.0.1+cu118.html
pip install torch-sparse -f https://data.pyg.org/whl/torch-2.0.1+cu118.html
pip install torch-cluster -f https://data.pyg.org/whl/torch-2.0.1+cu118.html
pip install torch-spline-conv -f https://data.pyg.org/whl/torch-2.0.1+cu118.html
pip install torch-geometric

# Install other dependencies
pip install numpy<2.0 networkx scikit-learn matplotlib psutil sympy==1.13.1 seaborn pandas python-louvain powerlaw ogb numpy