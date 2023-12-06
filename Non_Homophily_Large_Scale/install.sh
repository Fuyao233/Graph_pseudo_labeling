#!/bin/bash

#CUDA=$1
#
#pip install torch==1.7.1+$CUDA -f https://download.pytorch.org/whl/torch_stable.html
#
#pip install scipy==1.6.0
#pip install ogb==1.2.4
#pip install flake8==3.8.4
#pip install Babel==2.9.0
#pip install flask==1.1.2
#pip install gdown==4.4.0
#
#
#pip install --no-index torch-scatter==2.0.5 -f https://pytorch-geometric.com/whl/torch-1.7.0+$CUDA.html
#pip install --no-index torch-sparse==0.6.8 -f https://pytorch-geometric.com/whl/torch-1.7.0+$CUDA.html
#pip install --no-index torch-cluster==1.5.8 -f https://pytorch-geometric.com/whl/torch-1.7.0+$CUDA.html
#pip install --no-index torch-spline-conv==1.2.0 -f https://pytorch-geometric.com/whl/torch-1.7.0+$CUDA.html
#pip install torch-geometric==1.6.3

#if conda
conda install pytorch==2.0.0 torchvision==0.15.0 torchaudio==2.0.0 pytorch-cuda=11.7 -c pytorch -c nvidia
conda install pyg -c pyg
conda install scipy flake8 Babel flask
conda install -c conda-forge ogb
conda install -c conda-forge gdown
