# **Real-time Semantic Segmentation of Remote Sensing Images Using a Dual-Branch Network with Cross-Sampling Graph Convolution** <br />

This is a pytorch implementation of CSGCNet.

<div align="center">
  <img src="https://github.com/AnsonD0820/CSGCNet-pytorch/blob/main/fig1.jpg">
</div> <br />

## Prerequisites <br />
* Python 3.9
* Pytorch 2.0.0
* torchvision 0.15.0
* OpenCV
* CUDA >= 11.7

## Train
### Train potsdam dataset
bash train_CSGCNet_potsdam.sh

### Train UDD6 dataset
bash train_CSGCNet_UDD6.sh <br />

## Evaluate
### Evaluate potsdam dataset
bash predict_CSGCNet_potsdam.sh

### Evaluate UDD6 dataset
bash predict_CSGCNet_UDD6.sh
