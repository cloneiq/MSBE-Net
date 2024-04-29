## MSBE-Net:A multi-scale perception and boundary enhancement network for improving polyp segmentation performance

We investigate a novel polyp segmentation  network (MSBE-Net) based on multi-scale perception and boundary enhancement, aiming to improve polyp segmentation. The MSBE-Net contains a camouflaged object recognition module(CORM),a boundary feature enhancement module(BFEM),a dense receptive field module(DRF) . 

The details of this project are presented in the following paper:

[MSBE-Net:A multi-scale perception and boundary enhancement network for improving polyp segmentation performance]

## Usage 
### Setup 
```
Python 3.8
Pytorch 2.0.0
```
### Dataset 
Downloading training and testing datasets and move them into ./dataset/, which can be found in this [Baidu Drive](https://pan.baidu.com/s/1yP2VV-q78fZjCjMUMA9Agw) [code:o2mo].
### Pre-trained model 
Load pre-trained models from [Baidu Drive](https://pan.baidu.com/s/1nLaFNOt2WDU38hZL9LINWA) [code:01y5]
### Train the model 
Clone the repository
```
git clone 
cd MSBE-Net
bash train.sh
```
### Test the model
```
cd MSBE-Net 
bash test.sh
```

### Well-trained model 
Loading Well-trained model from [Baidu Drive](https://pan.baidu.com/s/1dVoGXWEqcpR-T2MacWi9Vw) [code:sy0v].

##  License
The source code is free for research and education use only. Any commercial use should get formal permission first.

## Acknowledgement
Thanks [Polyp-PVT](https://github.com/DengPingFan/Polyp-PVT) for serving as building blocks of MSBE-Net.