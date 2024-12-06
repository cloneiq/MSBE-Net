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
- downloading training dataset and move it into ```./data/TrainDataset/```. It contains two sub-datasets: Kvasir-SEG (900 train samples)[Link](https://datasets.simula.no/kvasir-seg/) and CVC-ClinicDB (550 train samples)[Link](https://polyp.grand-challenge.org/CVCClinicDB/).
- downloading testing dataset and move it into ```./data/TestDataset/```. It contains five sub-datsets: CVC-300 (60 test samples), CVC-ClinicDB (62 test samples), CVC-ColonDB (380 test samples)[Link](http://vi.cvc.uab.es/colon-qa/cvccolondb/), ETIS-LaribPolypDB (196 test samples)[Link](https://polyp.grand-challenge.org/ETISLarib/), Kvasir (100 test samples).
### Pre-trained model 
Load pre-trained models from 
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
Loading Well-trained model from [link](https://github.com/yuhuan-wu/P2T).

##  License
The source code is free for research and education use only. Any commercial use should get formal permission first.

## Acknowledgement
Thanks [Polyp-PVT](https://github.com/DengPingFan/Polyp-PVT) for serving as building blocks of MSBE-Net.
