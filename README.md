# FairGRAPE: Fairness-aware GRAdient Pruning mEthod for Face Attribute Classification

This repo presents an official implementation of FairGRAPE: Fairness-aware GRAdient Pruning mEthod for Face Attribute Classification

![FairGRAPE_concept2](https://user-images.githubusercontent.com/60991389/177890025-4599bd0f-176d-4f5f-aff8-73df9c963a6e.png)

## Dependencies

The code has been tested on the following environment:

```
python 3.9
pytorch 1.11。0
dlib 19.22.0
opencv2 4.5.5
```

Use Anaconda and the following command to replicate the full environment:

```
conda env create -f environment.yml
```

## Datasets

This code automatically downloads the following datasets for trianing, cross validation and testing: [FairFace](https://github.com/joojs/fairface), [UTKFace](https://susanqq.github.io/UTKFace/) and [CelebA](https://mmlab.ie.cuhk.edu.hk/projects/CelebA.html). Please request the person subtree of Imagenet through the offical [database](https://image-net.org/)


## Example Usage:

UTKFace experiments:
```
python main_test.py --sensitive_group 'gender' --loss_type 'race' --prune_type 'FairGRAPE' --network 'resnet34' --dataset 'UTKFace'  --prune_rate 0.9 --keep_per_iter 0.975
```

FairFace experiments
```
python main_test.py --sensitive_group race--loss_type 'race' --prune_type 'FairGRAPE' --network 'resnet34' --dataset 'FairFace'  --prune_rate 0.99
```

CelebA experiments
```
python main_test.py --sensitive_group 'gender' --loss_type 'classes' --prune_type 'FairGRAPE' --network 'mobilenetv2' --dataset 'CelebA' --prune_rate 0.9
```

Imagenet experiments:
```
python main_test.py --sensitive_group 'gender' --loss_type 'classes' --prune_type 'FairGRAPE' --network 'resnet34' --dataset 'Imagenet'  --prune_rate 0.5
```

Download trained models [here](https://www.dropbox.com/sh/rk362mypuikeklh/AADF93dWPQo3rPTUhyaLBn3Ga?dl=0).


## Acknowledgement 
Parts of code were borrowed from [SNIP](https://github.com/mil-ad/snip), [WS](https://github.com/mightydeveloper/Deep-Compression-PyTorch), [GraSP](https://github.com/alecwangcq/GraSP)
