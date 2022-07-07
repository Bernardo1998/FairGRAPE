# FairGRAPE: Fairness-aware GRAdient Pruning mEthod for Face Attribute Classification

## This repo presents an official implementation of FairGRAPE: Fairness-aware GRAdient Pruning mEthod for Face Attribute Classification

## Dependencies

```
pytorch 1.11
dlib 19.22.0
opencv2 4.5.5
```

## Datasets

This code automatically downloads the following datasets for trianing, cross validation and testing: [FairFace](https://github.com/joojs/fairface), [UTKFace](https://susanqq.github.io/UTKFace/) and [CelebA](https://mmlab.ie.cuhk.edu.hk/projects/CelebA.html). Please request the person subtree of Imagenet through the offical [database](https://image-net.org/)


## Usage:

FairFace experiments
```
python main_test.py --sensitive_group 'gender' --loss_type 'race' --prune_type 'FairGRAPE' --network 'resnet34' --dataset 'FairFace'  --prune_rate 0.99
```

UTKFace experiments:
```
python main_test.py --sensitive_group 'gender' --loss_type 'race' --prune_type 'FairGRAPE' --network 'resnet34' --dataset 'UTKFace'  --prune_rate 0.9 --keep_per_iter 0.975
```

CelebA experiments
```
python main_test.py --sensitive_group 'gender' --loss_type 'classes' --prune_type 'FairGRAPE' --network 'mobilenetv2' --dataset 'CelebA' --prune_rate 0.9
```

Imagenet experiments:
```
python main_test.py --sensitive_group 'gender' --loss_type 'classes' --prune_type 'FairGRAPE' --network 'resnet34' --dataset 'Imagenet'  --prune_rate 0.5
```


## Acknowledgement 
Parts of code were borrowed from [SNIP](https://github.com/mil-ad/snip), [WS](https://github.com/mightydeveloper/Deep-Compression-PyTorch),[GraSP](https://github.com/alecwangcq/GraSP)