# Graph Contrastive Learning with Automated

PyTorch implementation for [Graph Contrastive Learning Automated]() [[talk]]() [[poster]]() [[appendix]]()

Yuning You, Tianlong Chen, Yang Shen, Zhangyang Wang

In ICML 2021.

## Overview

In this repository, we propose a principled framework named joint augmentation selection (JOAO), to automatically, adaptively and dynamically select augmentations during [GraphCL](https://arxiv.org/abs/2010.13902) training.
Sanity check shows that the selection aligns with previous "best practices", as shown in Figure 2.

![](./joao.png)

## Dependencies

```
torch-geometric>=1.6.0
ogb==1.2.4
```

## Experiments

* Semi-supervised learning [[TU Datasets]](https://github.com/Shen-Lab/GraphCL_Automated/tree/master/semisupervised_TU) [[OGB]](https://github.com/Shen-Lab/GraphCL_Automated/tree/master/semisupervised_OGB)

* Unsupervised representation learning [[TU Datasets]](https://github.com/Shen-Lab/GraphCL_Automated/tree/master/unsupervised_TU)

* Transfer learning [[MoleculeNet and PPI]](https://github.com/Shen-Lab/GraphCL_Automated/tree/master/transferLearning_MoleculeNet_PPI)

## Citation

If you use this code for you research, please cite our paper.

```
TBD
```
