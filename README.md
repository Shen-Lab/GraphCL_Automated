# Graph Contrastive Learning with Automated

PyTorch implementation for [Graph Contrastive Learning Automated]() [[talk]]() [[poster]]() [[appendix]]()

Yuning You, Tianlong Chen, Yang Shen, Zhangyang Wang

In ICML 2021.

## Overview

In this repository, we propose a principled framework named joint augmentation selection (JOAO), to automatically, adaptively and dynamically select augmentations during [GraphCL](https://arxiv.org/abs/2010.13902) training.
Sanity check shows that the selection aligns with previous "best practices", as shown in Figure 2.

![](./joao.png)

## Experiments

* [The Role of Data Augmentation](https://github.com/Shen-Lab/GraphCL/tree/master/semisupervised_TU#exploring-the-role-of-data-augmentation-in-graphcl)

* Semi-supervised learning [[TU Datasets]](https://github.com/Shen-Lab/GraphCL/tree/master/semisupervised_TU#graphcl-with-sampled-augmentations) [[MNIST and CIFAR10]](https://github.com/Shen-Lab/GraphCL/tree/master/semisupervised_MNIST_CIFAR10)

* Unsupervised representation learning [[TU Datasets]](https://github.com/Shen-Lab/GraphCL/tree/master/unsupervised_TU) [[Cora and Citeseer]](https://github.com/Shen-Lab/GraphCL/tree/master/unsupervised_Cora_Citeseer)

* Transfer learning [[MoleculeNet and PPI]](https://github.com/Shen-Lab/GraphCL/tree/master/transferLearning_MoleculeNet_PPI)

* Adversarial robustness [[Component Graphs]](https://github.com/Shen-Lab/GraphCL/tree/master/adversarialRobustness_Component)

## Citation

If you use this code for you research, please cite our paper.

```
@inproceedings{You2020GraphCL,
 author = {You, Yuning and Chen, Tianlong and Sui, Yongduo and Chen, Ting and Wang, Zhangyang and Shen, Yang},
 booktitle = {Advances in Neural Information Processing Systems},
 editor = {H. Larochelle and M. Ranzato and R. Hadsell and M. F. Balcan and H. Lin},
 pages = {5812--5823},
 publisher = {Curran Associates, Inc.},
 title = {Graph Contrastive Learning with Augmentations},
 url = {https://proceedings.neurips.cc/paper/2020/file/3fe230348e9a12c13120749e3f9fa4cd-Paper.pdf},
 volume = {33},
 year = {2020}
}
```
