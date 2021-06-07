### JOAO Pre-Training: ###

```
cd ./pretrain
python main.py --dataset NCI1 --epochs 100 --lr 0.001 --gamma_joao 0.1  --suffix 0
python main.py --dataset NCI1 --epochs 100 --lr 0.001 --gamma_joao 0.1  --suffix 1
python main.py --dataset NCI1 --epochs 100 --lr 0.001 --gamma_joao 0.1  --suffix 2
python main.py --dataset NCI1 --epochs 100 --lr 0.001 --gamma_joao 0.1  --suffix 3
python main.py --dataset NCI1 --epochs 100 --lr 0.001 --gamma_joao 0.1  --suffix 4
```

### JOAO Finetuning: ###

```
cd ./finetune
python main.py --dataset NCI1 --pretrain_epoch 100 --pretrain_lr 0.001 --pretrain_gamma 0.1 --suffix 0 --n_splits 100
python main.py --dataset NCI1 --pretrain_epoch 100 --pretrain_lr 0.001 --pretrain_gamma 0.1 --suffix 1 --n_splits 100
python main.py --dataset NCI1 --pretrain_epoch 100 --pretrain_lr 0.001 --pretrain_gamma 0.1 --suffix 2 --n_splits 100
python main.py --dataset NCI1 --pretrain_epoch 100 --pretrain_lr 0.001 --pretrain_gamma 0.1 --suffix 3 --n_splits 100
python main.py --dataset NCI1 --pretrain_epoch 100 --pretrain_lr 0.001 --pretrain_gamma 0.1 --suffix 4 --n_splits 100
```

Five suffixes stand for five runs (with mean & std reported)

```lr``` should be tuned from {0.01, 0.001, 0.0001}, ```gamma_joao``` from {0.01, 0.1, 1} in pre-training, and ```pretrain_epoch``` in finetuning (this means the epoch checkpoint loaded from pre-trained model) from {20, 40, 60, 80, 100}.


### JOAOv2 Pre-Training: ###

```
cd ./pretrain_joaov2
python main.py --dataset NCI1 --epochs 200 --lr 0.001 --gamma_joao 0.1  --suffix 0
python main.py --dataset NCI1 --epochs 200 --lr 0.001 --gamma_joao 0.1  --suffix 1
python main.py --dataset NCI1 --epochs 200 --lr 0.001 --gamma_joao 0.1  --suffix 2
python main.py --dataset NCI1 --epochs 200 --lr 0.001 --gamma_joao 0.1  --suffix 3
python main.py --dataset NCI1 --epochs 200 --lr 0.001 --gamma_joao 0.1  --suffix 4
```

### JOAOv2 Finetuning: ###

```
cd ./finetune_joaov2
python main.py --dataset NCI1 --pretrain_epoch 100 --pretrain_lr 0.001 --pretrain_gamma 0.1 --suffix 0 --n_splits 100
python main.py --dataset NCI1 --pretrain_epoch 100 --pretrain_lr 0.001 --pretrain_gamma 0.1 --suffix 1 --n_splits 100
python main.py --dataset NCI1 --pretrain_epoch 100 --pretrain_lr 0.001 --pretrain_gamma 0.1 --suffix 2 --n_splits 100
python main.py --dataset NCI1 --pretrain_epoch 100 --pretrain_lr 0.001 --pretrain_gamma 0.1 --suffix 3 --n_splits 100
python main.py --dataset NCI1 --pretrain_epoch 100 --pretrain_lr 0.001 --pretrain_gamma 0.1 --suffix 4 --n_splits 100
```

Five suffixes stand for five runs (with mean & std reported)

```lr``` should be tuned from {0.01, 0.001, 0.0001}, ```gamma_joao``` from {0.01, 0.1, 1} in pre-training, and ```pretrain_epoch``` in finetuning (this means the epoch checkpoint loaded from pre-trained model) from {20, 40, 60, 80, 100, 120, 140, 160, 180, 200} since multiple projection heads are trained.

## Acknowledgements

The backbone implementation is reference to https://github.com/chentingpc/gfn.
