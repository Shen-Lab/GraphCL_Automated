### LP-InfoMin/InfoBN/Info(Min+BN) Pre-Training: ###

```
cd ./pretrain
python main.py --dataset COLLAB --epochs 100 --lr 0.001 --principle infomin --suffix 0
python main.py --dataset COLLAB --epochs 100 --lr 0.001 --principle infomin --suffix 1
python main.py --dataset COLLAB --epochs 100 --lr 0.001 --principle infomin --suffix 2
python main.py --dataset COLLAB --epochs 100 --lr 0.001 --principle infomin --suffix 3
python main.py --dataset COLLAB --epochs 100 --lr 0.001 --principle infomin --suffix 4
python main.py --dataset COLLAB --epochs 100 --lr 0.001 --principle infobn --suffix 0
python main.py --dataset COLLAB --epochs 100 --lr 0.001 --principle infobn --suffix 1
python main.py --dataset COLLAB --epochs 100 --lr 0.001 --principle infobn --suffix 2
python main.py --dataset COLLAB --epochs 100 --lr 0.001 --principle infobn --suffix 3
python main.py --dataset COLLAB --epochs 100 --lr 0.001 --principle infobn --suffix 4
python main.py --dataset COLLAB --epochs 100 --lr 0.001 --principle infominbn --suffix 0
python main.py --dataset COLLAB --epochs 100 --lr 0.001 --principle infominbn --suffix 1
python main.py --dataset COLLAB --epochs 100 --lr 0.001 --principle infominbn --suffix 2
python main.py --dataset COLLAB --epochs 100 --lr 0.001 --principle infominbn --suffix 3
python main.py --dataset COLLAB --epochs 100 --lr 0.001 --principle infominbn --suffix 4
```

### LP-InfoMin/InfoBN/Info(Min+BN) Finetuning: ###

```
cd ./finetune
python main.py --dataset NCI1 --pretrain_epoch 100 --pretrain_lr 0.001 ----model_path $PATH_OF_MODEL_WEIGHT --result_path $PATH_TO_SAVE_RESULT --suffix 0 --n_splits 10
python main.py --dataset NCI1 --pretrain_epoch 100 --pretrain_lr 0.001 ----model_path $PATH_OF_MODEL_WEIGHT --result_path $PATH_TO_SAVE_RESULT --suffix 1 --n_splits 10
python main.py --dataset NCI1 --pretrain_epoch 100 --pretrain_lr 0.001 ----model_path $PATH_OF_MODEL_WEIGHT --result_path $PATH_TO_SAVE_RESULT --suffix 2 --n_splits 10
python main.py --dataset NCI1 --pretrain_epoch 100 --pretrain_lr 0.001 ----model_path $PATH_OF_MODEL_WEIGHT --result_path $PATH_TO_SAVE_RESULT --suffix 3 --n_splits 10
python main.py --dataset NCI1 --pretrain_epoch 100 --pretrain_lr 0.001 ----model_path $PATH_OF_MODEL_WEIGHT --result_path $PATH_TO_SAVE_RESULT --suffix 4 --n_splits 10
```

Five suffixes stand for five runs (with mean & std reported)

```lr``` should be tuned from {0.01, 0.001, 0.0001} and ```pretrain_epoch``` in finetuning (this means the epoch checkpoint loaded from pre-trained model) from {20, 40, 60, 80, 100}.

## Acknowledgements

The backbone implementation is reference to https://github.com/chentingpc/gfn.
