#!/bin/bash

for number in {1..10}
do
  python finetune.py --gnn gin --num_workers 8 --n_splits $1 --pretrain $2 --pretrain_weight $3
done
