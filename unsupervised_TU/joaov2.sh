#!/bin/bash -ex

for seed in 0 1 2 3 4 
do
  CUDA_VISIBLE_DEVICES=$1 python gsimclr_minmax_condition.py --DS $2 --lr 0.001 --local --num-gc-layers 3 --aug minmax --gamma $3 --seed $seed
done

