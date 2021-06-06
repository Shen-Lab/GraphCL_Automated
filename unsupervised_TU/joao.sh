#!/bin/bash -ex

for seed in 0 1 2 3 4 
do
  CUDA_VISIBLE_DEVICES=$1 python gsimclr_minmax.py --DS $2 --lr 0.01 --local --num-gc-layers 3 --aug minmax --gamma $3 --seed $seed
done

