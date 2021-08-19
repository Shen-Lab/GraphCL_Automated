#!/bin/bash -ex

for seed in 0 1 2 3 4 
do
  python joao.py --DS $1 --lr 0.01 --local --num-gc-layers 3 --aug minmax --gamma $2 --seed $seed
done

