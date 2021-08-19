#!/bin/bash -ex

for seed in 0 1 2 3 4 
do
  python joaov2.py --DS $1 --lr 0.001 --local --num-gc-layers 3 --aug minmax --gamma $2 --seed $seed
done
