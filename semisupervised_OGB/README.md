### JOAO Pre-Training: ###

```
cd ./code
python main_pretrain_graphcl_joao.py --gamma 0.01
cd ./ppa
python main_pretrain_graphcl_joao.py --gamma 0.01
```


### JOAO Finetuning: ###

```
cd ./code
./run.sh ${N_SPLIT} ${RESULT_FILE} ./weights/joao_${gamma}_30.pt
cd ./ppa
./run.sh ${N_SPLIT} ${RESULT_FILE} ./weights/joao_${gamma}_100.pt
```

```gamma``` is tuned from {0.01, 0.1, 1}. ```N_SPLIT``` can be 100 or 10 for 1% or 10% label rate, and ```RESULT_FILE``` is the file to store the results.


### JOAOv2 Pre-Training: ###

```
cd ./code
python main_pretrain_graphcl_joaov2.py --gamma 0.01
cd ./ppa
python main_pretrain_graphcl_joaov2.py --gamma 0.01
```


### JOAOv2 Finetuning: ###

```
cd ./code
./run_joaov2.sh ${N_SPLIT} ${RESULT_FILE} ./weights/joaov2_${gamma}_30.pt
cd ./ppa
./run_joaov2.sh ${N_SPLIT} ${RESULT_FILE} ./weights/joaov2_${gamma}_100.pt
```

```gamma``` is tuned from {0.01, 0.1, 1}. ```N_SPLIT``` can be 100 or 10 for 1% or 10% label rate, and ```RESULT_FILE``` is the file to store the results.


## Acknowledgements

The backbone implementation is reference to https://github.com/snap-stanford/ogb.
