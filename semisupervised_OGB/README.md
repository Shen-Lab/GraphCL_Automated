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
./run.sh ${N_SPLIT} ${RESULT_FILE} ./weights/joao_0.01_30.pt
cd ./ppa
./run.sh ${N_SPLIT} ${RESULT_FILE} ./weights/joao_0.01_100.pt
```

```gamma``` is tuned from {0.01, 0.1, 1}.


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
./run_joaov2.sh ${N_SPLIT} ${RESULT_FILE} ./weights/joaov2_0.01_30.pt
cd ./ppa
./run_joaov2.sh ${N_SPLIT} ${RESULT_FILE} ./weights/joaov2_0.01_100.pt
```

```gamma``` is tuned from {0.01, 0.1, 1}.


## Acknowledgements

The backbone implementation is reference to https://github.com/snap-stanford/ogb.
