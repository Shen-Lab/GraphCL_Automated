### JOAO Pre-Training: ###

```
cd ./bio
python pretrain_joao.py --gamma 0.01
cd ./chem
python pretrain_joao.py --gamma 0.01
```


### JOAO Finetuning: ###

```
cd ./bio
./finetune.sh ./weights/joao_${gamma}_100.pth 1e-3 ${RESULT_FILE}
cd ./chem
./finetune.py bbbp ./weights/joao_${gamma}_100.pth 1e-3 ${RESULT_FILE}
./finetune.py tox21 ./weights/joao_${gamma}_100.pth 1e-3 ${RESULT_FILE}
./finetune.py toxcast ./weights/joao_${gamma}_100.pth 1e-3 ${RESULT_FILE}
./finetune.py sider ./weights/joao_${gamma}_100.pth 1e-3 ${RESULT_FILE}
./finetune.py clintox ./weights/joao_${gamma}_100.pth 1e-3 ${RESULT_FILE}
./finetune.py muv ./weights/joao_${gamma}_100.pth 1e-3 ${RESULT_FILE}
./finetune.py hiv ./weights/joao_${gamma}_100.pth 1e-3 ${RESULT_FILE}
./finetune.py bace ./weights/joao_${gamma}_100.pth 1e-3 ${RESULT_FILE}
```

```gamma``` is tuned from {0.01, 0.1, 1}. ```RESULT_FILE``` is the file to store the results.


### JOAOv2 Pre-Training: ###

```
cd ./bio
python pretrain_joaov2.py --gamma 0.01
cd ./chem
python pretrain_joaov2.py --gamma 0.01
```


### JOAOv2 Finetuning: ###

```
cd ./bio
./finetune.sh ./weights/joaov2_${gamma}_100.pth  1e-3 ${RESULT_FILE}
cd ./chem
./finetune.py bbbp ./weights/joaov2_${gamma}_100.pth 1e-3 ${RESULT_FILE}
./finetune.py tox21 ./weights/joaov2_${gamma}_100.pth 1e-3 ${RESULT_FILE}
./finetune.py toxcast ./weights/joaov2_${gamma}_100.pth 1e-3 ${RESULT_FILE}
./finetune.py sider ./weights/joaov2_${gamma}_100.pth 1e-3 ${RESULT_FILE}
./finetune.py clintox ./weights/joaov2_${gamma}_100.pth 1e-3 ${RESULT_FILE}
./finetune.py muv ./weights/joaov2_${gamma}_100.pth 1e-3 ${RESULT_FILE}
./finetune.py hiv ./weights/joaov2_${gamma}_100.pth 1e-3 ${RESULT_FILE}
./finetune.py bace ./weights/joaov2_${gamma}_100.pth 1e-3 ${RESULT_FILE}
```

```gamma``` is tuned from {0.01, 0.1, 1}. ```RESULT_FILE``` is the file to store the results.


## Acknowledgements

The backbone implementation is reference to https://github.com/snap-stanford/pretrain-gnns.
