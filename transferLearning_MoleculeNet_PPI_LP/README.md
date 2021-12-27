### LP-InfoMin/InfoBN/Info(Min+BN) Pre-Training: ###

```
cd ./bio
python pretrain_generative_infomin.py
python pretrain_generative_infobn.py
python pretrain_generative_infominbn.py
cd ./chem
python pretrain_generative_infomin.py
python pretrain_generative_infobn.py
python pretrain_generative_infominbn.py
```


### LP-InfoMin/InfoBN/Info(Min+BN) Finetuning: ###

```
cd ./bio
./finetune.sh ${MODEL_FILE} 1e-3 ${RESULT_FILE}
cd ./chem
./finetune.py bbbp ${MODEL_FILE} 1e-3 ${RESULT_FILE}
./finetune.py tox21 ${MODEL_FILE} 1e-3 ${RESULT_FILE}
./finetune.py toxcast ${MODEL_FILE} 1e-3 ${RESULT_FILE}
./finetune.py sider ${MODEL_FILE} 1e-3 ${RESULT_FILE}
./finetune.py clintox ${MODEL_FILE} 1e-3 ${RESULT_FILE}
./finetune.py muv ${MODEL_FILE} 1e-3 ${RESULT_FILE}
./finetune.py hiv ${MODEL_FILE} 1e-3 ${RESULT_FILE}
./finetune.py bace ${MODEL_FILE} 1e-3 ${RESULT_FILE}
```

```MODEL_FILE``` is the saved pre-training weight, and ```RESULT_FILE``` is the file to store the results.




## Acknowledgements

The backbone implementation is reference to https://github.com/snap-stanford/pretrain-gnns.
