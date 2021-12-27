### LP-InfoMin/InfoBN/Info(Min+BN) Pre-Training: ###

```
cd ./code
python main_pretrain_generative_infomin.py
python main_pretrain_generative_infobn.py
python main_pretrain_generative_infominbn.py
cd ./ppa
python main_pretrain_generative_infomin.py
python main_pretrain_generative_infobn.py
python main_pretrain_generative_infominbn.py
```


### LP-InfoMin/InfoBN/Info(Min+BN) Finetuning: ###

```
cd ./code
./finetune.sh ${N_SPLIT} ${RESULT_FILE} ${MODEL_PATH}
cd ./ppa
./finetune.sh ${N_SPLIT} ${RESULT_FILE} ${MODEL_PATH}
```

```N_SPLIT``` can be 100 or 10 for 1% or 10% label rate, ```RESULT_FILE``` is the file to store the results, and ```MODEL_PATH``` is the path to save the pre-trained model.


## Acknowledgements

The backbone implementation is reference to https://github.com/snap-stanford/ogb.
