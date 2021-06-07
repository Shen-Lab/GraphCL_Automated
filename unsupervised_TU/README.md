### JOAO Pre-Training & Finetuning: ###

```
./joao.sh NCI1 ${gamma}
```

```gamma``` is tuned from {0.01, 0.1, 1}.


### JOAOv2 Pre-Training & Finetuning: ###

```
./joaov2.sh NCI1 ${gamma}
```

```gamma``` is tuned from {0.01, 0.1, 1}. JOAOv2 is trained for 40 epochs since multiple projection heads are trained.
