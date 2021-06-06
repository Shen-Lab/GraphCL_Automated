#### GIN fine-tuning
split=species

model_file=$1
lr=$2
resultFile_name=$3

### for GIN
for runseed in 0 1 2 3 4 5 6 7 8 9
do
python finetune.py --model_file $model_file --split $split --epochs 50 --device 0 --runseed $runseed --gnn_type gin --lr $lr --resultFile_name $resultFile_name
done
