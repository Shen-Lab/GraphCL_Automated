#### GIN fine-tuning
split=scaffold

dataset=$1
model_file=$2
lr=$3
resultFile_name=$4

for runseed in 0 1 2 3 4 5 6 7 8 9
do
python finetune.py --input_model_file $model_file --split $split --runseed $runseed --gnn_type gin --dataset $dataset --lr $lr --epochs 100 --resultFile_name $resultFile_name
done
