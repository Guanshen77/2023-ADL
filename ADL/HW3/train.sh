model_path=$1
train_data_path=$2
test_data_path=$3
output_path=$4
python qlora.py \
    --model_name_or_path $model_path \
    --num_train_epochs 5 \
    --dataset $train_data_path \
    --dataset_eval $test_data_path \
    --output_dir $output_path 