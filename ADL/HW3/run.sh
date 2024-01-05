base_model_path=$1
peft_path=$2
test_data_path=$3
output_path=$4
python3 ./inference.py \
    --base_model_path $base_model_path \
    --peft_path $peft_path \
    --test_data_path $test_data_path \
    --output_path $output_path 
