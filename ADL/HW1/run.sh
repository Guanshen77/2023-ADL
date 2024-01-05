context_path=$1
test_path=$2
output_csv=$3
accelerate launch ./inference/part1_predict.py \
    --model_name_or_path ./HW1_model_data/part1/1020_macbert_large/epoch_0 \
    --validation_file $test_path\
    --context_file $context_path \
    --max_seq_length 512 \
    --output_dir ./inference/part1/final/

accelerate launch ./inference/part2_predict.py \
    --model_name_or_path ./HW1_model_data/part2/1020_macbert_large/epoch_5 \
    --validation_file ./inference/part1/final/part1_output.json \
    --context_file $context_path \
    --max_seq_length 512 \
    --csv_path $output_csv

