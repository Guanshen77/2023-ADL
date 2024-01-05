accelerate launch /tmp2/kevinchiu/ADL/HW1/inference/part2_predict.py \
    --model_name_or_path /tmp2/kevinchiu/ADL/HW1/final_model/part2/1020_macbert_large/epoch_5 \
    --validation_file /tmp2/kevinchiu/ADL/HW1/inference/part1/finalpart1_output.json \
    --context_file /tmp2/kevinchiu/ADL/HW1/ntuadl2023hw1/context.json \
    --max_seq_length 512 \
    --output_dir "/tmp2/kevinchiu/ADL/HW1/inference/part2_output/final"