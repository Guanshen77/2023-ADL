accelerate launch /tmp2/kevinchiu/ADL/HW1/inference/part1_predict.py \
    --model_name_or_path /tmp2/kevinchiu/ADL/HW1/final_model/part1/1020_macbert_large/epoch_0 \
    --validation_file /tmp2/kevinchiu/ADL/HW1/ntuadl2023hw1/test.json \
    --context_file /tmp2/kevinchiu/ADL/HW1/ntuadl2023hw1/context.json \
    --max_seq_length 512 \
    --output_dir "/tmp2/kevinchiu/ADL/HW1/inference/part1/final"