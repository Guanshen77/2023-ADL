train_file_path=$1
valid_file_path=$2
output_dir_path=$3

python run_summarization_no_trainer.py \
    --model_name_or_path google/mt5-small \
    --train_file $train_file_path \
    --validation_file $valid_file_path \
    --per_device_train_batch_size 4 \
    --gradient_accumulation_steps 4 \
    --max_source_length 256 \
    --max_target_length 64 \
    --learning_rate 1e-4 \
    --weight_decay 5e-5 \
    --num_beams 7 \
    --num_train_epochs 50 \
    --source_prefix "summarize: " \
    --output_dir $output_dir_path
