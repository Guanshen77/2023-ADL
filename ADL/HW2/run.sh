valid_file_path=$1
output_file_path=$2

python jsonltojson.py --jsonl_file $valid_file_path --json_file ./public.json
python inference.py \
    --model_name_or_path ./HW2_model/ \
    --validation_file ./public.json \
    --max_source_length 256 \
    --max_target_length 64 \
    --do_beams True\
    --num_beams  7 \
    --per_device_eval_batch_size 4 \
    --source_prefix "summarize: " \
    --output_dir ./output/ \
    --output_file $output_file_path