# HW2

## Enviroments
```bash
pip install -r requirements.txt
```

## Reproduce my result
### (Public: 0.8065, Private: 0.7859) 
#### run below commands in submission folder(r12922192)
```bash
bash download.sh
bash ./run.sh /path/to/context.json /path/to/test.json /path/to/pred/prediction.csv
```

## Paragraph Selection
### Training
#### It will save models for all epochs, and I train the pretrained model chinese-macbert-large, use epoch 0 model in my results.
```bash
accelerate launch train/run_swag_no_trainer.py \
    --model_name_or_path <model_name_or_path> \
    --train_file <train_file> \
    --validation_file <valid_file> \
    --context_file <context_file> \
    --max_seq_length 512 \
    --per_device_train_batch_size 2 \
    --gradient_accumulation_steps 4 \
    --num_train_epochs 2\
    --learning_rate 1e-5  \
    --output_dir <output_dir>
```


## Question Answering
### Training
#### It will save models for all epochs, and I train the pretrained model chinese-macbert-large, use epoch 5 model in my results.
```bash
accelerate launch train/run_qa_no_trainer.py \
    --model_name_or_path <model_name_or_path> \
    --train_file <train_file> \
    --validation_file <valid_file> \
    --context_file <context_file> \
    --max_seq_length 512 \
    --per_device_train_batch_size 2 \
    --gradient_accumulation_steps 4 \
    --num_train_epochs 10\
    --learning_rate 1e-5  \
    --output_dir <output_dir>
```

## plot 
#### After traning both selection will save training information in record.json for plot and use this command get image in current folder.
```bash
python plot.py --record_file <record_file>
```