# HW3

## Enviroments
```bash
pip install -r requirements.txt
```

## Reproduce my result
#### run below commands in submission folder(r12922192)
```bash
bash download.sh
bash run.sh \
    /path/to/Taiwan-Llama \
    /path/to/adapter_checkpoint/under/your/folder \
    /path/to/input \
    /path/to/output_file
```

## Training
### It will save models for steps for every 100 steps in output folder, and I finetune the pretrained model Taiwan-LLaMa for 5 epochs, use checkpoint-1200 model in my results.
#### run below commands in submission folder(r12922192)
```bash
#train
bash train.sh \
    /path/to/Taiwan-Llama \
    /path/to/train.json \
    /path/to/public_test.json \
    /path/to/output_dir
```

## (Bonus) Training on FlagAlpha/Llama2-Chinese-7b-Chat  
### It will save models for steps for every 100 steps in output folder, and I finetune the pretrained model FlagAlpha/Llama2-Chinese-7b-Chat for 5 epochs, checkpoint-1200 model is the best.
#### run below commands in submission folder(r12922192)
```bash
#train
bash train.sh \
    FlagAlpha/Llama2-Chinese-7b-Chat \
    /path/to/train.json \
    /path/to/public_test.json \
    /path/to/output_dir
```


## plot 
### plot loss 
#### run below commands in submission folder(r12922192)
```bash
python plot_loss.py --record_file /path/to/training/output/last/checkpoint/trainer_state.json
```