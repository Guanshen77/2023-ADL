# HW2

## Enviroments
```bash
pip install -r requirements.txt
```

## Reproduce my result
### (public score f1*100, rouge-1: 26.838, rouge-2: 10.826, rouge-L: 23.975) 
#### run below commands in submission folder(r12922192)
```bash
bash download.sh
bash ./run.sh /path/to/input.jsonl /path/to/output.jsonl
```

## Training
### It will save models for all epochs, and I fintune the pretrained model google/mt5-small for 50 epochs, use epoch 40 model in my results.
#### trainng file and validation file need json format.
```bash
python jsonltojson.py --jsonl_file <jsonl_file> --json_file <json_file>
```

```bash
#train
bash train_no_trainer.sh <train_file> <valid_file> <output_dir>
```


## plot 
#### First install twrouge and get rouge score by form (epoch 0) to (epoch you want-1) in training output directory 
```bash
git clone https://github.com/moooooser999/ADL23-HW2.git
cd ADL23-HW2
pip install -e tw_rouge

cd <output_dir>
python <run_twrouge_epoch.py_file> --epoch <epoch_you_want> --val_jsonl_file <answer_file> --eval_py_file <tw_rouge_eval.py_file> --output_json_file <output_file>
```
#### Second plot
```bash
python <plot_twrouge.py_file> --record_file <output_file from last step> --output_dir <output_dir>
```