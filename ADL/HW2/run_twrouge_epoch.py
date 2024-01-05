import subprocess
from tqdm import tqdm
import json
import argparse

def parse_args():
    parser = argparse.ArgumentParser(description="get command")
    parser.add_argument(
        "--epoch",
        type=int,
        default=None,
        help="input",
    )
    parser.add_argument(
        "--val_jsonl_file",
        type=str,
        default=None,
        help="input",
    )
    parser.add_argument(
        "--eval_py_file",
        type=str,
        default=None,
        help="output",
    )
    parser.add_argument(
        "--output_json_file",
        type=str,
        default=None,
        help="output",
    )

    args = parser.parse_args()
    return args

def main():
    args = parse_args()

    output = []   
    progress = tqdm(total=int(args.epoch), position=0, leave=True)
    for i in range(args.epoch):
        progress.update(1)
        # 定義要執行的指令
        command = "python "
        command += args.eval_py_file
        command += " -r " 
        command += args.val_jsonl_file
        command += " -s "
        command += "./output{}.jsonl".format(i)  
        # 使用subprocess執行指令
        result = subprocess.run(command, shell=True, capture_output=True, text=True)

        # 列印指令的輸出
        record= {}
        data = json.loads(result.stdout)
        record["epoch"] = i
        record["rouge_tw"] = data
        output.append(record)
    progress.close()


    output_path = args.output_json_file
    with open(output_path, 'w') as json_file:
        json.dump(output, json_file, indent=4)

if __name__ == "__main__":
    main()


