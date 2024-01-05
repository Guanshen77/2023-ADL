from peft import PeftModel
import torch
import numpy as np
from tqdm import tqdm
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer
import json
from utils import get_prompt, get_bnb_config
import argparse



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--base_model_path",
        type=str,
        default="",
        help="Path to the checkpoint of Taiwan-LLM-7B-v2.0-chat. If not set, this script will use "
        "the checkpoint from Huggingface (revision = 5073b2bbc1aa5519acdc865e99832857ef47f7c9)."
    )
    parser.add_argument(
        "--peft_path",
        type=str,
        default=None,
        help="Path to the saved PEFT checkpoint."
    )
    parser.add_argument(
        "--test_data_path",
        type=str,
        default="",
        required=True,
        help="Path to test data."
    )
    parser.add_argument(
        "--output_path",
        type=str,
        default="",
        required=True,
        help="Path to test data."
    )
    args = parser.parse_args()

    # Load model
    bnb_config = get_bnb_config()

    if args.base_model_path:
        model = AutoModelForCausalLM.from_pretrained(
            args.base_model_path,
            torch_dtype=torch.bfloat16,
            quantization_config=bnb_config
        )
        tokenizer = AutoTokenizer.from_pretrained(args.base_model_path)
    else:
        model_name = "yentinglin/Taiwan-LLM-7B-v2.0-chat"
        revision = "5073b2bbc1aa5519acdc865e99832857ef47f7c9"
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            revision=revision,
            torch_dtype=torch.bfloat16,
            quantization_config=bnb_config
        )
        tokenizer = AutoTokenizer.from_pretrained(
            model_name,
            revision=revision,
        )
    tokenizer.padding_side='left'
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id

    # Load LoRA
    if args.peft_path:
        model = PeftModel.from_pretrained(model, args.peft_path)

    #Load dataset
    data_files = {}
    if args.test_data_path is not None:
        data_files["validation"] = args.test_data_path
    extension = args.test_data_path.split(".")[-1]
    raw_datasets = load_dataset(extension, data_files=data_files)
    eval_dataset = raw_datasets["validation"]

    ###################################

    gen_kwargs = {
        'do_sample': False,
        'num_beams': 1,
        'num_beam_groups': 1,
        'penalty_alpha': None,
        'use_cache': True,

        # Hyperparameters for logit manipulation
        'max_length': 1024,
        'temperature': 1.0,
        'top_k': 50,
        'top_p': 1.0,
        'typical_p': 1.0,
        'diversity_penalty': 0.0,
        'repetition_penalty': 1.0,
        'length_penalty': 1.0,
        'no_repeat_ngram_size': 0,
    }

    model.eval()

    
    predictions = []
    progress = tqdm(total=len(eval_dataset), position=0, leave=True)
    instructions = [get_prompt(x["instruction"]) for x in eval_dataset]
    tokenized_instructions = tokenizer(instructions, add_special_tokens=False)
    for i in range(len(eval_dataset)):
        instruction_input_ids = [tokenizer.bos_token_id] + tokenized_instructions["input_ids"][i]
        tokenized_instructions["input_ids"][i] = instruction_input_ids 
        tokenized_instructions["attention_mask"][i] = [1] * len(tokenized_instructions["input_ids"][i])
        tokenized_instructions["input_ids"][i] = torch.tensor(tokenized_instructions["input_ids"][i][:1024])
        tokenized_instructions["attention_mask"][i] = torch.tensor(tokenized_instructions["attention_mask"][i][:1024])


    for i in range(0, len(eval_dataset)):
        progress.update(1)
    
        input_ids = tokenized_instructions["input_ids"][i].unsqueeze(0)
        attn_mask = tokenized_instructions["attention_mask"][i].unsqueeze(0)

        with torch.no_grad():
            outputs = model.generate(input_ids=input_ids.to("cuda"), attention_mask=attn_mask.to("cuda"), **gen_kwargs,)
            decoded_output = tokenizer.batch_decode(outputs, skip_special_tokens=True)[0]
            print(decoded_output)
            assistant_index = decoded_output.find("ASSISTANT:")
            if assistant_index != -1:
                # 刪除 "ASSISTANT："及其之前的部分，保留助理說的內容
                decoded_output= decoded_output[assistant_index + len("ASSISTANT:"):].strip()
            

            output_text = decoded_output.strip().replace("�", "")
            predictions.append(output_text)

    progress.close()
    

    print("generating json")
    output = []

    for i in range(len(predictions)):
        output_dict = {}
        output_dict["id"] = eval_dataset[i]["id"]
        output_dict["output"] = predictions[i]
        output.append(output_dict)

    with open(args.output_path, 'w', encoding='utf-8') as f:
        json.dump(output, f, ensure_ascii=False, indent=4)

        