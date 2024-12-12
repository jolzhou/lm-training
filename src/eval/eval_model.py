
"""
Evaluate a model on given test dataset.
"""

import os
import json
import pickle
import argparse
from tqdm import tqdm

import numpy as np
import torch
from datasets import Dataset

from transformers import PreTrainedTokenizerFast, AutoModelForCausalLM

import sys

def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--tokenizer_path", "-t", default="models/ft/XTREME-udpos-English-61360")
    parser.add_argument("--model_path", "-m", default="models/ft/XTREME-udpos-English-61360")
    parser.add_argument("--data_path", "-gp", default=None)
    parser.add_argument("--subset", "-s", default=None)
    parser.add_argument("--out_path", "-o", default=None)
    parser.add_argument("--max_num_examples", "-n", type=int, default=1000000)
    parser.add_argument("--device", "-d", default=("cuda" if torch.cuda.is_available() else "cpu"))
    parser.add_argument("--batch_size", "-b", type=int, default=8)
    parser.add_argument("--step-size", "-ss", type=float, default=None)
    parser.add_argument("--original", "-og", action="store_true")

    args = parser.parse_args()
    
    if args.step_size is not None and args.out_path is not None:
        raise ValueError(
            "Only last output is saved to file when doing a step-size sweep. " + 
            "Please remove --out-path or set --step-size to None."
        )

    return args

def main():
    args = parse_args()
    
    SEP_TOK = str(-3)

    tokenizer = PreTrainedTokenizerFast(tokenizer_file=args.tokenizer_path)
    tokenizer.eos_token="-4"
    tokenizer.eos_token_id=3
    model = AutoModelForCausalLM.from_pretrained(args.model_path).to(args.device)
    
    dataset_dict = {
        "input": [],
        "target": []
    }
    with open(os.path.join(args.data_path, f"{args.subset}.txt")) as f:
        for line in f.readlines():
            parts = line.strip().split(SEP_TOK)
            tokens = parts[0].strip()
            output = parts[1].strip().split(tokenizer.eos_token)[0].strip()
            
            dataset_dict["input"].append(tokens)
            dataset_dict["target"].append(output)
    
    dataset = Dataset.from_dict(dataset_dict)
    num_examples = min(len(dataset), args.max_num_examples)
    dataset = dataset.select(range(num_examples))

    preds = []
    log_probs = []
    max_seq_length = 20
    for i in tqdm(range(0, num_examples, args.batch_size)):
        batch = dataset[i:i + args.batch_size]
        batch_inputs = tokenizer(
            batch["input"], 
            padding='max_length', 
            max_length=max_seq_length, 
            truncation=False, 
            return_tensors="pt"
        ).to(args.device)
        
        with torch.no_grad():
            input_ids = batch_inputs.input_ids
            outputs = model.generate(
                input_ids, 
                max_new_tokens=max_seq_length,
                output_logits=True,
                return_dict_in_generate=True,
            )
        
        # Take completion only
        batch_logits = torch.stack(outputs.logits)  # (seq_length, batch_size, vocab_size)
        batch_log_probs = torch.nn.functional.softmax(batch_logits, dim=-1)
        batch_log_probs = torch.moveaxis(batch_log_probs, (0, 1, 2), (1, 0, 2))
                
        batch_preds = tokenizer.batch_decode(outputs.sequences[:, max_seq_length:], skip_special_tokens=True)
        
        preds.extend(batch_preds)
        log_probs.extend(batch_log_probs.tolist())

    accuracy = 0
    if args.out_path is not None:
        if not os.path.exists(args.out_path):
            os.makedirs(args.out_path)
            
        out_data = []
        for i in range(len(preds)):
            e = dataset[i]
            e["pred"] = preds[i]
            accuracy += 1 if e["pred"] == e["target"] else 0
            e["log_probs"] = [str([f"{x:.3f}" for x in log_prob]) for log_prob in log_probs[i]]
            out_data.append(e)
        with open(os.path.join(args.out_path, f"{args.subset}.txt"), "w+") as f:
            json.dump(out_data, f, indent=4)
    else:
        for i in range(len(preds)):
            accuracy += 1 if preds[i] == dataset[i]["target"] else 0
    accuracy /= len(preds)            
    
    print(f"Accuracy: {accuracy}")

if __name__ == '__main__':
    main()