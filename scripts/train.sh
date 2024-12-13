#!/bin/bash

. ~/.bashrc
conda activate lm-training-env/
cd lm-training

dataset=$1
program_name=$2

export NEPTUNE_API_TOKEN="eyJhcGlfYWRkcmVzcyI6Imh0dHBzOi8vYXBwLm5lcHR1bmUuYWkiLCJhcGlfdXJsIjoiaHR0cHM6Ly9hcHAubmVwdHVuZS5haSIsImFwaV9rZXkiOiIxZGUwNDViNC00NzMzLTQwMGEtOGRjYi0zY2IyOTJmOWVkMGIifQ=="
export NEPTUNE_PROJECT="jolie/learning-like-transformers"

python src/train_lm.py \
    --config-name train-prefix-transformer \
    'data.base_dir=data/'${dataset} \
    'training_args.output_dir=outputs/'${program_name}'/checkpoints' \
    'tokenizer.tokenizer_file=models/tokenizer/raspl.json' \
    'tokenizer.sep_token="-3"' \
    'tokenizer.eos_token="-4"' \
