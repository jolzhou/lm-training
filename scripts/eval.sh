#!/bin/bash

. ~/.bashrc
conda activate lm-training-env/
cd lm-training

test_dataset=$1
subset=$2
model_run=$3
# model_name=${${model_run}//\//\_}
checkpoint=$4

python src/eval/eval_model.py \
    --tokenizer_path "models/tokenizer/raspl.json" \
    --model_path "outputs/${model_run}/checkpoints/${checkpoint}" \
    --data_path data/${test_dataset} \
    --subset $subset \
    --out_path preds/greedy-${model_run}-${checkpoint}-${test_dataset} \

# python src/eval/eval_model.py \
#     --tokenizer_path "archive-no-eos/models-archive-no-eos/tokenizer/raspl.json" \
#     --model_path "archive-no-eos/outputs-archive-no-eos/${model_run}/checkpoints/${checkpoint}" \
#     --data_path "archive-no-eos/data-archive-no-eos/${test_dataset}" \
#     --subset $subset \
#     --out_path "archive-no-eos/preds-archive-no-eos/greedy-${model_run}-${checkpoint}-${test_dataset}" \

