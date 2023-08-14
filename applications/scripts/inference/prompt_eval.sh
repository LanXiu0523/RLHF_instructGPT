#!/bin/bash
set -x

OUTPUT=$1
ZERO_STAGE=$2
if [ "$OUTPUT" == "" ]; then
    OUTPUT=./output
fi


export CUDA_VISIBLE_DEVICES=0

python3 applications/inference/prompt_eval.py \
    --model_name_or_path_pretrain 'datafile/facebook/opt-1.3b' \
    --model_name_or_path_sft 'output/1_model_SFT' \
    --model_name_or_path_ppo 'output/3_model_PPO/actor' \
    --max_new_tokens 200 \
    2>&1 | tee $OUTPUT/logs/PPOmodel_eval.log 
