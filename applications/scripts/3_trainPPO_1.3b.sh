#!/bin/bash
set -x

OUTPUT=$1
ACTOR_ZERO_STAGE=$2
CRITIC_ZERO_STAGE=$3
if [ "$OUTPUT" == "" ]; then
    OUTPUT=./output
fi
if [ "$ACTOR_ZERO_STAGE" == "" ]; then
    ACTOR_ZERO_STAGE=0
fi
if [ "$CRITIC_ZERO_STAGE" == "" ]; then
    CRITIC_ZERO_STAGE=0
fi
mkdir -p $OUTPUT $OUTPUT/3_model_PPO $OUTPUT/logs

deepspeed --num_gpus 1 applications/train/3_trainPPO.py \
    --data_path 'datafile/Dahoas/rm-static' \
    --data_split 2,4,4 \
    --actor_model_name_or_path 'output/1_model_SFT' \
    --critic_model_name_or_path 'output/2_model_RM' \
    --num_padding_at_beginning 1 \
    --per_device_train_batch_size 16 \
    --per_device_mini_train_batch_size 16 \
    --generation_batch_numbers 1 \
    --ppo_epochs 1 \
    --max_answer_seq_len 256 \
    --max_prompt_seq_len 256 \
    --actor_learning_rate 9.65e-6 \
    --critic_learning_rate 5e-6 \
    --actor_weight_decay 0. \
    --critic_weight_decay 0. \
    --num_train_epochs 1 \
    --lr_scheduler_type cosine \
    --gradient_accumulation_steps 1 \
    --num_warmup_steps 100 \
    --seed 1234 \
    --enable_hybrid_engine \
    --actor_zero_stage 2 \
    --critic_zero_stage 2 \
    --enable_ema \
    --deepspeed \
    --output_dir $OUTPUT/3_model_PPO \
     2>&1 | tee $OUTPUT/logs/3_training_PPO.log

