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


deepspeed  --num_gpus 1 applications/train/3_trainPPO.py \
    --data_path 'datafile/Dahoas/rm-static' \
    --data_split 2,4,4 \
    --actor_model_name_or_path 'output/1_model_SFT' \
    --critic_model_name_or_path 'output/2_model_RM' \
    --actor_zero_stage $ACTOR_ZERO_STAGE \
    --critic_zero_stage $CRITIC_ZERO_STAGE \
    --num_padding_at_beginning 1 \
    --ppo_epochs 1 \
    --num_train_epochs 1 \
    --gradient_accumulation_steps 2 \
    --actor_lora_dim 128 \
    --seed 1234 \
    --actor_gradient_checkpointing \
    --disable_actor_dropout \
    --deepspeed \
    --output_dir $OUTPUT/3_model_PPO \
    2>&1 | tee $OUTPUT/logs/3_training_PPO.log


if [ ${PIPESTATUS[0]} -ne 0 ]; then
  exit 1
fi
   
