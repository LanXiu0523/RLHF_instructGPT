#!/bin/bash
set -x

OUTPUT=$1
ACTOR_ZERO_STAGE=$2
CRITIC_ZERO_STAGE=$3
if [ "$OUTPUT" == "" ]; then
    OUTPUT=./output
fi
if [ "$ACTOR_ZERO_STAGE" == "" ]; then
    ACTOR_ZERO_STAGE=2
fi
if [ "$CRITIC_ZERO_STAGE" == "" ]; then
    CRITIC_ZERO_STAGE=2
fi
mkdir -p $OUTPUT $OUTPUT/3_model_PPO $OUTPUT/logs


Num_Padding_at_Beginning=1
Actor_Lr=9.65e-6
Critic_Lr=5e-6
HOSTFILE='/home/cluster.conf'

deepspeed --hostfile=$HOSTFILE applications/train/3_trainPPO.py \
    --data_path 'datafile/Dahoas/rm-static' \
    --data_split 2,4,4 \
    --actor_model_name_or_path 'output/1_model_SFT' \
    --critic_model_name_or_path 'output/2_model_RM' \
    --num_padding_at_beginning 1 \
    --per_device_train_batch_size 4 \
    --per_device_mini_train_batch_size 4 \
    --ppo_epochs 1 \
    --max_answer_seq_len 256 \
    --max_prompt_seq_len 256 \
    --actor_learning_rate ${Actor_Lr} \
    --critic_learning_rate ${Critic_Lr} \
    --actor_weight_decay 0. \
    --critic_weight_decay 0. \
    --num_train_epochs 1 \
    --lr_scheduler_type cosine \
    --gradient_accumulation_steps 1 \
    --disable_actor_dropout \
    --num_warmup_steps 100 \
    --seed 1234 \
    --enable_hybrid_engine \
    --actor_zero_stage $ACTOR_ZERO_STAGE \
    --critic_zero_stage $CRITIC_ZERO_STAGE \
    --enable_ema \
    --deepspeed \
    --output_dir $OUTPUT/3_model_PPO \
    2>&1 | tee $OUTPUT/logs/3_training_PPO.log

if [ ${PIPESTATUS[0]} -ne 0 ]; then
  exit 1
fi

