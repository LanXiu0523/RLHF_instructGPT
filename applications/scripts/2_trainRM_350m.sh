#!/bin/bash
set -x

OUTPUT=$1
ZERO_STAGE=$2
if [ "$OUTPUT" == "" ]; then
    OUTPUT=./output
fi
if [ "$ZERO_STAGE" == "" ]; then
    ZERO_STAGE=0
fi
mkdir -p $OUTPUT $OUTPUT/2_model_RM $OUTPUT/logs

deepspeed --num_gpus 1 applications/train/2_trainRM.py \
    --data_path  'datafile/Dahoas/rm-static' \
    --data_split 2,4,4 \
    --model_name_or_path facebook/opt-350m \
    --per_device_train_batch_size 16 \
    --per_device_eval_batch_size 16 \
    --max_seq_len 512 \
    --learning_rate 5e-5 \
    --weight_decay 0.1 \
    --num_train_epochs 1 \
    --disable_dropout \
    --gradient_accumulation_steps 4 \
    --lr_scheduler_type cosine \
    --num_warmup_steps 0 \
    --seed 1234 \
    --zero_stage $ZERO_STAGE \
    --deepspeed \
    --output_dir $OUTPUT/2_model_RM \
    2>&1 | tee $OUTPUT/logs/2_training_RM.log
