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
    --model_name_or_path 'datafile/facebook/opt-350m' \
    --num_padding_at_beginning 1 \
    --weight_decay 0.1 \
    --num_train_epochs 1 \
    --disable_dropout \
    --gradient_accumulation_steps 4 \
    --zero_stage $ZERO_STAGE \
    --seed 1234 \
    --deepspeed \
    --output_dir $OUTPUT/2_model_RM \
    2>&1 | tee $OUTPUT/logs/2_training_RM.log


if [ ${PIPESTATUS[0]} -ne 0 ]; then
  exit 1
fi
   
