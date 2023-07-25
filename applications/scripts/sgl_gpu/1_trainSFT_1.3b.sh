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


if [ -d ${OUTPUT} ]; then
    rm -rf $OUTPUT
fi	
mkdir -p $OUTPUT $OUTPUT/1_model_SFT $OUTPUT/logs


deepspeed --num_gpus 1 applications/train/1_trainSFT.py \
    --data_path 'datafile/Dahoas/rm-static' \
    --data_split 2,4,4 \
    --model_name_or_path 'datafile/facebook/opt-1.3b' \
    --num_train_epochs 16 \
    --gradient_accumulation_steps 8 \
    --lora_dim 128 \
    --zero_stage $ZERO_STAGE \
    --num_warmup_steps 0 \
    --seed 1234 \
    --deepspeed \
    --output_dir $OUTPUT/1_model_SFT \
    2>&1 | tee $OUTPUT/logs/1_training_SFT.log


if [ ${PIPESTATUS[0]} -ne 0 ]; then
  exit 1
fi

