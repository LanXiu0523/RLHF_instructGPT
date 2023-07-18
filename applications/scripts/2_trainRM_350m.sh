#!/bin/bash
# Copyright (c) Microsoft Corporation.
# SPDX-License-Identifier: Apache-2.0

# DeepSpeed Team
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
    --model_name_or_path facebook/opt-350m \
    --num_padding_at_beginning 1 \
    --weight_decay 0.1 \
    --disable_dropout \
    --gradient_accumulation_steps 4 \
    --zero_stage $ZERO_STAGE \
    --deepspeed \
    --output_dir $OUTPUT/2_model_RM \
    2>&1 | tee $OUTPUT/logs/2_training_RM.log
