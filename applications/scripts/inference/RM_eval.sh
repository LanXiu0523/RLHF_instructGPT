#!/bin/bash
set -x

OUTPUT=$1
if [ "$OUTPUT" == "" ]; then
    OUTPUT=./output
fi


python3 applications/inference/RM_eval.py \
    --model_name_or_path 'output/2_model_RM' \
    2>&1 | tee $OUTPUT/logs/RM_eval.log
