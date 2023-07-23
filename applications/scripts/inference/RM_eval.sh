#!/bin/bash
set -x

python applications/inference/RM_eval.py \
    --model_name_or_path 'output/2_model_RM' \
    2>&1 | tee $OUTPUT/logs/RM_eval.log
