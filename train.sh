#!/bin/bash
set -ex

DEV_ENV=$1

if [ "$DEV_ENV" == "" ]; then
    DEV_ENV=sgl_mach
fi


bash applications/scripts/$DEV_ENV/1_trainSFT_1.3b.sh
bash applications/scripts/$DEV_ENV/2_trainRM_350m.sh
bash applications/scripts/$DEV_ENV/3_trainPPO_1.3b.sh
bash applications/scripts/inference/RM_eval.sh
bash applications/scripts/inference/prompt_eval.sh


SAVE_MODEL="true"
if [ "$SAVE_MODEL" = "true" ]; then
     timestamp=$(date "+%Y%m%d%H%M")
    mkdir -p history_model
    cp -r output history_model/output_$timestamp
    cat applications/scripts/$DEV_ENV/* > history_model/output_$timestamp/args.config
fi

