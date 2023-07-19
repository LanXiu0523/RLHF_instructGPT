#!/bin/bash
set -x

#train
#bash applications/scripts/1_trainSFT_1.3b.sh

#bash applications/scripts/2_trainRM_350m.sh

bash applications/scripts/3_trainPPO_1.3b.sh
