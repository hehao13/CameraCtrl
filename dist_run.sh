#!/bin/bash

set -x

CONFIG=$1
GPUS=$2
PT_SCRIPT=$3
RANDOM_PORT=$((49152 + RANDOM % 16384))

python -m torch.distributed.launch \
    --nproc_per_node=$GPUS \
    --master_port=$RANDOM_PORT \
    ${PT_SCRIPT} \
        --config=${CONFIG} \
        --launcher=pytorch \
        --port=${RANDOM_PORT}