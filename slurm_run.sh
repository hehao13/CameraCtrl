#!/bin/bash

set -x

PARTITION=$1
JOB_NAME=$2
GPUS=$3
CONFIG=$4
PT_SCRIPT=${5}
CPUS_PER_TASK=${CPUS_PER_TASK:-10}
RANDOM_PORT=$((49152 + RANDOM % 16384))
if [ $GPUS -lt 8 ]; then
    GPUS_PER_NODE=${GPUS_PER_NODE:-$GPUS}
else
    GPUS_PER_NODE=${GPUS_PER_NODE:-8}
fi

srun -p ${PARTITION} \
    --job-name=${JOB_NAME} \
    --gres=gpu:${GPUS_PER_NODE} \
    --ntasks=${GPUS} \
    --ntasks-per-node=${GPUS_PER_NODE} \
    --cpus-per-task=${CPUS_PER_TASK} \
    --kill-on-bad-exit=1 \
    python -u ${PT_SCRIPT} \
        --config=${CONFIG} \
        --launcher=slurm \
        --port=${RANDOM_PORT}