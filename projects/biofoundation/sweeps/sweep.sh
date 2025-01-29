#!/bin/bash

if [ "$#" -ne 2 ]; then
    echo "Usage: $0 <CUDA_DEVICE> <SWEEP_ID>"
    exit 1
fi

CUDA_DEVICE=$1
SWEEP_ID=$2

# Set the CUDA_VISIBLE_DEVICES environment variable so it uses correct GPU
export CUDA_VISIBLE_DEVICES=$CUDA_DEVICE

echo "Running wandb agent with CUDA_VISIBLE_DEVICES=$CUDA_DEVICE and Sweep ID=$SWEEP_ID"

# Run the wandb agent
wandb agent $SWEEP_ID

exit 0