#!/bin/bash
 
base_path="/workspace/logs/train/runs/XCL/eat/2024-05-17_075159/callback_checkpoints/checkpoint-"
 
for i in {0..29}
do
    # zero-padding
    checkpoint="${base_path}$(printf "%02d" $i).ckpt"
    echo "Running model with checkpoint: $checkpoint"
    python eval.py 'experiment=local/POW/eat_inference_POW_XCL' "module.network.model.local_checkpoint='$checkpoint'" seed=3 "logger.wandb.group='POW_inference_eat_xcl_seed_3'"

done