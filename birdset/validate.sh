#!/bin/bash
 
base_path="/workspace/logs/train/runs/XCL/efficientnet/2024-06-01_134656/callback_checkpoints/checkpoint-"
 
for i in {0..9}
do
    # zero-padding
    checkpoint="${base_path}$(printf "%02d" $i).ckpt"
    echo "Running model with checkpoint: $checkpoint"
    python eval.py 'experiment=local/POW/efficientnet_inference_XCL' "module.network.model.local_checkpoint='$checkpoint'" seed=2 "logger.wandb.group='POW_validation_effnent_xcl_seed_2'"  "trainer.devices=[2]"

done