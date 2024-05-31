#!/bin/bash
 
base_path="/workspace/logs/train/runs/XCM/eat/2024-05-25_192206/callback_checkpoints/checkpoint-"
 
for i in {0..29}
do
    # zero-padding
    checkpoint="${base_path}$(printf "%02d" $i).ckpt"
    echo "Running model with checkpoint: $checkpoint"
    python eval.py 'experiment=local/POW/eat_inference_POW_XCM' "module.network.model.local_checkpoint='$checkpoint'" seed=1 "logger.wandb.group='POW_inference_eat_xcm_seed_1'"  "trainer.devices=[0]"

done