#!/bin/bash
 
model_path='/workspace/logs/train/runs/XCL/eat/2024-05-19_105101/callback_checkpoints/checkpoint-26.ckpt'

 
# for test_set in HSN NBP NES PER SNE SSW UHH
for test_set in POW 
do
    echo "Running model at $test_set"
    python eval.py "experiment=local/$test_set/eat_inference_XCL" "module.network.model.local_checkpoint=$model_path" "module.network.model.pretrain_info.hf_name=$test_set" seed=1 "logger.wandb.group=inference_eat_xcl_$test_set" "trainer.devices=[2]"
done