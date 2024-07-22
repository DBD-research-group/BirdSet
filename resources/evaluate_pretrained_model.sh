#!/bin/bash
 
model_path='/workspace/logs/train/runs/XCL/efficientnet/2024-06-01_134656/callback_checkpoints/checkpoint-09.ckpt'

 
for test_set in HSN NBP NES PER SNE SSW UHH POW
# for test_set in POW 
do
    echo "Running model at $test_set"
    python eval.py "experiment=local/$test_set/efficientnet_inference_XCL" "module.network.model.local_checkpoint=$model_path" "module.network.model.pretrain_info.hf_name=$test_set" seed=2 "logger.wandb.group=inference_efficientnet_seed2_xcl_$test_set" "trainer.devices=[2]" "tags=['$test_set', 'efficientnet', 'multilabel', 'inference', 'XCL', 'proper-validation']" "ckpt_path=$model_path" "module.network.model.pretrain_info.hf_pretrain_name=XCL" 
done