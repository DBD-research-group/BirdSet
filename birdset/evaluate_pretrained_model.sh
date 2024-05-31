#!/bin/bash
 
model_path='/workspace/logs/train/runs/XCL/eat/2024-05-27_100508/callback_checkpoints/checkpoint-06.ckpt'

 
for test_set in HSN NBP NES PER SNE SSW UHH POW
# for test_set in POW 
do
    echo "Running model at $test_set"
    python eval.py "experiment=local/$test_set/eat_m_inference_XCM" "module.network.model.local_checkpoint=$model_path" "module.network.model.pretrain_info.hf_name=$test_set" seed=3 "logger.wandb.group=inference_eat_m_xcl_$test_set" "trainer.devices=[2]" "tags=['$test_set', 'eat-m', 'multilabel', 'inference', 'XCL', 'proper-validation']" "ckpt_path=$model_path" "module.network.model.pretrain_info.hf_pretrain_name=XCL" 
done