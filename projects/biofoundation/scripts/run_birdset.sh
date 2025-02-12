#!/bin/bash

# Datasets to loop through
# Possible datasets: ("HSN", "NBP", "NES", "PER", "POW", "SNE", "SSW", "UHH")
# Models: (12) aves, perch, audiomae, convnext, convnext_bs, eat_ssl, ssast, ast, beats, eat, biolingual, hubert

# Active datasets:
dnames=("HSN" "NBP" "NES")
group="birdset_jan25"
gpu=2


# Check if at least three arguments are provided: models, seeds, and config path
if [ $# -lt 3 ]; then
  echo "Usage: $0 <models> <seeds> <config_path>"
  echo "Example: $0 Perch,AnotherModel 1,2,3 birdset/linearprobing"
  exit 1
fi

# Extract the models, seeds, and config path from the arguments
models=(${1//,/ })
seeds=$2
config_path=$3

# Loop through provided models
for model in "${models[@]}"; do
  echo "Running experiments for model $model"
  # Loop through datasets
  for i in "${!dnames[@]}"; do
    dname=${dnames[$i]}
    echo "Running with dataset_name=$dname"
    projects/biofoundation/train_anti_crash.sh experiment="$config_path/$model" seed=$seeds logger.wandb.group=$group trainer.devices=[$gpu] datamodule.dataset.dataset_name=$dname datamodule.dataset.hf_name=$dname
  done
done