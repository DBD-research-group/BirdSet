#!/bin/bash

# Datasets to loop through
# Possible datasets: ("beans_watkins" "beans_bats" "beans_cbi" "beans_dogs" "beans_humbugdb")
# Models: aves, perch, audiomae, convnext, convnext_bs, eat_ssl, ssast, ast, beats, eat, biolingual, hubert

# Active datasets:
dnames=("beans_watkins" "beans_cbi" "beans_dogs" "beans_humbugdb" "beans_bats")
dclasses=(31 264 10 14 10)
group="beans_jan25"
gpu=0


# Check if at least three arguments are provided: models, seeds, and config path
if [ $# -lt 3 ]; then
  echo "Usage: $0 <models> <seeds> <config_path>"
  echo "Example: $0 Perch,AnotherModel 1,2,3 beans/linearprobing"
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
    dclass=${dclasses[$i]}
    echo "Running with dataset_name=$dname and n_classes=$dclass"
    projects/biofoundation/train_anti_crash.sh experiment="$config_path/$model" seed=$seeds logger.wandb.group=$group trainer.devices=[$gpu] datamodule.dataset.dataset_name=$dname datamodule.dataset.hf_path="DBD-research-group/$dname" datamodule.dataset.n_classes=$dclass
  done
done