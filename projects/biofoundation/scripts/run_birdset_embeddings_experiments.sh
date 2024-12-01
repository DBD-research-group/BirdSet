#!/bin/bash

# Initialize empty arrays for models and datasets
models=()
datasets=()
gpu_id="0" 


# Parse arguments
while [[ $# -gt 0 ]]; do
  case "$1" in
    --models)
      shift
      while [[ $# -gt 0 && $1 != --* ]]; do  # Collect models until next flag
        models+=("$1")
        shift
      done
      ;;
    --datasets)
      shift
      while [[ $# -gt 0 && $1 != --* ]]; do  # Collect datasets until next flag
        datasets+=("$1")
        shift
      done
      ;;
    --gpu)
      shift
      gpu_id="$1"  # Set the GPU ID to the specified value
      shift
      ;;
    *)
      echo "Unknown option $1. Usage: $0 --models beats convnext ...  --datasets HSN NBP ... --gpu 1"
      exit 1
      ;;
  esac
done

# Display models and datasets
echo "Models: ${models[@]}"
echo "Datasets: ${datasets[@]}"
echo "Using GPU: $gpu_id"

# Run experiments
for dataset in "${datasets[@]}"; do
  for model in "${models[@]}"; do
    echo "Processing model $model with dataset $dataset"
    python birdset/train.py experiment="biofoundation/embedding/BirdSet/$model" seed=2 datamodule.average=False datamodule.k_samples=0 trainer.devices=[$gpu_id] datamodule.dataset.dataset_name=$dataset datamodule.dataset.hf_name=$dataset
  done
done
