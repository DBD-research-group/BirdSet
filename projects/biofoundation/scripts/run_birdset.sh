#!/bin/bash

# Possible datasets: "PER" "POW" "NES" "UHH" "HSN" "NBP" "SSW" "SNE"
# Models: (12) aves, perch, audiomae, convnext, convnext_bs, eat_ssl, ssast, ast, beats, eat, biolingual, hubert

# Default values
default_models=("perch")
default_seeds=(1)
default_dnames=("PER" "POW" "NES" "UHH" "HSN" "NBP" "SSW" "SNE")
default_timeouts=(120 120 120 120 120 120 120 120) # All 2 hours for now
default_tags=()
gpu=0
extras=""

# Parse arguments
while [[ "$#" -gt 0 ]]; do
  case "$1" in
    --models)
      IFS=',' read -r -a models <<< "$2"
      shift 2;;
    --seeds)
      IFS=',' read -r -a seeds <<< "$2"
      shift 2;;
    --datasets)
      IFS=',' read -r -a selected_dnames <<< "$2"
      shift 2;;
    --tags)
      IFS=',' read -r -a tags <<< "$2"
      shift 2;;
    --gpu)
      gpu=$2
      shift 2;;
    --config)
      config_path="$2"
      shift 2;;
    --extras)
      extras="$2"
      shift 2;;  
    *)
      echo "Unknown argument: $1"
      exit 1;;
  esac
done

# Set defaults if not provided
models=("${models[@]:-${default_models[@]}}")
seeds=("${seeds[@]:-${default_seeds[@]}}")
dnames=("${default_dnames[@]}")
tags=("${tags[@]:-${default_tags[@]}}")

# Validate required argument
if [ -z "$config_path" ]; then
  echo "Error: --config is required"
  exit 1
fi

# Filter datasets if specific ones are given
if [ "${#selected_dnames[@]}" -gt 0 ]; then
  filtered_dnames=()
  filtered_timeouts=()
  for dataset in "${selected_dnames[@]}"; do
    for i in "${!dnames[@]}"; do
      if [ "${dnames[$i]}" == "$dataset" ]; then
        filtered_dnames+=("${dnames[$i]}")
        filtered_timeouts+=("${default_timeouts[$i]}")
      fi
    done
  done
  dnames=("${filtered_dnames[@]}")
  default_timeouts=("${filtered_timeouts[@]}")

fi

# Function to handle Ctrl+C (SIGINT) and decide behavior
trap ' 
  if [ "$first_ctrl_c_triggered" = true ]; then
    echo "Second Ctrl+C detected. Exiting..."
    exit 1
  else
    echo "Ctrl+C detected. Skipping current experiment... Press Ctrl+C again to exit"
    first_ctrl_c_triggered=true
  fi
' SIGINT

# Main loop
for model in "${models[@]}"; do
  echo "Running experiments for model $model"
  for i in "${!dnames[@]}"; do
    dname=${dnames[$i]}
    timeout=${default_timeouts[$i]}
    echo "Running with dataset_name=$dname"

    # Reset quit flag
    sleep 3 # This allows detecting a quick second Ctrl+C press
    first_ctrl_c_triggered=false

    # Build the extra arguments if --extras was provided
    if [ -n "$extras" ]; then
      extra_args=$(echo "$extras" | sed 's/,/ /g' | sed 's/=/=/g')
    fi

    # Conditionally add tags if they exist (to not override old one if none given)
    tag_args=""
    if [ -n "${tags[*]}" ]; then
      tag_args="+logger.wandb.tags=[$(IFS=,; echo "${tags[*]}")"]
    fi

    projects/biofoundation/train_anti_crash.sh \
      $timeout \
      experiment="$config_path/$model" \
      seed=$seeds \
      trainer.devices=[$gpu] \
      datamodule.dataset.dataset_name=$dname \
      datamodule.dataset.hf_name=$dname \
      trainer.devices=[$gpu] \
      $tag_args \
      $extra_args
  done
done