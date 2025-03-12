#!/bin/bash

# Possible datasets: watkins, beans_bats, beans_cbi, beans_dogs, beans_humbugdb
# Models: (12) aves, perch, audiomae, convnext, convnext_bs, eat_ssl, ssast, ast, beats, eat, biolingual, hubert

# Default values
default_models=("perch")
default_seeds=(1)
default_dnames=("beans_watkins" "beans_cbi" "beans_dogs" "beans_humbugdb" "beans_bats")
default_dclasses=(31 264 10 14 10)
default_timeouts=(60 180 60 130 70)
default_tags=("run")
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
dclasses=("${default_dclasses[@]}")
tags=("${tags[@]:-${default_tags[@]}}")

# Validate required argument
if [ -z "$config_path" ]; then
  echo "Error: --config is required"
  exit 1
fi

# Filter datasets if specific ones are given
if [ "${#selected_dnames[@]}" -gt 0 ]; then
  filtered_dnames=()
  filtered_dclasses=()
  filtered_timeouts=()
  for dataset in "${selected_dnames[@]}"; do
    for i in "${!dnames[@]}"; do
      if [ "${dnames[$i]}" == "$dataset" ]; then
        filtered_dnames+=("${dnames[$i]}")
        filtered_dclasses+=("${dclasses[$i]}")
        filtered_timeouts+=("${default_timeouts[$i]}")
      fi
    done
  done
  dnames=("${filtered_dnames[@]}")
  dclasses=("${filtered_dclasses[@]}")
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
    dclass=${dclasses[$i]}
    timeout=${default_timeouts[$i]}
    echo "Running with dataset_name=$dname, n_classes=$dclass"

    # Reset quit flag
    sleep 3 # This allows detecting a quick second Ctrl+C press
    first_ctrl_c_triggered=false
    # Build the extra arguments if --extras was provided
    if [ -n "$extras" ]; then
      extra_args=$(echo "$extras" | sed 's/,/ /g' | sed 's/=/=/g')
    fi

    projects/biofoundation/train_anti_crash.sh \
      $timeout \
      experiment="$config_path/$model" \
      seed=$seeds \
      trainer.devices=[$gpu] \
      datamodule.dataset.dataset_name=$dname \
      datamodule.dataset.hf_path="DBD-research-group/$dname" \
      datamodule.dataset.n_classes=$dclass \
      trainer.devices=[$gpu] \
      logger.wandb.tags=[$(IFS=,; echo "${tags[*]}")] \
      $extra_args
  done
done