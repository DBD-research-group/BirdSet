#!/bin/bash

# Datasets to loop through
#("beans_watkins" "beans_bats" "beans_cbi" "beans_dogs" "beans_humbugdb")
dnames=("beans_watkins" "beans_bats" "beans_cbi" "beans_dogs" "beans_humbugdb")
dclasses=(31 10 264 10 14)

# Check if at least one argument is provided
if [ $# -eq 0 ]; then
  echo "No model provided: Example: Perch"
  exit 1
fi

# Loop through provided experiments
for experiment in "$@"; do
  echo "Running experiment $experiment"
  for i in "${!dnames[@]}"; do
    dname=${dnames[$i]}
    dclass=${dclasses[$i]}
    echo "Running with dataset_name=$dname and n_classes=$dclass"
    python birdset/train.py experiment="local/embedding/BEANS/$experiment" datamodule.dataset.dataset_name=$dname datamodule.dataset.hf_path="DBD-research-group/$dname" datamodule.dataset.n_classes=$dclass
  done
done