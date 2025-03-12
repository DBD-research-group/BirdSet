#!/bin/bash

# Store timeout time
timeout="$1"
shift

# Run the python script with the specified command and pass any additional arguments
python projects/biofoundation/crash_detection.py "$timeout" birdset/train.py --config-path '../projects/biofoundation/configs' --config-dir 'configs' --multirun 'logger=wandb_biofoundation' "$@"