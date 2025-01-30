#!/bin/bash

# Run the python script with the specified command and pass any additional arguments
python projects/biofoundation/crash_detection.py birdset/train.py --config-path '../projects/biofoundation/configs' --config-dir 'configs' --multirun 'logger=wandb_biofoundation' "$@"