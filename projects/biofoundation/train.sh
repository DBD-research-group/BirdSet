#!/bin/bash

# Run the python script with the specified command and pass any additional arguments
python birdset/train.py --config-path '../projects/biofoundation/configs' --config-dir 'configs' 'logger=wandb_biofoundation' "$@"