#!/bin/bash

# Run the python script with the specified command and pass any additional arguments
python birdset/train.py 'hydra.searchpath=[pkg://projects/biofoundation/configs]' "$@"