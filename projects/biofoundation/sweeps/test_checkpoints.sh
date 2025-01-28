#!/bin/bash

GROUP="perch_linearprobing_BirdSetCheckpoints"

# Check if folder path is provided
if [ "$#" -ne 1 ]; then
    echo "Usage: $0 <folder_path>"
    exit 1
fi

folder_path=$1

# Check if the folder exists
if [ ! -d "$folder_path" ]; then
    echo "Error: Folder '$folder_path' does not exist or is not a directory."
    exit 1
fi

# Loop through each file in the folder
for file_path in "$folder_path"/*.ckpt; do
    # Skip if no .ckpt files are found
    if [ ! -f "$file_path" ]; then
        echo "No .ckpt files found in the folder."
        break
    fi
    file_name=$(basename "$file_path")
    echo "Processing file: $file_path"
    python birdset/train.py experiment="biofoundation/birdset/linearprobing/perch_checkpoints" train=False ckpt_path=$file_path logger.wandb.name=$file_name logger.wandb.group=$GROUP

    if [ $? -eq 0 ]; then
        echo "Successfully processed: $file_path"
    else
        echo "Failed to process: $file_path"
    fi
done


# Change the group name to the desired group name and then create report or view on WandB