#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --partition=main
#SBATCH --mem-per-cpu=32GB
#SBATCH --job-name=push_to_hf
#SBATCH --output=/mnt/work/bird2vec/logs/%x_%a.log
date;hostname;pwd
source /mnt/home/lrauch/.zshrc
conda activate uncertainty-evaluation
echo "Start" 

cd /mnt/work/bird2vec

srun python -u push_to_hf.py \
     data/xeno-canto/na_metadata_subset1k.csv \
     na_metadata_servertest_nodecode \
     --decode False


