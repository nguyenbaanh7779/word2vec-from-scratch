#!/bin/bash
#SBATCH --job-name=train
#SBATCH --time=08:00:00
#SBATCH --partition=rome                 
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --output=train_output_%j.log
#SBATCH --error=train_error_%j.log

# MODULES & ENVIRONMENT
module purge
module load 2025
module load Anaconda3/2025.06-1

# Initialize conda for this non-interactive shell and activate the env
eval "$(conda shell.bash hook)"
conda activate word2vec-from-scratch

cd "../"

python -u run.py --run-name snellius-fast --epochs 1 --max-window 5 --n-negatives 5 --embed-dim 100 --max-tokens 5000000