#!/bin/bash
#SBATCH --job-name=train
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --partition=gpu-long
#SBATCH --gpus=a100-40:1
#SBATCH --mem-per-cpu=64000
#SBATCH --time=1440

source ~/anaconda3/etc/profile.d/conda.sh
conda activate dual_flood_gnn

srun python competition_baseline/main.py
