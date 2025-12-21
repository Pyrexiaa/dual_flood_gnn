#!/bin/bash
#SBATCH --job-name=train
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --partition=gpu-long
#SBATCH --gpus=a100-40:1
#SBATCH --mem-per-cpu=64000
#SBATCH --time=1440
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err

source ~/anaconda3/etc/profile.d/conda.sh
conda activate dual_flood_gnn

mkdir -p notebook_runs logs

RUN_ID=${SLURM_JOB_ID}

srun papermill \
  eda_1d2d_model3.ipynb \
  notebook_runs/eda_1d2d_model3_run_${RUN_ID}.ipynb \
  -k python
