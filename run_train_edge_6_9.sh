#!/bin/bash
#SBATCH --job-name=train
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --partition=gpu-long
#SBATCH --gpus=a100-40:1
#SBATCH --mem-per-cpu=128000
#SBATCH --time=2880

source ~/anaconda3/etc/profile.d/conda.sh
conda activate dual_flood_gnn

# DUALFloodGNNNode1D2D
srun python train.py --config 'configs/model1_edge_v2/config_model1_node_only_6.yaml' --model 'DUALFloodGNNNodeEdge1D2D'
srun python train.py --config 'configs/model2_edge_v2/config_model2_node_only_6.yaml' --model 'DUALFloodGNNNodeEdge1D2D'

srun python train.py --config 'configs/model1_edge_v2/config_model1_node_only_7.yaml' --model 'DUALFloodGNNNodeEdge1D2D'
srun python train.py --config 'configs/model2_edge_v2/config_model2_node_only_7.yaml' --model 'DUALFloodGNNNodeEdge1D2D'

srun python train.py --config 'configs/model1_edge_v2/config_model1_node_only_8.yaml' --model 'DUALFloodGNNNodeEdge1D2D'
srun python train.py --config 'configs/model2_edge_v2/config_model2_node_only_8.yaml' --model 'DUALFloodGNNNodeEdge1D2D'

srun python train.py --config 'configs/model1_edge_v2/config_model1_node_only_9.yaml' --model 'DUALFloodGNNNodeEdge1D2D'
srun python train.py --config 'configs/model2_edge_v2/config_model2_node_only_9.yaml' --model 'DUALFloodGNNNodeEdge1D2D'