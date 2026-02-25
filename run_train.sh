#!/bin/bash
#SBATCH --job-name=train
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --partition=gpu-long
#SBATCH --gpus=a100-40:1
#SBATCH --mem-per-cpu=128000
#SBATCH --time=2880

source ~/anaconda3/etc/profile.d/conda.sh
conda activate dual_flood_gnn

python train.py --config 'configs/unified_model1/config_model1_node_only_1_extrapolate.yaml' --model 'UnifiedDUALFloodGNN1D2D'

# DUALFloodGNNNode1D2D
srun python train.py --config 'configs/model1_v2/config_model1_node_only_1.yaml' --model 'DUALFloodGNNNode1D2D'
srun python train.py --config 'configs/model2_v2/config_model1_node_only_1.yaml' --model 'DUALFloodGNNNode1D2D'

srun python train.py --config 'configs/model1_v2/config_model1_node_only_2.yaml' --model 'DUALFloodGNNNode1D2D'
srun python train.py --config 'configs/model2_v2/config_model2_node_only_2.yaml' --model 'DUALFloodGNNNode1D2D'

srun python train.py --config 'configs/model1_v2/config_model1_node_only_3.yaml' --model 'DUALFloodGNNNode1D2D'
srun python train.py --config 'configs/model2_v2/config_model2_node_only_3.yaml' --model 'DUALFloodGNNNode1D2D'

srun python train.py --config 'configs/model1_v2/config_model1_node_only_4.yaml' --model 'DUALFloodGNNNode1D2D'
srun python train.py --config 'configs/model2_v2/config_model2_node_only_4.yaml' --model 'DUALFloodGNNNode1D2D'

srun python train.py --config 'configs/model1_v2/config_model1_node_only_5.yaml' --model 'DUALFloodGNNNode1D2D'
srun python train.py --config 'configs/model2_v2/config_model2_node_only_5.yaml' --model 'DUALFloodGNNNode1D2D'