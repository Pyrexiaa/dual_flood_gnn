#!/bin/bash
#SBATCH --job-name=train-unified-align2d
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --partition=gpu-long
#SBATCH --gpus=a100-40:1
#SBATCH --mem-per-cpu=128000
#SBATCH --time=2880

source ~/anaconda3/etc/profile.d/conda.sh
conda activate dual_flood_gnn

srun python train.py --config 'configs/unified_model1/config_model1_node_only_1_align2d.yaml' --model 'UnifiedDUALFloodGNNNode1D2D'
srun python train.py --config 'configs/unified_model2/config_model2_node_only_1_align2d.yaml' --model 'UnifiedDUALFloodGNNNode1D2D'

# srun python test.py --config 'configs/unified_model1/config_model1_node_only_1_align2d.yaml' --model 'UnifiedDUALFloodGNNNode1D2D' --model_path 'saved_models/model1_timestep1_align2d.pt'
# srun python test.py --config 'configs/unified_model2/config_model2_node_only_1_align2d.yaml' --model 'UnifiedDUALFloodGNNNode1D2D' --model_path 'saved_models/model2_timestep1_align2d.pt'
# srun python process_kaggle_submissions.py --timestep_to_remove 8 --model1_test_csv 'kaggle_submissions/model1_test.csv' --model2_test_csv 'kaggle_submissions/model2_test.csv' --model1_saved_event_dir 'kaggle_submissions/Model1_1' --model2_saved_event_dir 'kaggle_submissions/Model2_1' --output_file 'kaggle_submissions/node_only_1.csv'
