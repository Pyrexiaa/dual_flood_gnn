#!/bin/bash
#SBATCH --job-name=test_node_edge_6_9
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --partition=gpu
#SBATCH --gpus=a100-40:1
#SBATCH --mem-per-cpu=128000
#SBATCH --time=2880

source ~/anaconda3/etc/profile.d/conda.sh
conda activate dual_flood_gnn

# DUALFloodGNNNode1D2D
# Base Architecture
srun python test.py --config 'configs/model1_edge_v2/config_model1_node_only_6.yaml' --model 'DUALFloodGNNNodeEdge1D2D' --model_path 'saved_models/model1_timestep6.pt'
srun python test.py --config 'configs/model2_edge_v2/config_model2_node_only_6.yaml' --model 'DUALFloodGNNNode1D2D' --model_path 'saved_models/model2_timestep6.pt'
srun python process_kaggle_submissions.py --timestep_to_remove 3 --model1_test_csv 'kaggle_submissions/model1_test.csv' --model2_test_csv 'kaggle_submissions/model2_test.csv' --model1_saved_event_dir 'kaggle_submissions/Model1_6' --model2_saved_event_dir 'kaggle_submissions/Model2_6' --output_file 'kaggle_submissions/node_only_6.csv'
srun python visualize.py --input_csv 'kaggle_submissions/node_only_6.csv' --gt_csv 'kaggle_submissions/ground_truth.csv' --output_csv 'kaggle_submissions/node_only_6_gt.csv'

srun python test.py --config 'configs/model1_edge_v2/config_model1_node_only_7.yaml' --model 'DUALFloodGNNNodeEdge1D2D' --model_path 'saved_models/model1_timestep7.pt'
srun python test.py --config 'configs/model2_edge_v2/config_model2_node_only_7.yaml' --model 'DUALFloodGNNNodeEdge1D2D' --model_path 'saved_models/model2_timestep7.pt'
srun python process_kaggle_submissions.py --timestep_to_remove 2 --model1_test_csv 'kaggle_submissions/model1_test.csv' --model2_test_csv 'kaggle_submissions/model2_test.csv' --model1_saved_event_dir 'kaggle_submissions/Model1_7' --model2_saved_event_dir 'kaggle_submissions/Model2_7' --output_file 'kaggle_submissions/node_only_7.csv'
srun python visualize.py --input_csv 'kaggle_submissions/node_only_7.csv' --gt_csv 'kaggle_submissions/ground_truth.csv' --output_csv 'kaggle_submissions/node_only_7_gt.csv'

srun python test.py --config 'configs/model1_edge_v2/config_model1_node_only_8.yaml' --model 'DUALFloodGNNNodeEdge1D2D' --model_path 'saved_models/model1_timestep8.pt'
srun python test.py --config 'configs/model2_edge_v2/config_model2_node_only_8.yaml' --model 'DUALFloodGNNNodeEdge1D2D' --model_path 'saved_models/model2_timestep8.pt'
srun python process_kaggle_submissions.py --timestep_to_remove 1 --model1_test_csv 'kaggle_submissions/model1_test.csv' --model2_test_csv 'kaggle_submissions/model2_test.csv' --model1_saved_event_dir 'kaggle_submissions/Model1_8' --model2_saved_event_dir 'kaggle_submissions/Model2_8' --output_file 'kaggle_submissions/node_only_8.csv'
srun python visualize.py --input_csv 'kaggle_submissions/node_only_8.csv' --gt_csv 'kaggle_submissions/ground_truth.csv' --output_csv 'kaggle_submissions/node_only_8_gt.csv'

srun python test.py --config 'configs/model1_edge_v2/config_model1_node_only_9.yaml' --model 'DUALFloodGNNNodeEdge1D2D' --model_path 'saved_models/model1_timestep9.pt'
srun python test.py --config 'configs/model2_edge_v2/config_model2_node_only_9.yaml' --model 'DUALFloodGNNNodeEdge1D2D' --model_path 'saved_models/model2_timestep9.pt'
srun python process_kaggle_submissions.py --timestep_to_remove 0 --model1_test_csv 'kaggle_submissions/model1_test.csv' --model2_test_csv 'kaggle_submissions/model2_test.csv' --model1_saved_event_dir 'kaggle_submissions/Model1_9' --model2_saved_event_dir 'kaggle_submissions/Model2_9' --output_file 'kaggle_submissions/node_only_9.csv'
srun python visualize.py --input_csv 'kaggle_submissions/node_only_9.csv' --gt_csv 'kaggle_submissions/ground_truth.csv' --output_csv 'kaggle_submissions/node_only_9_gt.csv'
