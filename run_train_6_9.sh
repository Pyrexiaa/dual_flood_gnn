#!/bin/bash
#SBATCH --job-name=train
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --partition=gpu-long
#SBATCH --gpus=a100-40:1
#SBATCH --mem-per-cpu=64000
#SBATCH --time=1440

source ~/anaconda3/etc/profile.d/conda.sh
conda activate dual_flood_gnn

# DUALFloodGNNNode1D2D
srun python train.py --config 'configs/model1/config_model1_node_only_6.yaml' --model 'DUALFloodGNNNode1D2D'
srun python train.py --config 'configs/model2/config_model1_node_only_6.yaml' --model 'DUALFloodGNNNode1D2D'
srun python process_kaggle_submissions.py --model1_test_csv 'data/model1/raw/test.csv' --model2_test_csv 'data/model2/raw/test.csv' --model1_saved_event_dir 'kaggle_submissions/model1_6' --model2_saved_event_dir 'kaggle_submissions/model2_6' --output_file 'kaggle_submissions/node_only_6.csv'
srun python visualize.py --input_csv 'kaggle_submissions/node_only_6.csv' --gt_csv 'kaggle_submissions/ground_truth.csv' --output_csv 'kaggle_submissions/node_only_6_gt.csv'

srun python train.py --config 'configs/model1/config_model1_node_only_7.yaml' --model 'DUALFloodGNNNode1D2D'
srun python train.py --config 'configs/model2/config_model1_node_only_7.yaml' --model 'DUALFloodGNNNode1D2D'
srun python process_kaggle_submissions.py --model1_test_csv 'data/model1/raw/test.csv' --model2_test_csv 'data/model2/raw/test.csv' --model1_saved_event_dir 'kaggle_submissions/model1_7' --model2_saved_event_dir 'kaggle_submissions/model2_7' --output_file 'kaggle_submissions/node_only_7.csv'
srun python visualize.py --input_csv 'kaggle_submissions/node_only_7.csv' --gt_csv 'kaggle_submissions/ground_truth.csv' --output_csv 'kaggle_submissions/node_only_7_gt.csv'

srun python train.py --config 'configs/model1/config_model1_node_only_8.yaml' --model 'DUALFloodGNNNode1D2D'
srun python train.py --config 'configs/model2/config_model1_node_only_8.yaml' --model 'DUALFloodGNNNode1D2D'
srun python process_kaggle_submissions.py --model1_test_csv 'data/model1/raw/test.csv' --model2_test_csv 'data/model2/raw/test.csv' --model1_saved_event_dir 'kaggle_submissions/model1_8' --model2_saved_event_dir 'kaggle_submissions/model2_8' --output_file 'kaggle_submissions/node_only_8.csv'
srun python visualize.py --input_csv 'kaggle_submissions/node_only_8.csv' --gt_csv 'kaggle_submissions/ground_truth.csv' --output_csv 'kaggle_submissions/node_only_8_gt.csv'

srun python train.py --config 'configs/model1/config_model1_node_only_9.yaml' --model 'DUALFloodGNNNode1D2D'
srun python train.py --config 'configs/model2/config_model1_node_only_9.yaml' --model 'DUALFloodGNNNode1D2D'
srun python process_kaggle_submissions.py --model1_test_csv 'data/model1/raw/test.csv' --model2_test_csv 'data/model2/raw/test.csv' --model1_saved_event_dir 'kaggle_submissions/model1_9' --model2_saved_event_dir 'kaggle_submissions/model2_9' --output_file 'kaggle_submissions/node_only_9.csv'
srun python visualize.py --input_csv 'kaggle_submissions/node_only_9.csv' --gt_csv 'kaggle_submissions/ground_truth.csv' --output_csv 'kaggle_submissions/node_only_9_gt.csv'
