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
# srun python train.py --config 'configs/model1/config_model1_node_only_1.yaml' --model 'DUALFloodGNNNode1D2D'
# srun python train.py --config 'configs/model2/config_model1_node_only_1.yaml' --model 'DUALFloodGNNNode1D2D'
# srun python process_kaggle_submissions.py --model1_test_csv 'data/model1/raw/test.csv' --model2_test_csv 'data/model2/raw/test.csv' --model1_saved_event_dir 'kaggle_submissions/model1_1' --model2_saved_event_dir 'kaggle_submissions/model2_1' --output_file 'kaggle_submissions/node_only_1.csv'
# srun python visualize.py --input_csv 'kaggle_submissions/node_only_1.csv' --gt_csv 'kaggle_submissions/ground_truth.csv' --output_csv 'kaggle_submissions/node_only_1_gt.csv'

srun python train.py --config 'configs/model1/config_model1_node_only_2.yaml' --model 'DUALFloodGNNNode1D2D'
srun python train.py --config 'configs/model2/config_model1_node_only_2.yaml' --model 'DUALFloodGNNNode1D2D'
srun python process_kaggle_submissions.py --model1_test_csv 'data/model1/raw/test.csv' --model2_test_csv 'data/model2/raw/test.csv' --model1_saved_event_dir 'kaggle_submissions/model1_2' --model2_saved_event_dir 'kaggle_submissions/model2_2' --output_file 'kaggle_submissions/node_only_2.csv'
srun python visualize.py --input_csv 'kaggle_submissions/node_only_2.csv' --gt_csv 'kaggle_submissions/ground_truth.csv' --output_csv 'kaggle_submissions/node_only_2_gt.csv'

srun python train.py --config 'configs/model1/config_model1_node_only_3.yaml' --model 'DUALFloodGNNNode1D2D'
srun python train.py --config 'configs/model2/config_model1_node_only_3.yaml' --model 'DUALFloodGNNNode1D2D'
srun python process_kaggle_submissions.py --model1_test_csv 'data/model1/raw/test.csv' --model2_test_csv 'data/model2/raw/test.csv' --model1_saved_event_dir 'kaggle_submissions/model1_3' --model2_saved_event_dir 'kaggle_submissions/model2_3' --output_file 'kaggle_submissions/node_only_3.csv'
srun python visualize.py --input_csv 'kaggle_submissions/node_only_3.csv' --gt_csv 'kaggle_submissions/ground_truth.csv' --output_csv 'kaggle_submissions/node_only_3_gt.csv'

srun python train.py --config 'configs/model1/config_model1_node_only_4.yaml' --model 'DUALFloodGNNNode1D2D'
srun python train.py --config 'configs/model2/config_model1_node_only_4.yaml' --model 'DUALFloodGNNNode1D2D'
srun python process_kaggle_submissions.py --model1_test_csv 'data/model1/raw/test.csv' --model2_test_csv 'data/model2/raw/test.csv' --model1_saved_event_dir 'kaggle_submissions/model1_4' --model2_saved_event_dir 'kaggle_submissions/model2_4' --output_file 'kaggle_submissions/node_only_4.csv'
srun python visualize.py --input_csv 'kaggle_submissions/node_only_4.csv' --gt_csv 'kaggle_submissions/ground_truth.csv' --output_csv 'kaggle_submissions/node_only_4_gt.csv'

srun python train.py --config 'configs/model1/config_model1_node_only_5.yaml' --model 'DUALFloodGNNNode1D2D'
srun python train.py --config 'configs/model2/config_model1_node_only_5.yaml' --model 'DUALFloodGNNNode1D2D'
srun python process_kaggle_submissions.py --model1_test_csv 'data/model1/raw/test.csv' --model2_test_csv 'data/model2/raw/test.csv' --model1_saved_event_dir 'kaggle_submissions/model1_5' --model2_saved_event_dir 'kaggle_submissions/model2_5' --output_file 'kaggle_submissions/node_only_5.csv'
srun python visualize.py --input_csv 'kaggle_submissions/node_only_5.csv' --gt_csv 'kaggle_submissions/ground_truth.csv' --output_csv 'kaggle_submissions/node_only_5_gt.csv'