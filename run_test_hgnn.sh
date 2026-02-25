#!/bin/bash
#SBATCH --job-name=test
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --partition=gpu
#SBATCH --gpus=a100-40:1
#SBATCH --mem-per-cpu=128000
#SBATCH --time=2880

source ~/anaconda3/etc/profile.d/conda.sh
conda activate dual_flood_gnn

# DUALFloodGNNNode1D2D
# Base Architecture
srun python test.py --config 'configs/hgnn/config_model1_node_only_1.yaml' --model 'DUALFloodHGNN1D2D' --model_path 'saved_models/model1_timestep1.pt'
srun python test.py --config 'configs/hgnn/config_model2_node_only_1.yaml' --model 'DUALFloodHGNN1D2D' --model_path 'saved_models/model2_timestep1.pt'
srun python process_kaggle_submissions.py --timestep_to_remove 8 --model1_test_csv 'kaggle_submissions/model1_test.csv' --model2_test_csv 'kaggle_submissions/model2_test.csv' --model1_saved_event_dir 'kaggle_submissions/Model1_1' --model2_saved_event_dir 'kaggle_submissions/Model2_1' --output_file 'kaggle_submissions/node_only_1.csv'
srun python visualize.py --input_csv 'kaggle_submissions/node_only_1.csv' --gt_csv 'kaggle_submissions/ground_truth.csv' --output_csv 'kaggle_submissions/node_only_1_gt.csv'

srun python test.py --config 'configs/hgnn/config_model1_node_only_2.yaml' --model 'DUALFloodHGNN1D2D' --model_path 'saved_models/model1_timestep2.pt'
srun python test.py --config 'configs/hgnn/config_model2_node_only_2.yaml' --model 'DUALFloodHGNN1D2D' --model_path 'saved_models/model2_timestep2.pt'
srun python process_kaggle_submissions.py --timestep_to_remove 7 --model1_test_csv 'kaggle_submissions/model1_test.csv' --model2_test_csv 'kaggle_submissions/model2_test.csv' --model1_saved_event_dir 'kaggle_submissions/Model1_2' --model2_saved_event_dir 'kaggle_submissions/Model2_2' --output_file 'kaggle_submissions/node_only_2.csv'
srun python visualize.py --input_csv 'kaggle_submissions/node_only_2.csv' --gt_csv 'kaggle_submissions/ground_truth.csv' --output_csv 'kaggle_submissions/node_only_2_gt.csv'

srun python test.py --config 'configs/hgnn/config_model1_node_only_3.yaml' --model 'DUALFloodHGNN1D2D' --model_path 'saved_models/model1_timestep3.pt'
srun python test.py --config 'configs/hgnn/config_model2_node_only_3.yaml' --model 'DUALFloodHGNN1D2D' --model_path 'saved_models/model2_timestep3.pt'
srun python process_kaggle_submissions.py --timestep_to_remove 6 --model1_test_csv 'kaggle_submissions/model1_test.csv' --model2_test_csv 'kaggle_submissions/model2_test.csv' --model1_saved_event_dir 'kaggle_submissions/Model1_3' --model2_saved_event_dir 'kaggle_submissions/Model2_3' --output_file 'kaggle_submissions/node_only_3.csv'
srun python visualize.py --input_csv 'kaggle_submissions/node_only_3.csv' --gt_csv 'kaggle_submissions/ground_truth.csv' --output_csv 'kaggle_submissions/node_only_3_gt.csv'

srun python test.py --config 'configs/hgnn/config_model1_node_only_4.yaml' --model 'DUALFloodHGNN1D2D' --model_path 'saved_models/model1_timestep4.pt'
srun python test.py --config 'configs/hgnn/config_model2_node_only_4.yaml' --model 'DUALFloodHGNN1D2D' --model_path 'saved_models/model2_timestep4.pt'
srun python process_kaggle_submissions.py --timestep_to_remove 5 --model1_test_csv 'kaggle_submissions/model1_test.csv' --model2_test_csv 'kaggle_submissions/model2_test.csv' --model1_saved_event_dir 'kaggle_submissions/Model1_4' --model2_saved_event_dir 'kaggle_submissions/Model2_4' --output_file 'kaggle_submissions/node_only_4.csv'
srun python visualize.py --input_csv 'kaggle_submissions/node_only_4.csv' --gt_csv 'kaggle_submissions/ground_truth.csv' --output_csv 'kaggle_submissions/node_only_4_gt.csv'

srun python test.py --config 'configs/hgnn/config_model1_node_only_5.yaml' --model 'DUALFloodHGNN1D2D' --model_path 'saved_models/model1_timestep5.pt'
srun python test.py --config 'configs/hgnn/config_model2_node_only_5.yaml' --model 'DUALFloodHGNN1D2D' --model_path 'saved_models/model2_timestep5.pt'
srun python process_kaggle_submissions.py --timestep_to_remove 4 --model1_test_csv 'kaggle_submissions/model1_test.csv' --model2_test_csv 'kaggle_submissions/model2_test.csv' --model1_saved_event_dir 'kaggle_submissions/Model1_5' --model2_saved_event_dir 'kaggle_submissions/Model2_5' --output_file 'kaggle_submissions/node_only_5.csv'
srun python visualize.py --input_csv 'kaggle_submissions/node_only_5.csv' --gt_csv 'kaggle_submissions/ground_truth.csv' --output_csv 'kaggle_submissions/node_only_5_gt.csv'

srun python test.py --config 'configs/hgnn/config_model1_node_only_6.yaml' --model 'DUALFloodHGNN1D2D' --model_path 'saved_models/model1_timestep6.pt'
srun python test.py --config 'configs/hgnn/config_model2_node_only_6.yaml' --model 'DUALFloodHGNN1D2D' --model_path 'saved_models/model2_timestep6.pt'
srun python process_kaggle_submissions.py --timestep_to_remove 3 --model1_test_csv 'kaggle_submissions/model1_test.csv' --model2_test_csv 'kaggle_submissions/model2_test.csv' --model1_saved_event_dir 'kaggle_submissions/Model1_6' --model2_saved_event_dir 'kaggle_submissions/Model2_6' --output_file 'kaggle_submissions/node_only_6.csv'
srun python visualize.py --input_csv 'kaggle_submissions/node_only_6.csv' --gt_csv 'kaggle_submissions/ground_truth.csv' --output_csv 'kaggle_submissions/node_only_6_gt.csv'

srun python test.py --config 'configs/hgnn/config_model1_node_only_7.yaml' --model 'DUALFloodHGNN1D2D' --model_path 'saved_models/model1_timestep7.pt'
srun python test.py --config 'configs/hgnn/config_model2_node_only_7.yaml' --model 'DUALFloodHGNN1D2D' --model_path 'saved_models/model2_timestep7.pt'
srun python process_kaggle_submissions.py --timestep_to_remove 2 --model1_test_csv 'kaggle_submissions/model1_test.csv' --model2_test_csv 'kaggle_submissions/model2_test.csv' --model1_saved_event_dir 'kaggle_submissions/Model1_7' --model2_saved_event_dir 'kaggle_submissions/Model2_7' --output_file 'kaggle_submissions/node_only_7.csv'
srun python visualize.py --input_csv 'kaggle_submissions/node_only_7.csv' --gt_csv 'kaggle_submissions/ground_truth.csv' --output_csv 'kaggle_submissions/node_only_7_gt.csv'

srun python test.py --config 'configs/hgnn/config_model1_node_only_8.yaml' --model 'DUALFloodHGNN1D2D' --model_path 'saved_models/model1_timestep8.pt'
srun python test.py --config 'configs/hgnn/config_model2_node_only_8.yaml' --model 'DUALFloodHGNN1D2D' --model_path 'saved_models/model2_timestep8.pt'
srun python process_kaggle_submissions.py --timestep_to_remove 1 --model1_test_csv 'kaggle_submissions/model1_test.csv' --model2_test_csv 'kaggle_submissions/model2_test.csv' --model1_saved_event_dir 'kaggle_submissions/Model1_8' --model2_saved_event_dir 'kaggle_submissions/Model2_8' --output_file 'kaggle_submissions/node_only_8.csv'
srun python visualize.py --input_csv 'kaggle_submissions/node_only_8.csv' --gt_csv 'kaggle_submissions/ground_truth.csv' --output_csv 'kaggle_submissions/node_only_8_gt.csv'

srun python test.py --config 'configs/hgnn/config_model1_node_only_9.yaml' --model 'DUALFloodHGNN1D2D' --model_path 'saved_models/model1_timestep9.pt'
srun python test.py --config 'configs/hgnn/config_model2_node_only_9.yaml' --model 'DUALFloodHGNN1D2D' --model_path 'saved_models/model2_timestep9.pt'
srun python process_kaggle_submissions.py --timestep_to_remove 0 --model1_test_csv 'kaggle_submissions/model1_test.csv' --model2_test_csv 'kaggle_submissions/model2_test.csv' --model1_saved_event_dir 'kaggle_submissions/Model1_9' --model2_saved_event_dir 'kaggle_submissions/Model2_9' --output_file 'kaggle_submissions/node_only_9.csv'
srun python visualize.py --input_csv 'kaggle_submissions/node_only_9.csv' --gt_csv 'kaggle_submissions/ground_truth.csv' --output_csv 'kaggle_submissions/node_only_9_gt.csv'