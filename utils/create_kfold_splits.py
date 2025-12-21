"""
Script to split a CSV file into k-fold train/test sets.

Creates foldN_train.csv and foldN_test.csv files for each fold N.
"""

import argparse
import pandas as pd
from pathlib import Path
import os

def generate_dataset_csv(hdf_folder: str, output_csv: str, start_run_id: int = 1):
    files = sorted([f for f in os.listdir(hdf_folder) if f.endswith(".hdf")])

    data = []
    run_id = start_run_id

    for fname in files:
        filepath = os.path.join(hdf_folder, fname)
        splited_filepath = filepath.split("/")
        last_two = "/".join(splited_filepath[-2:])

        data.append({
            "Run_ID": run_id,
            "Event": "",
            "HECRAS_Filepath": last_two,
            "Rain": "",
            "Inflow": "",
            "Time_Interval": "5 m"
        })

        run_id += 1

    df = pd.DataFrame(data)
    df.to_csv(output_csv, index=False)
    print(f"Generated dataset with {len(df)} rows → {output_csv}")

def create_kfold_splits(input_csv: str, output_dir: str, n_splits: int = 5):
    """
    Split a CSV file into k-fold train/test sets.
    
    Args:
        input_csv: Path to input CSV file
        output_dir: Directory to save the fold CSV files
        n_splits: Number of folds (default: 5)
    """
    # Read the input CSV
    df = pd.read_csv(input_csv)
    print(f"Loaded {len(df)} rows from {input_csv}")
    
    # Create output directory if it doesn't exist
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Calculate fold size
    n_samples = len(df)
    fold_size = n_samples // n_splits
    remainder = n_samples % n_splits
    
    # Create splits manually
    for fold_num in range(1, n_splits + 1):
        # Calculate test indices for this fold
        start_idx = (fold_num - 1) * fold_size
        # Distribute remainder samples across first folds
        if fold_num <= remainder:
            start_idx += fold_num - 1
            end_idx = start_idx + fold_size + 1
        else:
            start_idx += remainder
            end_idx = start_idx + fold_size
        
        # Split data
        test_df = df.iloc[start_idx:end_idx]
        train_df = pd.concat([df.iloc[:start_idx], df.iloc[end_idx:]], ignore_index=True)
        
        # Save train and test files
        train_file = output_path / f"fold{fold_num}_train.csv"
        test_file = output_path / f"fold{fold_num}_test.csv"
        
        train_df.to_csv(train_file, index=False)
        test_df.to_csv(test_file, index=False)
        
        print(f"Fold {fold_num}: {len(train_df)} train, {len(test_df)} test samples")
        print(f"  - Saved: {train_file}")
        print(f"  - Saved: {test_file}")
    
    print(f"\nSuccessfully created {n_splits} folds in {output_dir}")


def main():
    parser = argparse.ArgumentParser(
        description="Split a CSV file into k-fold train/test sets"
    )
    parser.add_argument(
        "input_csv",
        type=str,
        help="Path to input CSV file"
    )
    parser.add_argument(
        "-o", "--output-dir",
        type=str,
        default=None,
        help="Output directory for fold files (default: same directory as input)"
    )
    parser.add_argument(
        "-k", "--n-splits",
        type=int,
        default=5,
        help="Number of folds (default: 5)"
    )
    
    args = parser.parse_args()
    
    # Set output directory to input file's directory if not specified
    if args.output_dir is None:
        args.output_dir = Path(args.input_csv).parent
    
    create_kfold_splits(
        input_csv=args.input_csv,
        output_dir=args.output_dir,
        n_splits=args.n_splits
    )


if __name__ == "__main__":
    generate_dataset_csv(
        hdf_folder="data/Model4/raw/HEC-RAS_Results",
        output_csv="data/Model4/raw/full_dataset.csv"
    )
    create_kfold_splits(
        input_csv="data/Model4/raw/full_dataset.csv",
        output_dir="data/Model4/raw",
        n_splits=5
    )
    # main()
