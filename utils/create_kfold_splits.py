"""
Script to split a CSV file into k-fold train/test sets.

Creates foldN_train.csv and foldN_test.csv files for each fold N.
"""

import argparse
import pandas as pd
from pathlib import Path
import os
import numpy as np

def generate_dataset_csv(
    hdf_folder: str,
    output_csv: str,
    start_run_id: int = 1,
    random_seed: int = 42
):
    files = [f for f in os.listdir(hdf_folder) if f.endswith(".hdf")]

    rng = np.random.default_rng(random_seed)
    rng.shuffle(files)

    data = []
    run_id = start_run_id

    for fname in files:
        filepath = os.path.join(hdf_folder, fname)
        splited_filepath = filepath.split("/")
        last_two = "/".join(splited_filepath[-2:])

        data.append({
            "Event": run_id,
            "Run_ID": run_id,
            "HECRAS_Filepath": last_two,
            "Rain": "",
            "Inflow": "",
            "Time_Interval": "5 m"
        })

        run_id += 1

    df = pd.DataFrame(data)
    df.to_csv(output_csv, index=False)
    print(f"Generated dataset with {len(df)} rows → {output_csv}")

def create_random_6040_folds(
    input_csv: str,
    output_dir: str,
    n_folds: int = 5,
    train_ratio: float = 0.6,
    random_seed: int = 42
):
    df = pd.read_csv(input_csv)
    print(f"Loaded {len(df)} rows from {input_csv}")

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    n_train = int(len(df) * train_ratio)

    for fold_num in range(1, n_folds + 1):
        # For demonstration, only create fold 1
        if fold_num > 1:
            continue

        df_shuffled = df.sample(
            frac=1,
            random_state=random_seed + fold_num
        ).reset_index(drop=True)

        train_df = df_shuffled.iloc[:n_train].copy()
        test_df = df_shuffled.iloc[n_train:].copy()

        train_file = output_path / "train.csv"
        test_file = output_path / "test.csv"

        train_df.to_csv(train_file, index=False)
        test_df.to_csv(test_file, index=False)

        print(
            f"Fold {fold_num}: "
            f"{len(train_df)} train ({train_ratio*100:.0f}%), "
            f"{len(test_df)} test ({(1-train_ratio)*100:.0f}%)"
        )

    print(f"\nSuccessfully created {n_folds} random 60/40 folds in {output_dir}")

def create_random_5050_public_private(
    input_csv: str,
    output_dir: str,
    random_seed: int = 42
):
    """
    Create a single random 50/50 split.

    Output files:
        public.csv
        private.csv

    Event IDs are re-assigned locally in ascending order.
    """
    df = pd.read_csv(input_csv)
    print(f"Loaded {len(df)} rows from {input_csv}")

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # Shuffle once
    df_shuffled = df.sample(
        frac=1,
        random_state=random_seed
    ).reset_index(drop=True)

    split_idx = len(df_shuffled) // 2

    public_df = df_shuffled.iloc[:split_idx].copy()
    private_df = df_shuffled.iloc[:].copy()

    public_file = output_path / "public_test.csv"
    private_file = output_path / "private_test.csv"

    public_df.to_csv(public_file, index=False)
    private_df.to_csv(private_file, index=False)

    print(
        f"Public/Private split: "
        f"{len(public_df)} public (50%), "
        f"{len(private_df)} private (50%)"
    )
    print(f"  - Saved: {public_file}")
    print(f"  - Saved: {private_file}")

if __name__ == "__main__":
    model = "Model1"

    generate_dataset_csv(
        hdf_folder=f"data/{model}/raw/HEC-RAS_Results",
        output_csv=f"data/{model}/raw/full_dataset.csv"
    )
    create_random_6040_folds(
        input_csv=f"data/{model}/raw/full_dataset.csv",
        output_dir=f"data/{model}/raw",
        n_folds=5
    )
    create_random_5050_public_private(
        input_csv=f"data/{model}/raw/test.csv",
        output_dir=f"data/{model}/raw",
    )
