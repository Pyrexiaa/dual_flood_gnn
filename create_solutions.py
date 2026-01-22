import pandas as pd
from pathlib import Path
import re
import numpy as np


def create_solution_files(
    model1_dir,
    model2_dir,
    output_dir=None,
    solution_filename="solution.parquet",
    timestep_threshold=10,
    to_csv=False,
):
    """
    Create solution Parquet file from model directories - FIXED VERSION.

    Key fix: Properly handle dataframe copies and column selection to avoid NaN values.
    """
    model_dirs = {
        1: Path(model1_dir),
        2: Path(model2_dir),
    }

    if output_dir is None:
        output_dir = Path.cwd()
    else:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

    all_data = []

    for model_id, model_dir in model_dirs.items():
        if not model_dir.exists():
            raise ValueError(f"Model directory does not exist: {model_dir}")

        print(f"\nProcessing Model {model_id}: {model_dir}")

        event_folders = sorted(
            [
                d
                for d in model_dir.iterdir()
                if d.is_dir() and d.name.startswith("event_")
            ]
        )

        if not event_folders:
            print(f"  Warning: No event folders found in {model_dir}")
            continue

        print(f"  Found {len(event_folders)} event folders")

        for event_folder in event_folders:
            event_match = re.search(r"event_(\d+)", event_folder.name)
            if not event_match:
                print(f"  Warning: Could not parse event ID from {event_folder.name}")
                continue

            event_id = int(event_match.group(1))

            nodes_1d_path = event_folder / "1d_nodes_dynamic_all.csv"
            if nodes_1d_path.exists():
                try:
                    df_1d_original = pd.read_csv(nodes_1d_path)

                    required_cols = ["timestep", "node_idx", "water_level"]
                    if not all(col in df_1d_original.columns for col in required_cols):
                        print(f"  Warning: Missing columns in {nodes_1d_path}")
                        print(f"    Available: {list(df_1d_original.columns)}")
                        continue

                    df_1d_filtered = df_1d_original[
                        df_1d_original["timestep"] >= timestep_threshold
                    ].copy()

                    if len(df_1d_filtered) > 0:
                        df_1d_final = pd.DataFrame(
                            {
                                "model_id": model_id,
                                "event_id": event_id,
                                "node_type": 1,
                                "node_id": df_1d_filtered["node_idx"].values,
                                "water_level": df_1d_filtered["water_level"].values,
                            }
                        )

                        all_data.append(df_1d_final)

                except Exception as e:
                    print(f"  Error reading {nodes_1d_path}: {e}")
                    import traceback

                    traceback.print_exc()

            nodes_2d_path = event_folder / "2d_nodes_dynamic_all.csv"
            if nodes_2d_path.exists():
                try:
                    df_2d_original = pd.read_csv(nodes_2d_path)

                    required_cols = ["timestep", "node_idx", "water_level"]
                    if not all(col in df_2d_original.columns for col in required_cols):
                        print(f"  Warning: Missing columns in {nodes_2d_path}")
                        print(f"    Available: {list(df_2d_original.columns)}")
                        continue

                    df_2d_filtered = df_2d_original[
                        df_2d_original["timestep"] >= timestep_threshold
                    ].copy()

                    if len(df_2d_filtered) > 0:
                        df_2d_final = pd.DataFrame(
                            {
                                "model_id": model_id,
                                "event_id": event_id,
                                "node_type": 2,
                                "node_id": df_2d_filtered["node_idx"].values,
                                "water_level": df_2d_filtered["water_level"].values,
                            }
                        )

                        all_data.append(df_2d_final)

                except Exception as e:
                    print(f"  Error reading {nodes_2d_path}: {e}")
                    import traceback

                    traceback.print_exc()

    if not all_data:
        raise ValueError("No data was collected from any model directories")

    solution = pd.concat(all_data, ignore_index=True)

    solution = solution.sort_values(
        ["model_id", "event_id", "node_type", "node_id"]
    ).reset_index(drop=True)

    nan_count = solution["water_level"].isna().sum()
    if nan_count > 0:
        print(f"\n⚠️  WARNING: {nan_count} NaN values found in water_level!")
        print("First few rows with NaN:")
        print(solution[solution["water_level"].isna()].head())

    solution_path = output_dir / solution_filename
    if to_csv:
        solution.to_csv(
            solution_path, index=False
        )
    else:
        solution.to_parquet(
            solution_path, index=False, engine="pyarrow", compression="snappy"
        )

    print(f"\n{'=' * 80}")
    print(f"✓ Created solution file: {solution_path}")
    print(f"{'=' * 80}")
    print(f"Total rows: {len(solution):,}")
    print(f"NaN water_level values: {nan_count:,}")
    print(f"Unique models: {solution['model_id'].nunique()}")
    print(f"Unique events: {solution['event_id'].nunique()}")
    print(
        f"Unique nodes (1D): {solution[solution['node_type'] == 0]['node_id'].nunique()}"
    )
    print(
        f"Unique nodes (2D): {solution[solution['node_type'] == 1]['node_id'].nunique()}"
    )

    print("\nBreakdown by model:")
    for model_id in sorted(solution["model_id"].unique()):
        model_data = solution[solution["model_id"] == model_id]
        print(
            f"  Model {model_id}: {len(model_data):,} rows, {model_data['event_id'].nunique()} events"
        )

    print("\nBreakdown by node type:")
    for node_type in sorted(solution["node_type"].unique()):
        type_name = "1D" if node_type == 0 else "2D"
        type_data = solution[solution["node_type"] == node_type]
        print(f"  {type_name} nodes: {len(type_data):,} rows")

    print("\nSample of solution file:")
    print(solution.head(10))
    print(f"{'=' * 80}\n")

    return solution_path


def add_usage_and_row_id(
    solution_path,
    output_path=None,
    public_ratio=0.5,
    random_seed=42,
    id_column_name="row_id",
    to_csv=False
):
    """
    Add 'row_id' (first column) and 'Usage' (last column) to solution file.

    Each event is assigned entirely to either 'Public' or 'Private'.
    Approximately 50% of events are Public, 50% are Private.
    All rows within the same event get the same Usage label.

    Args:
        solution_path: Path to the solution Parquet file
        output_path: Path to save output (default: overwrites input file)
        public_ratio: Target ratio of Public events (default: 0.5)
        random_seed: Random seed for reproducibility (default: 42)
        id_column_name: Name for the ID column (default: 'row_id')

    Returns:
        Path to the updated solution file
    """
    if to_csv:
        solution = pd.read_csv(solution_path)
    else:
        solution = pd.read_parquet(solution_path)

    print(f"Loading solution file: {solution_path}")
    print(f"Original shape: {solution.shape}")
    print(f"Columns: {list(solution.columns)}")

    if id_column_name in solution.columns:
        print(
            f"Warning: '{id_column_name}' column already exists. It will be overwritten."
        )
        solution = solution.drop(columns=[id_column_name])

    if "Usage" in solution.columns:
        print("Warning: 'Usage' column already exists. It will be overwritten.")
        solution = solution.drop(columns=["Usage"])

    np.random.seed(random_seed)

    unique_events = solution[["model_id", "event_id"]].drop_duplicates()
    print(f"\nTotal unique (model_id, event_id) combinations: {len(unique_events)}")

    n_events = len(unique_events)
    n_public = int(n_events * public_ratio)
    n_private = n_events - n_public

    usage_labels = np.array(["Public"] * n_public + ["Private"] * n_private)
    np.random.shuffle(usage_labels)

    unique_events["Usage"] = usage_labels

    solution = solution.merge(
        unique_events[["model_id", "event_id", "Usage"]],
        on=["model_id", "event_id"],
        how="left",
    )

    unlabeled_count = solution["Usage"].isna().sum()
    if unlabeled_count > 0:
        raise ValueError(f"Error: {unlabeled_count} rows were not labeled!")

    solution.insert(0, id_column_name, range(len(solution)))

    cols = [
        id_column_name,
        "model_id",
        "event_id",
        "node_type",
        "node_id",
        "water_level",
        "Usage",
    ]
    solution = solution[cols]

    if output_path is None:
        output_path = solution_path
    else:
        output_path = Path(output_path)

    if to_csv:
        solution.to_csv(
            output_path, index=False
        )
    else:
        solution.to_parquet(
            output_path, index=False, engine="pyarrow", compression="snappy"
        )

    print(f"\n{'=' * 80}")
    print(f"✓ Added '{id_column_name}' and 'Usage' columns to solution file")
    print(f"{'=' * 80}")
    print(f"Saved to: {output_path}")
    print(f"Total rows: {len(solution):,}")
    print(f"Row IDs: 0 to {len(solution) - 1}")

    print("\nUsage Distribution (by rows):")
    usage_counts = solution["Usage"].value_counts()
    for usage, count in usage_counts.items():
        percentage = (count / len(solution)) * 100
        print(f"  {usage}: {count:,} rows ({percentage:.2f}%)")

    print("\nUsage Distribution (by events):")
    event_usage = solution[["model_id", "event_id", "Usage"]].drop_duplicates()
    event_usage_counts = event_usage["Usage"].value_counts()
    for usage, count in event_usage_counts.items():
        percentage = (count / len(event_usage)) * 100
        print(f"  {usage}: {count} events ({percentage:.2f}%)")

    print("\nBreakdown by model:")
    for model_id in sorted(solution["model_id"].unique()):
        model_events = event_usage[event_usage["model_id"] == model_id]
        model_event_usage = model_events["Usage"].value_counts()
        print(f"  Model {model_id}:")
        for usage in ["Public", "Private"]:
            count = model_event_usage.get(usage, 0)
            total = len(model_events)
            percentage = (count / total) * 100 if total > 0 else 0
            print(f"    {usage}: {count}/{total} events ({percentage:.1f}%)")

    print("\nColumn order in solution file:")
    for i, col in enumerate(solution.columns):
        print(f"  {i + 1}. {col}")

    print("\nSample of updated solution file:")
    print(solution.head(10))

    print(f"\n{'=' * 80}")
    print("Verification:")
    print(f"  row_id is first column: {solution.columns[0] == id_column_name}")
    print(f"  row_id is unique: {solution[id_column_name].is_unique}")
    print(f"  All rows have Usage: {solution['Usage'].notna().all()}")
    print(
        f"  Only Public/Private values: {set(solution['Usage'].unique()) == {'Public', 'Private'}}"
    )

    event_consistency = solution.groupby(["model_id", "event_id"])["Usage"].nunique()
    if (event_consistency > 1).any():
        print("  ⚠ Warning: Some events have mixed Usage labels!")
    else:
        print("  Event consistency: ✓ All rows in same event have same label")

    print(f"{'=' * 80}\n")

    return output_path


def add_row_id_and_create_submissions(
    solution_path,
    output_dir=None,
    random_seed=42,
    id_column_name="row_id",
    to_csv=False
):
    """
    Add 'row_id' to solution file and create two Parquet files:
    1. solution.parquet - Full solution with water_level values
    2. sample_submission.parquet - Template with blank water_level column

    Args:
        solution_path: Path to the solution Parquet file
        output_dir: Directory to save output files (default: same as input file)
        random_seed: Random seed for reproducibility (default: 42)
        id_column_name: Name for the ID column (default: 'row_id')

    Returns:
        tuple: (solution_path, sample_submission_path)
    """
    if to_csv:
        solution = pd.read_csv(solution_path)
    else:
        solution = pd.read_parquet(solution_path)

    print(f"Loading solution file: {solution_path}")
    print(f"Original shape: {solution.shape}")
    print(f"Columns: {list(solution.columns)}")

    if id_column_name in solution.columns:
        print(
            f"Warning: '{id_column_name}' column already exists. It will be overwritten."
        )
        solution = solution.drop(columns=[id_column_name])

    if "Usage" in solution.columns:
        print("Warning: 'Usage' column already exists. It will be overwritten.")
        solution = solution.drop(columns=["Usage"])

    np.random.seed(random_seed)

    unique_events = solution[["model_id", "event_id"]].drop_duplicates()
    print(f"\nTotal unique (model_id, event_id) combinations: {len(unique_events)}")

    solution = solution.merge(
        unique_events[["model_id", "event_id"]],
        on=["model_id", "event_id"],
        how="left",
    )

    solution.insert(0, id_column_name, range(len(solution)))

    cols = [
        id_column_name,
        "model_id",
        "event_id",
        "node_type",
        "node_id",
        "water_level",
    ]
    solution = solution[cols]

    if output_dir is None:
        output_dir = Path(solution_path).parent
    else:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

    
    if to_csv:
        solution_output_path = output_dir / "solution_without_usage.csv"
        solution.to_csv(
            solution_output_path, index=False
        )
    else:
        solution_output_path = output_dir / "solution_without_usage.parquet"
        solution.to_parquet(
            solution_output_path, index=False, engine="pyarrow", compression="snappy"
        )

    sample_submission = solution.copy()
    sample_submission["water_level"] = ""
    
    if to_csv:
        sample_submission_path = output_dir / "sample_submission.csv"
        sample_submission.to_csv(
            sample_submission_path, index=False
        )
    else:
        sample_submission_path = output_dir / "sample_submission.parquet"
        sample_submission.to_parquet(
            sample_submission_path, index=False, engine="pyarrow", compression="snappy"
        )

    print(f"\n{'=' * 80}")
    print("✓ Created solution and sample submission files")
    print(f"{'=' * 80}")
    print("\n1. Solution file (with water_level values):")
    print(f"   Path: {solution_output_path}")
    print(f"   Total rows: {len(solution):,}")
    print(f"   Row IDs: 0 to {len(solution) - 1}")

    print("\n2. Sample submission file (blank water_level):")
    print(f"   Path: {sample_submission_path}")
    print(f"   Total rows: {len(sample_submission):,}")

    print("\nColumn order in both files:")
    for i, col in enumerate(solution.columns):
        print(f"  {i + 1}. {col}")

    print("\nSample of solution.parquet (with values):")
    print(solution.head(10))

    print("\nSample of sample_submission.parquet (blank water_level):")
    print(sample_submission.head(10))

    print(f"\n{'=' * 80}")
    print("Verification:")
    print(f"  row_id is first column: {solution.columns[0] == id_column_name}")
    print(f"  row_id is unique: {solution[id_column_name].is_unique}")
    print(
        f"  Both files have same structure: {solution.columns.equals(sample_submission.columns)}"
    )
    print(
        f"  Both files have same number of rows: {len(solution) == len(sample_submission)}"
    )
    print(f"{'=' * 80}\n")

    return solution_output_path, sample_submission_path


if __name__ == "__main__":
    solution_path = create_solution_files(
        model1_dir="/Users/jiayulim/Documents/GitHub/dual_flood_gnn/data/Model1/processed/features_csv/test_edited",
        model2_dir="/Users/jiayulim/Documents/GitHub/dual_flood_gnn/data/Model2/processed/features_csv/test_edited",
        output_dir="kaggle_submission",
        solution_filename="solution.csv",
        timestep_threshold=10,
        to_csv=True,
    )

    add_usage_and_row_id(
        solution_path="kaggle_submission/solution.csv",
        output_path="kaggle_submission/solutions.csv",
        to_csv=True,
    )

    add_row_id_and_create_submissions(
        solution_path="kaggle_submission/solution.csv",
        output_dir="kaggle_submission",
        to_csv=True,
    )

    solution_path = create_solution_files(
        model1_dir="/Users/jiayulim/Documents/GitHub/dual_flood_gnn/data/Model1/processed/features_csv/test_edited",
        model2_dir="/Users/jiayulim/Documents/GitHub/dual_flood_gnn/data/Model2/processed/features_csv/test_edited",
        output_dir="kaggle_submission",
        solution_filename="solution.parquet",
        timestep_threshold=10,
    )

    add_usage_and_row_id(
        solution_path="kaggle_submission/solution.parquet",
        output_path="kaggle_submission/solutions.parquet",
    )

    add_row_id_and_create_submissions(
        solution_path="kaggle_submission/solution.parquet",
        output_dir="kaggle_submission",
    )
