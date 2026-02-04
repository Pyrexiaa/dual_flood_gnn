import pandas as pd
import os
from argparse import ArgumentParser, Namespace
import torch

def remove_timesteps_per_node(df, first_n=0, last_n=0, row_idx_col="row_id"):
    """
    Remove the first N and last N timesteps for each unique node.

    Parameters:
    - df: DataFrame with columns including model_id, event_id, node_type, node_id
    - first_n: Number of timesteps to remove from the beginning for each node
    - last_n: Number of timesteps to remove from the end for each node

    Returns:
    - Trimmed DataFrame
    """
    if first_n == 0 and last_n == 0:
        return df

    # Group by node identifiers and remove timesteps
    trimmed_dfs = []

    for (model_id, event_id, node_type, node_id), group in df.groupby(
        ["model_id", "event_id", "node_type", "node_id"]
    ):
        # Sort by row_id to ensure proper ordering
        group_sorted = group.sort_values(row_idx_col)

        # Remove first_n and last_n rows
        if first_n > 0:
            group_sorted = group_sorted.iloc[first_n:]
        if last_n > 0:
            group_sorted = group_sorted.iloc[:-last_n]

        trimmed_dfs.append(group_sorted)

    # Combine all trimmed groups
    if trimmed_dfs:
        return pd.concat(trimmed_dfs, ignore_index=False)
    else:
        return pd.DataFrame(columns=df.columns)


def check_and_concatenate_events(
    test_csv_model1, test_csv_model2, model1_dir, model2_dir, output_file
):
    """
    Check if all test events have corresponding CSV files and concatenate them.

    Parameters:
    - test_csv_model1: Path to test.csv for model 1
    - test_csv_model2: Path to test.csv for model 2
    - model1_dir: Directory containing prediction CSVs for model 1
    - model2_dir: Directory containing prediction CSVs for model 2
    - output_file: Path for the output concatenated CSV
    """

    # Read test CSVs
    test_model1 = pd.read_csv(test_csv_model1)
    test_model2 = pd.read_csv(test_csv_model2)

    # Get event IDs
    events_model1 = test_model1["Event"].tolist()
    events_model2 = test_model2["Event"].tolist()

    print("=" * 60)
    print("CHECKING MODEL 1")
    print("=" * 60)
    print(f"Test events for Model 1: {events_model1}")

    # Check Model 1
    missing_model1 = []
    found_model1 = []
    for event in events_model1:
        filepath = os.path.join(model1_dir, f"predictions_event_{event}.csv")
        if os.path.exists(filepath):
            found_model1.append(event)
            print(f"✓ Found: predictions_event_{event}.csv")
        else:
            missing_model1.append(event)
            print(f"✗ Missing: predictions_event_{event}.csv")

    print(f"\nModel 1 Summary: {len(found_model1)}/{len(events_model1)} files found")
    if missing_model1:
        print(f"Missing events: {missing_model1}")

    print("\n" + "=" * 60)
    print("CHECKING MODEL 2")
    print("=" * 60)
    print(f"Test events for Model 2: {events_model2}")

    # Check Model 2
    missing_model2 = []
    found_model2 = []
    for event in events_model2:
        filepath = os.path.join(model2_dir, f"predictions_event_{event}.csv")
        if os.path.exists(filepath):
            found_model2.append(event)
            print(f"✓ Found: predictions_event_{event}.csv")
        else:
            missing_model2.append(event)
            print(f"✗ Missing: predictions_event_{event}.csv")

    print(f"\nModel 2 Summary: {len(found_model2)}/{len(events_model2)} files found")
    if missing_model2:
        print(f"Missing events: {missing_model2}")

    # Concatenate if all files are present
    if missing_model1 or missing_model2:
        print("\n" + "=" * 60)
        print("WARNING: Cannot concatenate - some files are missing!")
        print("=" * 60)
        return None

    print("\n" + "=" * 60)
    print("CONCATENATING FILES")
    print("=" * 60)

    # Concatenate Model 1 events
    dfs = []
    for event in events_model1:
        filepath = os.path.join(model1_dir, f"predictions_event_{event}.csv")
        df = pd.read_csv(filepath)

        # Remove timesteps for Model 1: remove first 8 and last 48 for each node
        df_trimmed = remove_timesteps_per_node(df, first_n=8, last_n=48)
        df_trimmed["model_id"] = 1

        dfs.append(df_trimmed)
        print(
            f"Added Model 1 - Event {event}: {len(df)} rows → {len(df_trimmed)} rows (removed first 8 and last 48 timesteps per node)"
        )

    # Concatenate Model 2 events
    for event in events_model2:
        filepath = os.path.join(model2_dir, f"predictions_event_{event}.csv")
        df = pd.read_csv(filepath)

        # Remove timesteps for Model 2: remove first 8 and last 36 for each node
        df_trimmed = remove_timesteps_per_node(df, first_n=8, last_n=36)
        df_trimmed["model_id"] = 2

        dfs.append(df_trimmed)
        print(
            f"Added Model 2 - Event {event}: {len(df)} rows → {len(df_trimmed)} rows (removed first 8 and last 36 timesteps per node)"
        )

    # Combine all dataframes
    combined_df = pd.concat(dfs, ignore_index=False)

    # Set row_id as index
    combined_df.set_index("row_id", inplace=True)

    combined_df_rearranged = rearrange_by_node(combined_df)

    # Save to output file
    combined_df_rearranged.to_csv(output_file)

    print(f"\n✓ Successfully concatenated {len(dfs)} files")
    print(f"✓ Total rows: {len(combined_df_rearranged)}")
    print(f"✓ Output saved to: {output_file}")

    return combined_df_rearranged


def check_timesteps_per_node(csv_file):
    """
    Check the number of timesteps for each node combination.

    Parameters:
    - csv_file: Path to the CSV file

    Returns:
    - DataFrame with timestep counts per node
    """
    # Read the CSV file
    df = pd.read_csv(csv_file)

    # Count rows (timesteps) for each unique node combination
    timesteps = (
        df.groupby(["model_id", "event_id", "node_type", "node_id"])
        .size()
        .reset_index(name="timesteps")
    )

    print("=" * 80)
    print("TIMESTEPS PER NODE")
    print("=" * 80)
    print(timesteps.to_string(index=False))
    print("=" * 80)

    # Check if all nodes have the same number of timesteps
    unique_timesteps = timesteps["timesteps"].unique()

    if len(unique_timesteps) == 1:
        print(f"\n✓ All nodes have the same number of timesteps: {unique_timesteps[0]}")
    else:
        print("\n✗ WARNING: Inconsistent timestep counts detected!")
        print(f"   Different timestep counts found: {sorted(unique_timesteps)}")

        # Show which nodes have different counts
        print("\n   Breakdown by timestep count:")
        for ts_count in sorted(unique_timesteps):
            nodes_with_count = timesteps[timesteps["timesteps"] == ts_count]
            print(f"   - {ts_count} timesteps: {len(nodes_with_count)} nodes")

    print(f"\nTotal unique nodes: {len(timesteps)}")
    print(f"Total rows in file: {len(df)}")

    return timesteps


def find_inconsistent_nodes(csv_file):
    """
    Find nodes that don't have the expected number of timesteps.

    Parameters:
    - csv_file: Path to the CSV file

    Returns:
    - DataFrame with nodes that have inconsistent timestep counts
    """
    # Read the CSV file
    df = pd.read_csv(csv_file)

    # Count timesteps per node
    timesteps = (
        df.groupby(["model_id", "event_id", "node_type", "node_id"])
        .size()
        .reset_index(name="timesteps")
    )

    # Find the most common timestep count (expected count)
    expected_count = timesteps["timesteps"].mode()[0]

    # Find nodes that don't match the expected count
    inconsistent = timesteps[timesteps["timesteps"] != expected_count]

    if len(inconsistent) > 0:
        print("=" * 80)
        print(f"NODES WITH INCONSISTENT TIMESTEPS (Expected: {expected_count})")
        print("=" * 80)
        print(inconsistent.to_string(index=False))
        print("=" * 80)
    else:
        print(f"✓ All nodes have the expected {expected_count} timesteps")

    return inconsistent


def get_timestep_summary(csv_file):
    """
    Get a summary of timestep counts across different groupings.

    Parameters:
    - csv_file: Path to the CSV file
    """
    # Read the CSV file
    df = pd.read_csv(csv_file)

    print("=" * 80)
    print("SUMMARY STATISTICS")
    print("=" * 80)

    # Count by model_id and event_id
    print("\nTimesteps per (model_id, event_id):")
    model_event_counts = df.groupby(
        ["model_id", "event_id", "node_type", "node_id"]
    ).size()
    print(f"  Min: {model_event_counts.min()}")
    print(f"  Max: {model_event_counts.max()}")
    print(f"  Mean: {model_event_counts.mean():.2f}")
    print(f"  Median: {model_event_counts.median():.0f}")

    # Count unique nodes
    print(
        f"\nUnique nodes (model_id, event_id, node_type, node_id): {len(model_event_counts)}"
    )

    # Show unique values
    print(f"\nUnique model_ids: {sorted(df['model_id'].unique())}")
    print(f"Unique event_ids: {sorted(df['event_id'].unique())}")
    print(f"Unique node_types: {sorted(df['node_type'].unique())}")
    print(f"Unique node_ids: {sorted(df['node_id'].unique())}")

    print("=" * 80)


def rearrange_by_node(csv_file, output_file=None, verbose=True, include_gt=False):
    """
    Rearrange CSV by sorting on model_id, event_id, node_type, node_id, row_id.
    This groups all rows for the same node together and resets row_id to start from 0.

    OPTIMIZED: Only prints sample of data, not entire DataFrame.

    Parameters:
    - csv_file: Path to the input CSV file (or DataFrame)
    - output_file: Path to save the rearranged CSV (optional)
    - verbose: If True, print detailed summary (default: True)

    Returns:
    - Rearranged DataFrame
    """
    # Read the CSV file if it's a string path, otherwise use DataFrame directly
    if isinstance(csv_file, str):
        if verbose:
            print(f"\nReading CSV file: {csv_file}")
        df = pd.read_csv(csv_file)
    else:
        df = csv_file.copy()

    if verbose:
        print(f"  Total rows to sort: {len(df):,}")
        print(f"  Columns before sort: {df.columns.tolist()}")

    # Sort by model_id, event_id, node_type, node_id, then row_id
    if verbose:
        print("\nSorting data...")
    df_sorted = df.sort_values(
        by=["model_id", "event_id", "node_type", "node_id", "row_id"], ascending=True
    ).reset_index(drop=True)

    # Reset row_id to start from 0 and increase sequentially
    df_sorted["row_id"] = range(len(df_sorted))

    if include_gt:
        if verbose:
            print("\nAdding ground truth from solutions.csv...")
            print(f"  Columns before adding GT: {df_sorted.columns.tolist()}")
            print(f"  df_sorted shape: {df_sorted.shape}")

        gt = pd.read_csv("kaggle_submissions/solutions.csv")

        if verbose:
            print(f"  Ground truth shape: {gt.shape}")
            print(f"  Ground truth columns: {gt.columns.tolist()}")

        assert df_sorted.shape[0] == gt.shape[0], (
            f"Ground truth has {len(gt)} rows but dataframe has {len(df_sorted)} rows"
        )

        # DEBUG: Check if assignment is working
        if verbose:
            print("\n  Attempting to add target_water_level column...")
            print(
                f"  GT water_level - Min: {gt['water_level'].min():.4f}, Max: {gt['water_level'].max():.4f}"
            )

        # Try explicit copy to avoid any view issues
        df_sorted = df_sorted.copy()
        df_sorted["target_water_level"] = gt[
            "water_level"
        ].values  # Use .values to avoid index alignment issues

        if verbose:
            print(f"  Columns after adding GT: {df_sorted.columns.tolist()}")
            print(
                f"  'target_water_level' in columns: {'target_water_level' in df_sorted.columns}"
            )

            if "target_water_level" in df_sorted.columns:
                print("  ✓ Added target_water_level column")
                print(f"    Min: {df_sorted['target_water_level'].min():.4f}")
                print(f"    Max: {df_sorted['target_water_level'].max():.4f}")
                print(f"    Non-zero: {(df_sorted['target_water_level'] != 0).sum():,}")
                print(
                    f"    First 10 values: {df_sorted['target_water_level'].head(10).tolist()}"
                )
            else:
                print("  ❌ ERROR: target_water_level column was NOT added!")

    if verbose:
        # Display summary - ONLY FIRST 20 ROWS, NOT ENTIRE DATAFRAME
        print("\n" + "=" * 80)
        print("REARRANGED DATA BY NODE")
        print("=" * 80)
        print(f"Columns in df_sorted: {df_sorted.columns.tolist()}")
        print("\nFirst 20 rows:")
        print(df_sorted.head(20).to_string(index=False))
        print("\nLast 20 rows:")
        print(df_sorted.tail(20).to_string(index=False))
        print("\n" + "=" * 80)

        # Count rows per node
        node_counts = (
            df_sorted.groupby(["model_id", "event_id", "node_type", "node_id"])
            .size()
            .reset_index(name="count")
        )

        print("\nROWS PER NODE (first 20 nodes):")
        print("-" * 40)
        print(node_counts.head(20).to_string(index=False))
        print("-" * 40)
        print(f"Total unique nodes: {len(node_counts):,}")
        print(f"Total rows: {len(df_sorted):,}")
        print(f"Average rows per node: {len(df_sorted) / len(node_counts):.1f}")

    # Save to file if specified
    if output_file:
        if verbose:
            print(f"\nSaving to file: {output_file}")
            print(f"  Columns being saved: {df_sorted.columns.tolist()}")
        df_sorted.to_csv(output_file, index=False)
        if verbose:
            print(f"✓ Saved rearranged data to: {output_file}")

            # Verify the saved file
            print("\nVerifying saved file...")
            df_check = pd.read_csv(output_file, nrows=5)
            print(f"  Columns in saved file: {df_check.columns.tolist()}")
            if include_gt and "target_water_level" in df_check.columns:
                print("  ✓ target_water_level is in the saved file!")
            elif include_gt:
                print("  ❌ target_water_level is NOT in the saved file!")

    return df_sorted


def combine_model_predictions(
    model1_csv, model2_csv, output_file=None, include_gt=False
):
    """
    Combine two model prediction CSV files into a unified format.

    Original format:
    - sample_idx, event_id, node_id, timestep, node_type, target_water_level,
      predicted_water_level, target_water_level_normalized, predicted_water_level_normalized

    Output format:
    - row_id, model_id, event_id, node_type, node_id, water_level

    Transformations:
    - row_id: Sequential from 0 (after combining both files)
    - model_id: 1 for model1_csv, 2 for model2_csv
    - event_id: From original event_id
    - node_type: 1 if original was 0, 2 if original was 1
    - node_id: From original node_id
    - water_level: From predicted_water_level

    Parameters:
    - model1_csv: Path to model 1 predictions CSV
    - model2_csv: Path to model 2 predictions CSV
    - output_file: Path to save combined CSV (optional)

    Returns:
    - Combined DataFrame
    """

    print("=" * 80)
    print("COMBINING MODEL PREDICTIONS")
    print("=" * 80)

    # Read both CSV files
    print(f"\nReading {model1_csv}...")
    df1 = pd.read_csv(model1_csv)
    print(f"  Rows: {len(df1)}")

    print(f"\nReading {model2_csv}...")
    df2 = pd.read_csv(model2_csv)
    print(f"  Rows: {len(df2)}")

    # Transform model 1
    if include_gt:
        df1_transformed = pd.DataFrame(
            {
                "sample_idx": df1["sample_idx"],
                "timestep": df1["timestep"],
                "model_id": 1,
                "event_id": df1["event_id"],
                "node_type": df1["node_type"] + 1,  # 0 -> 1, 1 -> 2
                "node_id": df1["node_id"],
                "water_level": df1["predicted_water_level"],
            }
        )
    else:
        df1_transformed = pd.DataFrame(
            {
                "sample_idx": df1["sample_idx"],
                "model_id": 1,
                "event_id": df1["event_id"],
                "node_type": df1["node_type"] + 1,  # 0 -> 1, 1 -> 2
                "node_id": df1["node_id"],
                "water_level": df1["predicted_water_level"],
            }
        )
    df1_trimmed = remove_timesteps_per_node(
        df1_transformed, first_n=4, last_n=0, row_idx_col="sample_idx"
    )
    df1_trimmed = df1_trimmed.reset_index(drop=True)

    # Transform model 2
    if include_gt:
        df2_transformed = pd.DataFrame(
            {
                "sample_idx": df2["sample_idx"],
                "timestep": df2["timestep"],
                "model_id": 2,
                "event_id": df2["event_id"],
                "node_type": df2["node_type"] + 1,  # 0 -> 1, 1 -> 2
                "node_id": df2["node_id"],
                "water_level": df2["predicted_water_level"],
            }
        )
    else:
        df2_transformed = pd.DataFrame(
            {
                "sample_idx": df2["sample_idx"],
                "model_id": 2,
                "event_id": df2["event_id"],
                "node_type": df2["node_type"] + 1,  # 0 -> 1, 1 -> 2
                "node_id": df2["node_id"],
                "water_level": df2["predicted_water_level"],
            }
        )
    df2_trimmed = remove_timesteps_per_node(
        df2_transformed, first_n=4, last_n=0, row_idx_col="sample_idx"
    )
    df2_trimmed = df2_trimmed.reset_index(drop=True)

    # Combine both dataframes
    df_combined = pd.concat([df1_trimmed, df2_trimmed], ignore_index=True)

    # Add row_id starting from 0
    df_combined.insert(0, "row_id", range(len(df_combined)))

    # Reorder columns to match specification
    if include_gt:
        df_combined = df_combined[
            [
                "row_id",
                "timestep",
                "model_id",
                "event_id",
                "node_type",
                "node_id",
                "water_level",
            ]
        ]
    else:
        df_combined = df_combined[
            ["row_id", "model_id", "event_id", "node_type", "node_id", "water_level"]
        ]

    print("\n" + "=" * 80)
    print("COMBINED DATA SUMMARY")
    print("=" * 80)
    print(f"Total rows: {len(df_combined)}")
    print(f"  Model 1: {len(df1_transformed)} rows")
    print(f"  Model 2: {len(df2_transformed)} rows")
    print("\nNode type distribution:")
    print(df_combined["node_type"].value_counts().sort_index().to_string())
    print("\nFirst 10 rows:")
    print(df_combined.head(10).to_string(index=False))

    # Save to temporary file before rearranging
    if output_file:
        if include_gt:
            temp_file = output_file.replace(".csv", "_temp_gt.csv")
        else:
            temp_file = output_file.replace(".csv", "_temp.csv")
        df_combined.to_csv(temp_file, index=False)
        print(f"\n✓ Saved combined data to temporary file: {temp_file}")

    return df_combined


def combine_and_rearrange(model1_csv, model2_csv, output_file, include_gt=False):
    """
    Complete pipeline: Combine two model CSVs and rearrange by node.

    Parameters:
    - model1_csv: Path to model 1 predictions CSV
    - model2_csv: Path to model 2 predictions CSV
    - output_file: Path to save final rearranged CSV

    Returns:
    - Final rearranged DataFrame
    """
    # Step 1: Combine the files
    df_combined = combine_model_predictions(
        model1_csv, model2_csv, include_gt=include_gt
    )

    # Step 2: Rearrange by node
    df_final = rearrange_by_node(df_combined, output_file, include_gt=include_gt)

    print("\n" + "=" * 80)
    print("PIPELINE COMPLETE")
    print("=" * 80)
    print(f"✓ Combined {model1_csv} and {model2_csv}")
    print("✓ Rearranged by node")
    print(f"✓ Saved to {output_file}")

    return df_final

def parse_args() -> Namespace:
    parser = ArgumentParser(description="")
    parser.add_argument(
        "--model1_test_csv", type=str, required=True, help="Path to model1 test csv file"
    )
    parser.add_argument(
        "--model2_test_csv", type=str, required=True, help="Path to model2 test csv file"
    )
    parser.add_argument(
        "--model1_saved_event_dir", type=str, required=True, help="Path to model1 saved predictions event directory"
    )
    parser.add_argument(
        "--model2_saved_event_dir", type=str, required=True, help="Path to model2 saved predictions event directory"
    )
    parser.add_argument(
        "--timestep_to_remove", type=str, required=True, help="Remove the first 10 timesteps, if sliding window length is 1, then 8 of them should be removed. If it's 2, then 7 of them should be removed."
    )
    parser.add_argument(
        "--output_file", type=str, required=True, help="Path to concatenated saved predictions"
    )
    parser.add_argument(
        "--device",
        type=str,
        default=("cuda" if torch.cuda.is_available() else "cpu"),
        help="Device to run on",
    )
    return parser.parse_args()


# Example usage:
if __name__ == "__main__":
    args = parse_args()
    # Update these paths according to your directory structure
    # test_csv_model1 = 'kaggle_submissions/model1_test.csv'
    # test_csv_model2 = 'kaggle_submissions/model2_test.csv'
    # model1_dir = 'kaggle_submissions/model1'
    # model2_dir = 'kaggle_submissions/model2'
    # output_file = "combined_node_only_predictions"

    result = check_and_concatenate_events(
        args.model1_test_csv,
        args.model2_test_csv,
        args.model1_saved_event_dir,
        args.model2_saved_event_dir,
        args.output_file
    )

    # if result is not None:
    #     print("\nFirst few rows of combined data:")
    #     print(result.head(10))

    # rearrange_by_node(f"{output_file}.csv", f"{output_file}_rearranged.csv")

    # csv_file = "predictions_event_5.csv"
    # # Method 1: Check all timesteps per node
    # print("\nMethod 1: Check timesteps for all nodes")
    # timesteps_df = check_timesteps_per_node(csv_file)

    # print("\n" + "="*80 + "\n")

    # # Method 2: Find inconsistent nodes
    # print("Method 2: Find nodes with inconsistent timestep counts")
    # inconsistent_df = find_inconsistent_nodes(csv_file)

    # print("\n" + "="*80 + "\n")

    # # Method 3: Get summary statistics
    # print("Method 3: Summary statistics")
    # get_timestep_summary(csv_file)
