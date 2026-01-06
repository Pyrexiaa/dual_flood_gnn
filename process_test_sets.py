from glob import glob
import pandas as pd
from typing import Dict, List, Optional, Union, Set
import shutil
from pathlib import Path


def get_different_numbers(
    csv1_path: str, csv2_path: str, column_name: str
) -> List[Union[int, float]]:
    """
    Returns numbers that appear in only one of the two CSV files
    (symmetric difference).
    """
    df1 = pd.read_csv(csv1_path)
    df2 = pd.read_csv(csv2_path)

    set1 = set(df1[column_name].dropna())
    set2 = set(df2[column_name].dropna())

    diff = set1.symmetric_difference(set2)
    return sorted(diff)


def copy_csv_excluding_integers(
    input_csv: str,
    output_csv: str,
    exclude_values: List[int],
    column_name: str = "Event",
):
    """
    Copy a CSV file while excluding rows where column_name
    contains any value in exclude_values.
    """

    df = pd.read_csv(input_csv)

    # Ensure column is numeric for safe comparison
    df[column_name] = pd.to_numeric(df[column_name], errors="coerce")

    filtered_df = df[~df[column_name].isin(exclude_values)]

    filtered_df.to_csv(output_csv, index=False)


def copy_dir_excluding_csv_files(
    csv_path: str,
    source_dir: str,
    target_dir: str,
    filepath_column: str = "HECRAS_Filepath",
):
    """
    Copy all files from source_dir to target_dir,
    excluding files listed in csv_path under filepath_column.
    Both the .hdf file and the same filename without extension are excluded.
    """

    source_dir = Path(source_dir)
    target_dir = Path(target_dir)
    target_dir.mkdir(parents=True, exist_ok=True)

    # --- Read CSV ---
    df = pd.read_csv(csv_path)

    # --- Build exclusion set ---
    exclude_files: Set[str] = set()

    for fp in df[filepath_column].dropna():
        p = Path(fp)
        base = p.stem  # BeaverLAKE_Pipe_Mod.p60
        exclude_files.add(p.name)  # with extension
        exclude_files.add(base)  # without extension

    # --- Copy files ---
    copied = 0
    skipped = 0

    for file in source_dir.iterdir():
        if file.is_file():
            if file.name in exclude_files:
                skipped += 1
                continue

            shutil.copy2(file, target_dir / file.name)
            copied += 1

    print("Copy complete.")
    print(f"Copied:  {copied} files")
    print(f"Skipped: {skipped} files")


def copy_dir_excluding_folders_with_integers(
    source_dir: str, target_dir: str, exclude_integers: List[int]
):
    """
    Copy all files and folders from source_dir to target_dir,
    excluding any folder whose name contains an integer
    from exclude_integers.
    """

    source_dir = Path(source_dir)
    target_dir = Path(target_dir)
    target_dir.mkdir(parents=True, exist_ok=True)

    exclude_strs = [str(i) for i in exclude_integers]

    for item in source_dir.iterdir():
        # --- Exclude folders by name match ---
        if item.is_dir():
            if any(s in item.name for s in exclude_strs):
                continue
            shutil.copytree(item, target_dir / item.name)

        # --- Copy files at root level ---
        elif item.is_file():
            shutil.copy2(item, target_dir / item.name)


def copy_event_csvs_with_selective_timesteps(
    source_events: Union[str, Path],
    target_events: Union[str, Path],
    column_timestep_limits: Dict[str, int],
    fill_value: Optional[any] = None,
    csv_files: Optional[List[str]] = None,
    timestep_col: str = "timestep",
    id_cols: Optional[Dict[str, str]] = None,
):
    """
    Copy CSV files from multiple event folders while keeping only certain timesteps
    for specified columns. Each node/edge keeps the first N timesteps.

    Args:
        source_events: Path pattern to source event folders (e.g., 'data/test/event_*/')
        target_events: Path pattern to target event folders (e.g., 'data/test_edited/event_*')
        column_timestep_limits: Dict mapping column names to max timestep (0-based)
                               Example: {'flow': 9} keeps timesteps 0-9 for flow column
        fill_value: Value to fill excluded cells with (default: None/NaN)
        csv_files: List of CSV filenames to process. If None, uses default list.
        timestep_col: Name of the timestep column (default: 'timestep')
        id_cols: Dict mapping CSV filename to ID column name
                Example: {'1d_nodes_dynamic_all.csv': 'node_idx',
                         '1d_edges_dynamic_all.csv': 'edge_idx'}
                If None, auto-detects based on filename

    Example:
        # Keep first 10 timesteps (0-9) for each node/edge
        copy_event_csvs_with_selective_timesteps(
            source_events='data/Model1/test/event_*/',
            target_events='data/Model1/test_edited/event_*',
            column_timestep_limits={'flow': 9, 'velocity': 9, 'water_level': 9}
        )
    """
    # Default CSV files to process
    if csv_files is None:
        csv_files = [
            "1d_nodes_dynamic_all.csv",
            "1d_edges_dynamic_all.csv",
            "2d_nodes_dynamic_all.csv",
            "2d_edges_dynamic_all.csv",
        ]

    # Default ID column mapping if not provided
    if id_cols is None:
        id_cols = {
            "1d_nodes_dynamic_all.csv": "node_idx",
            "1d_edges_dynamic_all.csv": "edge_idx",
            "2d_nodes_dynamic_all.csv": "node_idx",
            "2d_edges_dynamic_all.csv": "edge_idx",
        }

    # Convert to Path objects and handle wildcards
    source_pattern = str(source_events)

    # Find all event folders matching the pattern
    if "*" in source_pattern:
        event_folders = sorted(glob(source_pattern))
    else:
        event_folders = [source_pattern]

    if not event_folders:
        print(f"❌ No event folders found matching pattern: {source_pattern}")
        return

    print(f"Found {len(event_folders)} event folders to process")
    print("=" * 80)

    # Process each event folder
    processed_events = 0
    processed_files = 0

    for source_event_path in event_folders:
        source_event = Path(source_event_path)

        if not source_event.exists() or not source_event.is_dir():
            print(f"⚠️  Skipping non-existent folder: {source_event}")
            continue

        # Extract event name (e.g., 'event_1')
        event_name = source_event.name

        # Construct target event folder path
        target_base = Path(str(target_events).replace("event_*", "").rstrip("/"))
        target_event = target_base / event_name

        # Create target event folder
        target_event.mkdir(parents=True, exist_ok=True)

        print(f"\nProcessing: {event_name}")
        print(f"  Source: {source_event}")
        print(f"  Target: {target_event}")

        # Process each CSV file
        event_files_processed = 0

        for csv_filename in csv_files:
            source_csv = source_event / csv_filename
            target_csv = target_event / csv_filename

            if not source_csv.exists():
                print(f"  ⚠️  File not found: {csv_filename}")
                continue

            try:
                # Read the CSV
                df = pd.read_csv(source_csv)
                original_rows = len(df)

                # Determine the ID column for this CSV
                id_col = id_cols.get(csv_filename, None)

                # Auto-detect ID column if not specified
                if id_col is None or id_col not in df.columns:
                    possible_id_cols = ["node_idx", "edge_idx", "node_id", "edge_id"]
                    for col in possible_id_cols:
                        if col in df.columns:
                            id_col = col
                            break

                if timestep_col not in df.columns:
                    print(f"  ⚠️  {csv_filename}: No '{timestep_col}' column found")
                    # Copy as-is if no timestep column
                    df.to_csv(target_csv, index=False)
                    continue

                if id_col is None or id_col not in df.columns:
                    print(
                        f"  ⚠️  {csv_filename}: No ID column found (tried: {list(id_cols.values())})"
                    )
                    # Copy as-is if no ID column
                    df.to_csv(target_csv, index=False)
                    continue

                columns_modified = []

                # Process each column with timestep limits
                for col_name, max_timestep in column_timestep_limits.items():
                    if col_name in df.columns:
                        # Create mask: keep values where timestep <= max_timestep
                        # Set to fill_value where timestep > max_timestep
                        mask = df[timestep_col] > max_timestep
                        if mask.any():
                            df.loc[mask, col_name] = fill_value
                            columns_modified.append(col_name)

                # Save to target
                df.to_csv(target_csv, index=False)

                # Calculate statistics
                unique_ids = df[id_col].nunique()
                unique_timesteps = df[timestep_col].nunique()
                rows_per_id = len(df) / unique_ids if unique_ids > 0 else 0

                if columns_modified:
                    print(f"  ✓ {csv_filename}:")
                    print(
                        f"      Total rows: {len(df)}, IDs: {unique_ids}, Timesteps: {unique_timesteps}"
                    )
                    print(f"      Modified columns: {columns_modified}")
                    print(f"      Kept timesteps 0-{max_timestep} for each {id_col}")
                else:
                    print(
                        f"  ✓ {csv_filename}: {len(df)} rows, no matching columns found"
                    )

                event_files_processed += 1
                processed_files += 1

            except Exception as e:
                print(f"  ❌ Error processing {csv_filename}: {str(e)}")
                import traceback

                traceback.print_exc()

        if event_files_processed > 0:
            processed_events += 1

        # Copy any other files in the event folder
        for item in source_event.iterdir():
            if item.is_file() and item.name not in csv_files:
                try:
                    shutil.copy2(item, target_event / item.name)
                except Exception as e:
                    print(f"  ⚠️  Could not copy {item.name}: {str(e)}")

    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)
    print(f"Total events processed: {processed_events}/{len(event_folders)}")
    print(f"Total CSV files processed: {processed_files}")
    print(f"Target location: {target_base}")
    print("=" * 80)


def inspect_csv_structure(csv_path: Union[str, Path], show_sample: bool = True):
    """
    Inspect the structure of a CSV file to understand its format.
    """
    df = pd.read_csv(csv_path)

    print(f"File: {Path(csv_path).name}")
    print(f"  Rows: {len(df)}")
    print(f"  Columns: {list(df.columns)}")

    # Detect ID and timestep columns
    id_cols = [col for col in df.columns if "idx" in col.lower() or "id" in col.lower()]
    timestep_cols = [
        col for col in df.columns if "timestep" in col.lower() or "time" in col.lower()
    ]

    if id_cols:
        print(f"  Potential ID columns: {id_cols}")
        for id_col in id_cols:
            print(f"    {id_col}: {df[id_col].nunique()} unique values")

    if timestep_cols:
        print(f"  Potential timestep columns: {timestep_cols}")
        for ts_col in timestep_cols:
            print(
                f"    {ts_col}: min={df[ts_col].min()}, max={df[ts_col].max()}, unique={df[ts_col].nunique()}"
            )

    if show_sample:
        print("\n  First few rows:")
        print(df.head(10).to_string(index=False))

    print()

def remove_last_timesteps_from_events(
    source_events: Union[str, Path],
    target_events: Union[str, Path],
    n_timesteps_to_remove: int,
    timesteps_filename: str = "timesteps.csv",
    timestep_col: str = "timestep",
    csv_files: Optional[List[str]] = None,
    copy_other_files: bool = True,
):
    """
    Remove the last N timesteps from all CSV files in event folders.
    Reads timesteps.csv to determine which timestep indices to remove,
    then removes all rows with those timestep values from other CSV files.

    Args:
        source_events: Path pattern to source event folders (e.g., 'data/test/event_*/')
        target_events: Path pattern to target event folders (e.g., 'data/test_trimmed/event_*')
        n_timesteps_to_remove: Number of last timesteps to remove (e.g., 48)
        timesteps_filename: Name of the timesteps reference file (default: 'timesteps.csv')
        timestep_col: Name of the timestep column in data files (default: 'timestep')
        csv_files: List of CSV filenames to process. If None, processes all CSVs except timesteps.csv
        copy_other_files: Whether to copy non-CSV files to target (default: True)

    Example:
        # Remove last 48 timesteps from all event CSVs
        remove_last_timesteps_from_events(
            source_events='data/Model1/test/event_*/',
            target_events='data/Model1/test_trimmed/event_*',
            n_timesteps_to_remove=48
        )
        
        # This will:
        # 1. Read timesteps.csv to find the last 48 timestep indices
        # 2. Remove all rows with those timestep values from all other CSVs
        # 3. Save trimmed files to target directory
    """
    # Convert to Path objects and handle wildcards
    source_pattern = str(source_events)

    # Find all event folders matching the pattern
    if "*" in source_pattern:
        event_folders = sorted(glob(source_pattern))
    else:
        event_folders = [source_pattern]

    if not event_folders:
        print(f"❌ No event folders found matching pattern: {source_pattern}")
        return

    print(f"Found {len(event_folders)} event folders to process")
    print(f"Will remove last {n_timesteps_to_remove} timesteps from each event")
    print("=" * 80)

    # Statistics
    processed_events = 0
    processed_files = 0
    total_rows_removed = 0

    for source_event_path in event_folders:
        source_event = Path(source_event_path)

        if not source_event.exists() or not source_event.is_dir():
            print(f"⚠️  Skipping non-existent folder: {source_event}")
            continue

        # Extract event name (e.g., 'event_1')
        event_name = source_event.name

        # Construct target event folder path
        target_base = Path(str(target_events).replace("event_*", "").rstrip("/"))
        target_event = target_base / event_name

        # Create target event folder
        target_event.mkdir(parents=True, exist_ok=True)

        print(f"\nProcessing: {event_name}")
        print(f"  Source: {source_event}")
        print(f"  Target: {target_event}")

        # Read timesteps.csv to determine which timesteps to remove
        timesteps_csv = source_event / timesteps_filename
        
        if not timesteps_csv.exists():
            print(f"  ❌ {timesteps_filename} not found, skipping event")
            continue

        try:
            # Read timesteps file
            df_timesteps = pd.read_csv(timesteps_csv)
            
            # Identify the timestep index column (usually first column or named 'timestep_idx')
            possible_idx_cols = ['timestep_idx', 'timestep', 'time_idx', 'idx']
            timestep_idx_col = None
            
            for col in possible_idx_cols:
                if col in df_timesteps.columns:
                    timestep_idx_col = col
                    break
            
            # If not found, use first column
            if timestep_idx_col is None:
                timestep_idx_col = df_timesteps.columns[0]
            
            # Get total number of timesteps
            total_timesteps = len(df_timesteps)
            
            if n_timesteps_to_remove >= total_timesteps:
                print(f"  ⚠️  Warning: Trying to remove {n_timesteps_to_remove} timesteps but only {total_timesteps} exist")
                print(f"  Will remove all timesteps except the first one")
                n_to_remove = max(0, total_timesteps - 1)
            else:
                n_to_remove = n_timesteps_to_remove
            
            # Determine which timestep indices to remove (last N)
            timesteps_to_remove = set(df_timesteps[timestep_idx_col].iloc[-n_to_remove:].values)
            timesteps_to_keep = set(df_timesteps[timestep_idx_col].iloc[:-n_to_remove].values)
            
            print(f"  Timesteps info:")
            print(f"    Total timesteps: {total_timesteps}")
            print(f"    Keeping: {len(timesteps_to_keep)} (indices {min(timesteps_to_keep)}-{max(timesteps_to_keep)})")
            print(f"    Removing: {len(timesteps_to_remove)} (indices {min(timesteps_to_remove) if timesteps_to_remove else 'N/A'}-{max(timesteps_to_remove) if timesteps_to_remove else 'N/A'})")
            
            # Save trimmed timesteps.csv
            df_timesteps_trimmed = df_timesteps[df_timesteps[timestep_idx_col].isin(timesteps_to_keep)]
            df_timesteps_trimmed.to_csv(target_event / timesteps_filename, index=False)
            
        except Exception as e:
            print(f"  ❌ Error reading {timesteps_filename}: {str(e)}")
            import traceback
            traceback.print_exc()
            continue

        # Get list of CSV files to process
        if csv_files is None:
            # Process all CSV files except timesteps.csv
            all_csv_files = [f.name for f in source_event.glob("*.csv") if f.name != timesteps_filename]
        else:
            all_csv_files = [f for f in csv_files if f != timesteps_filename]

        # Process each CSV file
        event_files_processed = 0
        event_rows_removed = 0

        for csv_filename in all_csv_files:
            source_csv = source_event / csv_filename
            target_csv = target_event / csv_filename

            if not source_csv.exists():
                continue

            try:
                # Read the CSV
                df = pd.read_csv(source_csv)
                original_rows = len(df)

                # Check if file has timestep column
                if timestep_col not in df.columns:
                    # No timestep column, copy as-is
                    df.to_csv(target_csv, index=False)
                    print(f"  ℹ️  {csv_filename}: No '{timestep_col}' column, copied as-is")
                    continue

                # Remove rows with timesteps in the removal set
                df_trimmed = df[df[timestep_col].isin(timesteps_to_keep)].copy()
                rows_removed = original_rows - len(df_trimmed)

                # Save trimmed CSV
                df_trimmed.to_csv(target_csv, index=False)

                print(f"  ✓ {csv_filename}:")
                print(f"      Original: {original_rows:,} rows")
                print(f"      Trimmed:  {len(df_trimmed):,} rows")
                print(f"      Removed:  {rows_removed:,} rows")

                event_files_processed += 1
                processed_files += 1
                event_rows_removed += rows_removed
                total_rows_removed += rows_removed

            except Exception as e:
                print(f"  ❌ Error processing {csv_filename}: {str(e)}")
                import traceback
                traceback.print_exc()

        if event_files_processed > 0:
            processed_events += 1
            print(f"  Total rows removed in this event: {event_rows_removed:,}")

        # Copy non-CSV files if requested
        if copy_other_files:
            non_csv_files = [f for f in source_event.iterdir() if f.is_file() and f.suffix != '.csv']
            for item in non_csv_files:
                try:
                    shutil.copy2(item, target_event / item.name)
                except Exception as e:
                    print(f"  ⚠️  Could not copy {item.name}: {str(e)}")

    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)
    print(f"Total events processed: {processed_events}/{len(event_folders)}")
    print(f"Total CSV files processed: {processed_files}")
    print(f"Total rows removed: {total_rows_removed:,}")
    print(f"Target location: {target_base}")
    print("=" * 80)

if __name__ == "__main__":
    model = "Model4"

    # # Inspect CSV structure first
    # print("=" * 80)
    # print("INSPECTING CSV STRUCTURE")
    # print("=" * 80)
    # inspect_csv_structure(
    #     f"data/{model}/processed/features_csv/test/event_5/1d_edges_dynamic_all.csv"
    # )

    # Remove timesteps
    remove_timesteps = 48 if model == "Model1" else 36

    remove_last_timesteps_from_events(
        source_events=f"data/{model}/processed/features_csv/test/event_*/",
        target_events=f"data/{model}/processed/features_csv/test_edited/event_*",
        n_timesteps_to_remove=remove_timesteps,
        csv_files=[
            '1d_nodes_dynamic_all.csv',
            '1d_edges_dynamic_all.csv',
            '2d_nodes_dynamic_all.csv',
            '2d_edges_dynamic_all.csv'

        ]
    )

    # Basic usage - keep first 10 timesteps for each node/edge
    copy_event_csvs_with_selective_timesteps(
        source_events=f"data/{model}/processed/features_csv/test_edited/event_*/",
        target_events=f"data/{model}/processed/features_csv/test_edited2/event_*",
        column_timestep_limits={
            "flow": 9,  # Keep timesteps 0-9 for each edge
            "velocity": 9,
            "water_level": 9,
            "inlet_flow": 9,
            "water_volume": 9,
        },
    )

    
