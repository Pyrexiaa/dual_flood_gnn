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
    timestep_col: str = 'timestep',
    id_cols: Optional[Dict[str, str]] = None
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
            '1d_nodes_dynamic_all.csv',
            '1d_edges_dynamic_all.csv',
            '2d_nodes_dynamic_all.csv',
            '2d_edges_dynamic_all.csv'
        ]
    
    # Default ID column mapping if not provided
    if id_cols is None:
        id_cols = {
            '1d_nodes_dynamic_all.csv': 'node_idx',
            '1d_edges_dynamic_all.csv': 'edge_idx',
            '2d_nodes_dynamic_all.csv': 'node_idx',
            '2d_edges_dynamic_all.csv': 'edge_idx'
        }
    
    # Convert to Path objects and handle wildcards
    source_pattern = str(source_events)
    
    # Find all event folders matching the pattern
    if '*' in source_pattern:
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
        target_base = Path(str(target_events).replace('event_*', '').rstrip('/'))
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
                    possible_id_cols = ['node_idx', 'edge_idx', 'node_id', 'edge_id']
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
                    print(f"  ⚠️  {csv_filename}: No ID column found (tried: {list(id_cols.values())})")
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
                    print(f"      Total rows: {len(df)}, IDs: {unique_ids}, Timesteps: {unique_timesteps}")
                    print(f"      Modified columns: {columns_modified}")
                    print(f"      Kept timesteps 0-{max_timestep} for each {id_col}")
                else:
                    print(f"  ✓ {csv_filename}: {len(df)} rows, no matching columns found")
                
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

def inspect_csv_structure(
    csv_path: Union[str, Path],
    show_sample: bool = True
):
    """
    Inspect the structure of a CSV file to understand its format.
    """
    df = pd.read_csv(csv_path)
    
    print(f"File: {Path(csv_path).name}")
    print(f"  Rows: {len(df)}")
    print(f"  Columns: {list(df.columns)}")
    
    # Detect ID and timestep columns
    id_cols = [col for col in df.columns if 'idx' in col.lower() or 'id' in col.lower()]
    timestep_cols = [col for col in df.columns if 'timestep' in col.lower() or 'time' in col.lower()]
    
    if id_cols:
        print(f"  Potential ID columns: {id_cols}")
        for id_col in id_cols:
            print(f"    {id_col}: {df[id_col].nunique()} unique values")
    
    if timestep_cols:
        print(f"  Potential timestep columns: {timestep_cols}")
        for ts_col in timestep_cols:
            print(f"    {ts_col}: min={df[ts_col].min()}, max={df[ts_col].max()}, unique={df[ts_col].nunique()}")
    
    if show_sample:
        print(f"\n  First few rows:")
        print(df.head(10).to_string(index=False))
    
    print()

if __name__ == "__main__":
    model = "Model4"

    # Inspect CSV structure first
    print("=" * 80)
    print("INSPECTING CSV STRUCTURE")
    print("=" * 80)
    inspect_csv_structure(
        f'data/{model}/processed/features_csv/test/event_5/1d_edges_dynamic_all.csv'
    )
    
    # Basic usage - keep first 10 timesteps for each node/edge
    copy_event_csvs_with_selective_timesteps(
        source_events=f'data/{model}/processed/features_csv/test/event_*/',
        target_events=f'data/{model}/processed/features_csv/test_edited/event_*',
        column_timestep_limits={
            'flow': 9,           # Keep timesteps 0-9 for each edge
            'velocity': 9,
            'water_level': 9,
            'inlet_flow': 9,
            'water_volume': 9
        }
    )

    # private_test_events = get_different_numbers(
    #     f'/Users/jiayulim/Documents/GitHub/dual_flood_gnn/data/{model}/raw/test.csv',
    #     f'/Users/jiayulim/Documents/GitHub/dual_flood_gnn/data/{model}/raw/public_test.csv',
    #     'Event'
    # )

    # copy_csv_excluding_integers(
    #     input_csv=f'/Users/jiayulim/Documents/GitHub/dual_flood_gnn/data/{model}/raw/test.csv',
    #     output_csv=f'/Users/jiayulim/Documents/GitHub/dual_flood_gnn/data/{model}/raw/private_test_only.csv',
    #     exclude_values=private_test_events,
    #     column_name='Event'
    # )

    # copy_dir_excluding_csv_files(
    #     csv_path=f'/Users/jiayulim/Documents/GitHub/dual_flood_gnn/data/{model}/raw/private_test_only.csv',
    #     source_dir=f'/Users/jiayulim/Documents/GitHub/dual_flood_gnn/data/{model}/raw/HEC-RAS_Results',
    #     target_dir=f'/Users/jiayulim/Documents/GitHub/dual_flood_gnn/data/{model}/raw/HEC-RAS_Results_excluded'
    # )

    # copy_dir_excluding_folders_with_integers(
    #     source_dir=f'/Users/jiayulim/Documents/GitHub/dual_flood_gnn/data/{model}/processed/features_csv/test',
    #     target_dir=f'/Users/jiayulim/Documents/GitHub/dual_flood_gnn/data/{model}/processed/features_csv/test_v2',
    #     exclude_integers=private_test_events
    # )
