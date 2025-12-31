import pandas as pd
from typing import List, Union, Set
import shutil
from pathlib import Path

def get_different_numbers(
    csv1_path: str,
    csv2_path: str,
    column_name: str
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
    column_name: str = "Event"
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
    filepath_column: str = "HECRAS_Filepath"
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
        base = p.stem           # BeaverLAKE_Pipe_Mod.p60
        exclude_files.add(p.name)   # with extension
        exclude_files.add(base)     # without extension

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

    print(f"Copy complete.")
    print(f"Copied:  {copied} files")
    print(f"Skipped: {skipped} files")

def copy_dir_excluding_folders_with_integers(
    source_dir: str,
    target_dir: str,
    exclude_integers: List[int]
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

model = 'Model4'

private_test_events = get_different_numbers(
    f'/Users/jiayulim/Documents/GitHub/dual_flood_gnn/data/{model}/raw/test.csv',
    f'/Users/jiayulim/Documents/GitHub/dual_flood_gnn/data/{model}/raw/public_test.csv',
    'Event'
)

copy_csv_excluding_integers(
    input_csv=f'/Users/jiayulim/Documents/GitHub/dual_flood_gnn/data/{model}/raw/test.csv',
    output_csv=f'/Users/jiayulim/Documents/GitHub/dual_flood_gnn/data/{model}/raw/private_test_only.csv',
    exclude_values=private_test_events,
    column_name='Event'
)

copy_dir_excluding_csv_files(
    csv_path=f'/Users/jiayulim/Documents/GitHub/dual_flood_gnn/data/{model}/raw/private_test_only.csv',
    source_dir=f'/Users/jiayulim/Documents/GitHub/dual_flood_gnn/data/{model}/raw/HEC-RAS_Results',
    target_dir=f'/Users/jiayulim/Documents/GitHub/dual_flood_gnn/data/{model}/raw/HEC-RAS_Results_excluded'
)

copy_dir_excluding_folders_with_integers(
    source_dir=f'/Users/jiayulim/Documents/GitHub/dual_flood_gnn/data/{model}/processed/features_csv/test',
    target_dir=f'/Users/jiayulim/Documents/GitHub/dual_flood_gnn/data/{model}/processed/features_csv/test_v2',
    exclude_integers=private_test_events
)