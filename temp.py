import os
import re
import argparse
import random
import csv
import pandas as pd


def rename_files(folder_path, dry_run=False):
    """
    Rename files by moving the part number (e.g. p02) before the extension
    and formatting it as _part2.

    Examples:
        Davis_Pipe_Model.p02.hdf  ->  Davis_Pipe_Model_part2.p02
        Davis_Pipe_Model.p02      ->  Davis_Pipe_Model_part2.p02
    """
    # Matches a .pNN segment anywhere in the filename (e.g. .p02, .p12)
    pattern = re.compile(r'^(.+?)\.(p(\d+))(\..*)?$', re.IGNORECASE)

    renamed, skipped = 0, 0

    for filename in sorted(os.listdir(folder_path)):
        match = pattern.match(filename)
        if not match:
            skipped += 1
            continue

        base       = match.group(1)          # Davis_Pipe_Model
        part_ext   = match.group(2)          # p02
        part_num   = int(match.group(3))     # 2  (strips leading zeros)
        extra_ext  = match.group(4) or ''    # .hdf  (may be empty)

        # Preserve the trailing extension (e.g. .hdf) if present
        new_filename = f"{base}_part2.{part_ext}{extra_ext}"

        if new_filename == filename:
            skipped += 1
            continue

        src = os.path.join(folder_path, filename)
        dst = os.path.join(folder_path, new_filename)

        if dry_run:
            print(f"[DRY RUN]  {filename}  ->  {new_filename}")
        else:
            if os.path.exists(dst):
                print(f"[SKIP] Target already exists: {new_filename}")
                skipped += 1
                continue
            os.rename(src, dst)
            print(f"[RENAMED]  {filename}  ->  {new_filename}")

        renamed += 1

    print(f"\nDone. {renamed} file(s) renamed, {skipped} skipped.")

def generate_hdf_splits(folder_path: str, output_dir: str = ".", seed: int = 42):
    """
    Scans a folder for .hdf files, assigns Event/Run IDs, and generates:
      - full_dataset.csv
      - train.csv  (60%)
      - test.csv   (40%)

    Args:
        folder_path: Path to the folder containing .hdf files.
        output_dir:  Directory where the CSV files will be saved.
        seed:        Random seed for reproducibility.
    """
    # Collect all .hdf files
    hdf_files = [
        f for f in os.listdir(folder_path) if f.lower().endswith(".hdf")
    ]

    if not hdf_files:
        print(f"No .hdf files found in '{folder_path}'.")
        return

    print(f"Found {len(hdf_files)} .hdf file(s).")

    # Shuffle and assign sequential Event/Run IDs
    random.seed(seed)
    random.shuffle(hdf_files)

    rows = []
    for idx, filename in enumerate(hdf_files, start=1):
        rows.append({
            "Event": idx,
            "Run_ID": idx,
            "HECRAS_Filepath": os.path.join(folder_path, filename).replace("\\", "/"),
            "Rain": "",
            "Inflow": "",
            "Time_Interval": "5 m",
        })

    # 60/40 split
    split_index = int(len(rows) * 0.6)
    train_rows = rows[:split_index]
    test_rows = rows[split_index:]

    fieldnames = ["Event", "Run_ID", "HECRAS_Filepath", "Rain", "Inflow", "Time_Interval"]

    os.makedirs(output_dir, exist_ok=True)

    def write_csv(filepath, data):
        with open(filepath, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(data)
        print(f"Written: {filepath}  ({len(data)} rows)")

    write_csv(os.path.join(output_dir, "full_dataset.csv"), rows)
    write_csv(os.path.join(output_dir, "train.csv"), train_rows)
    write_csv(os.path.join(output_dir, "test.csv"), test_rows)

    print(
        f"\nSummary:\n"
        f"  Total files : {len(rows)}\n"
        f"  Train (60%) : {len(train_rows)}\n"
        f"  Test  (40%) : {len(test_rows)}"
    )

def generate_empty_hdf_files(
    base_name: str,
    start: int,
    end: int,
    output_dir: str = ".",
    extension: str = ".hdf"
):
    """
    Generates empty placeholder files with a numbered suffix.

    Args:
        base_name:  The base name prefix (e.g., "NewOrleans_Pipe_Mod.p")
        start:      Starting number (inclusive)
        end:        Ending number (inclusive)
        output_dir: Directory where files will be created
        extension:  File extension (default: ".hdf")
    """
    os.makedirs(output_dir, exist_ok=True)

    filename = f"{base_name}10{extension}"
    filepath = os.path.join(output_dir, filename)
    open(filepath, 'w').close()
    print(f"Created: {filepath}")

    filename = f"{base_name}11{extension}"
    filepath = os.path.join(output_dir, filename)
    open(filepath, 'w').close()
    print(f"Created: {filepath}")

    filename = f"{base_name}14{extension}"
    filepath = os.path.join(output_dir, filename)
    open(filepath, 'w').close()
    print(f"Created: {filepath}")

    for i in range(start, end + 1):
        filename = f"{base_name}{i:02d}{extension}"
        filepath = os.path.join(output_dir, filename)
        open(filepath, 'w').close()
        print(f"Created: {filepath}")

    print(f"\nDone. {end - start + 1} file(s) created in '{output_dir}'.")

def filter_existing_files(input_csv: str, output_csv: str):
    """
    Filters rows in a CSV based on whether the file in HECRAS_Filepath exists.
    Saves the filtered result to output_csv.

    Args:
        input_csv:  Path to the input CSV file.
        output_csv: Path to save the filtered CSV file.
    """
    df = pd.read_csv(input_csv)

    original_count = len(df)
    df_filtered = df[df["HECRAS_Filepath"].apply(os.path.exists)]
    filtered_count = len(df_filtered)

    df_filtered.to_csv(output_csv, index=False)

    print(f"Original rows : {original_count}")
    print(f"Kept rows     : {filtered_count}")
    print(f"Removed rows  : {original_count - filtered_count}")
    print(f"Saved to      : {output_csv}")

if __name__ == "__main__":

    rename_files("data/model2/raw/part2", dry_run=False)
    # generate_empty_hdf_files(
    #     base_name="Model.p",
    #     start=16,
    #     end=35,
    #     output_dir="/Users/jiayulim/Documents/GitHub/flood_pi_gnn/data/model4/raw/HEC-RAS_Results"
    # )

    # generate_hdf_splits(folder_path="/Users/jiayulim/Documents/GitHub/flood_pi_gnn/data/model4/raw/HEC-RAS_Results", output_dir="/Users/jiayulim/Documents/GitHub/flood_pi_gnn/data/model4/raw/HEC-RAS_Results_sub")
    # filter_existing_files(input_csv="/Users/jiayulim/Documents/GitHub/flood_pi_gnn/data/model1/raw/HEC-RAS_Results_processed/train.csv", output_csv="/Users/jiayulim/Documents/GitHub/flood_pi_gnn/data/model1/raw/train.csv")
    # # generate_hdf_splits(folder_path="/Users/jiayulim/Documents/GitHub/flood_pi_gnn/data/model3/raw/HEC-RAS_Results", output_dir="/Users/jiayulim/Documents/GitHub/flood_pi_gnn/data/model3/raw/HEC-RAS_Results_processed")