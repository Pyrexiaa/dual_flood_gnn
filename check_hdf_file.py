import os
import sys
import h5py

def print_structure(name, obj):
    print(f"    {name}")

def check_hdf_file(filepath):
    print(f"\nChecking: {filepath}")
    try:
        with h5py.File(filepath, "r") as f:
            print("  Opened successfully.")
            print("  Keys:")
            f.visititems(print_structure)
        print("  ✓ OK")
        return True
    except Exception as e:
        print(f"  ✗ ERROR: {e}")
        return False

def main(folder_path):
    if not os.path.isdir(folder_path):
        print("Folder does not exist.")
        sys.exit(1)

    hdf_files = [f for f in os.listdir(folder_path) if f.endswith(".hdf")]

    if not hdf_files:
        print("No .hdf files found.")
        return

    total = len(hdf_files)
    success = 0

    for filename in sorted(hdf_files):
        filepath = os.path.join(folder_path, filename)
        if check_hdf_file(filepath):
            success += 1

    print("\n==============================")
    print(f"Checked {total} files.")
    print(f"Successful: {success}")
    print(f"Failed: {total - success}")
    print("==============================")

if __name__ == "__main__":
    main("/Users/jiayulim/Documents/GitHub/flood_pi_gnn/data/model3/raw/HEC-RAS_Results")
