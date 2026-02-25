import pandas as pd
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np


# Define the mapping
model1_1d_mapping = {
    0: 0,
    1: 12,
    2: 15,
    3: 4,
    4: 5,
    5: 7,
    6: 10,
    7: 9,
    8: 11,
    9: 13,
    10: 14,
    11: 16,
    12: 3,
    13: 2,
    14: 6,
    15: 8,
    16: 1
}

model2_1d_mapping = {
    0: 108,
    1: 25,
    2: 37,
    3: 42,
    4: 121,
    5: 84,
    6: 120,
    7: 131,
    8: 64,
    9: 83,
    10: 119,
    11: 82,
    12: 61,
    13: 118,
    14: 116,
    15: 81,
    16: 104,
    17: 80,
    18: 15,
    19: 60,
    20: 103,
    21: 59,
    22: 79,
    23: 58,
    24: 78,
    25: 87,
    26: 21,
    27: 57,
    28: 77,
    29: 102,
    30: 115,
    31: 41,
    32: 40,
    33: 56,
    34: 101,
    35: 100,
    36: 130,
    37: 55,
    38: 76,
    39: 99,
    40: 75,
    41: 129,
    42: 134,
    43: 30,
    44: 39,
    45: 136,
    46: 135,
    47: 106,
    48: 122,
    49: 128,
    50: 54,
    51: 62,
    52: 74,
    53: 98,
    54: 53,
    55: 97,
    56: 127,
    57: 114,
    58: 73,
    59: 72,
    60: 126,
    61: 52,
    62: 71,
    63: 48,
    64: 47,
    65: 51,
    66: 70,
    67: 86,
    68: 96,
    69: 95,
    70: 113,
    71: 125,
    72: 132,
    73: 17,
    74: 94,
    75: 93,
    76: 69,
    77: 92,
    78: 35,
    79: 133,
    80: 112,
    81: 68,
    82: 67,
    83: 124,
    84: 5,
    85: 1,
    86: 110,
    87: 65,
    88: 91,
    89: 66,
    90: 24,
    91: 50,
    92: 109,
    93: 23,
    94: 123,
    95: 85,
    96: 46,
    97: 45,
    98: 22,
    99: 38,
    100: 36,
    101: 49,
    102: 107,
    103: 90,
    104: 63,
    105: 19,
    106: 89,
    107: 34,
    108: 33,
    109: 32,
    110: 31,
    111: 88,
    112: 29,
    113: 18,
    114: 14,
    115: 9,
    116: 28,
    117: 16,
    118: 8,
    119: 2,
    120: 3,
    121: 0,
    122: 105,
    123: 20,
    124: 43,
    125: 44,
    126: 27,
    127: 26,
    128: 11,
    129: 12,
    130: 13,
    131: 6,
    132: 4,
    133: 137,
    134: 138,
    135: 139,
    136: 140,
    137: 141,
    138: 142,
    139: 143,
    140: 144,
    141: 111,
    142: 145,
    143: 146,
    144: 147,
    145: 10,
    146: 148,
    147: 149,
    148: 150,
    149: 151,
    150: 152,
    151: 153,
    152: 154,
    153: 155,
    154: 156,
    155: 157,
    156: 158,
    157: 159,
    158: 160,
    159: 161,
    160: 162,
    161: 163,
    162: 164,
    163: 165,
    164: 166,
    165: 167,
    166: 168,
    167: 169,
    168: 117,
    169: 170,
    170: 171,
    171: 172,
    172: 173,
    173: 174,
    174: 175,
    175: 176,
    176: 177,
    177: 178,
    178: 179,
    179: 180,
    180: 181,
    181: 182,
    182: 184,
    183: 185,
    184: 183,
    185: 186,
    186: 187,
    187: 188,
    188: 189,
    189: 190,
    190: 191,
    191: 192,
    192: 7,
    193: 193,
    194: 194,
    195: 195,
    196: 196,
    197: 197
}

def remap_node_indices(train_dir, mapping=None):
    """
    Remap node indices in 1d_nodes_dynamic_all.csv files based on the provided mapping.
    
    Parameters:
    -----------
    train_dir : str or Path
        Path to the parent 'train' directory containing event_* folders
    mapping : dict, optional
        Dictionary mapping original node_idx to new node_idx.
        If None, uses model1_1d_mapping by default.
    """
    if mapping is None:
        mapping = model1_1d_mapping
    
    train_path = Path(train_dir)
    
    if not train_path.exists():
        print(f"Error: Directory {train_path} does not exist")
        return
    
    # Find all event_* folders
    event_folders = sorted([f for f in train_path.iterdir() if f.is_dir() and f.name.startswith('event_')])
    
    if not event_folders:
        print(f"No event_* folders found in {train_path}")
        return
    
    print(f"Found {len(event_folders)} event folders")
    
    for event_folder in event_folders:
        csv_file = event_folder / "1d_nodes_dynamic_all.csv"
        
        if not csv_file.exists():
            print(f"Warning: {csv_file} not found, skipping...")
            continue
        
        print(f"Processing {csv_file}...")
        
        try:
            # Read the CSV file
            df = pd.read_csv(csv_file)
            
            # Verify required columns exist
            required_cols = ['timestep', 'node_idx', 'water_level', 'inlet_flow']
            if not all(col in df.columns for col in required_cols):
                print(f"Error: Missing required columns in {csv_file}")
                continue
            
            # For each timestep, rearrange rows according to the mapping
            # mapping[FID] = HEC-RAS_column means: output position FID gets data from input position HEC-RAS_column
            
            result_dfs = []
            for timestep in df['timestep'].unique():
                timestep_df = df[df['timestep'] == timestep].copy()
                
                # Create new rows in the correct order
                new_rows = []
                for new_fid in sorted(mapping.keys()):
                    # Get which old position should go to this new FID position
                    old_position = mapping[new_fid]
                    
                    # Find the row with that original node_idx
                    matching_row = timestep_df[timestep_df['node_idx'] == old_position]
                    
                    if not matching_row.empty:
                        row = matching_row.iloc[0].copy()
                        row['node_idx'] = new_fid  # Set to the new FID position
                        new_rows.append(row)
                    else:
                        print(f"  Warning: No data found for old node_idx={old_position} at timestep={timestep}")
                
                if new_rows:
                    new_timestep_df = pd.DataFrame(new_rows)
                    result_dfs.append(new_timestep_df)
            
            # Combine all timesteps
            if result_dfs:
                df = pd.concat(result_dfs, ignore_index=True)
            else:
                print(f"Warning: No data could be remapped for {csv_file}")
                continue
            
            # Convert node_idx to integer
            df['node_idx'] = df['node_idx'].astype(int)
            
            # Already sorted by construction, but ensure it
            df = df.sort_values(['timestep', 'node_idx']).reset_index(drop=True)
            
            # Save the remapped CSV
            output_file = event_folder / "1d_nodes_dynamic_all_remapped.csv"
            df.to_csv(output_file, index=False)
            
            print(f"  ✓ Saved remapped file to {output_file}")
            
        except Exception as e:
            print(f"Error processing {csv_file}: {str(e)}")
            continue
    
    print("\nProcessing complete!")

def plot_water_levels(train_dir, static_csv_path, hecras_csv_path, event_names=None):
    """
    Plot water level across timesteps for each node, comparing with invert elevation and HEC-RAS results.
    
    Parameters:
    -----------
    train_dir : str or Path
        Path to the parent 'train' directory containing event_* folders
    static_csv_path : str or Path
        Path to the static CSV file with node information (invert_elevation, etc.)
    hecras_csv_path : str or Path
        Path to the HEC-RAS results CSV file with ground truth data
    event_names : list of str, optional
        List of event folder names to plot (e.g., ['event_1', 'event_5']). 
        If None, plots first 2 events.
    """
    train_path = Path(train_dir)
    static_path = Path(static_csv_path)
    hecras_path = Path(hecras_csv_path)
    
    # Read static node data
    print(f"Reading static node data from {static_path}...")
    static_df = pd.read_csv(static_path)
    
    # Read HEC-RAS ground truth data
    print(f"Reading HEC-RAS ground truth data from {hecras_path}...")
    hecras_df = pd.read_csv(hecras_path)
    
    # Create a dictionary mapping node_idx to invert_elevation
    node_invert_elevation = dict(zip(static_df['node_idx'], static_df['invert_elevation']))
    
    # Find event folders
    if event_names is not None:
        # Use specified event names
        event_folders = []
        for event_name in event_names:
            event_path = train_path / event_name
            if event_path.exists() and event_path.is_dir():
                event_folders.append(event_path)
            else:
                print(f"Warning: Event folder '{event_name}' not found, skipping...")
    else:
        # Default to first 2 events
        event_folders = sorted([f for f in train_path.iterdir() 
                               if f.is_dir() and f.name.startswith('event_')])[:2]
    
    if not event_folders:
        print(f"No event folders found in {train_path}")
        return
    
    print(f"Found {len(event_folders)} event folders to plot")
    
    # Get unique nodes from static data
    unique_nodes = sorted(static_df['node_idx'].unique())
    num_nodes = len(unique_nodes)
    
    print(f"Number of nodes to plot: {num_nodes}")
    
    # Process HEC-RAS data
    # Remove 'No' and 'Date' columns to get only node data
    hecras_node_cols = [col for col in hecras_df.columns if col not in ['No', 'Date']]
    hecras_data_only = hecras_df[hecras_node_cols].copy()
    
    # Convert column names to integers (they represent node indices)
    hecras_data_only.columns = [int(col) for col in hecras_data_only.columns]
    
    print(f"\nHEC-RAS data shape: {hecras_data_only.shape}")
    print(f"HEC-RAS nodes: {sorted(hecras_data_only.columns.tolist())}")
    
    # Validate and trim HEC-RAS data if needed
    num_timesteps = None
    for event_folder in event_folders:
        csv_file = event_folder / "1d_nodes_dynamic_all.csv"
        if csv_file.exists():
            temp_df = pd.read_csv(csv_file)
            num_timesteps = len(temp_df['timestep'].unique())
            break
    
    if num_timesteps is not None:
        expected_rows = num_timesteps
        actual_rows = len(hecras_data_only)
        
        print(f"Expected HEC-RAS rows: {expected_rows} ({num_nodes} nodes × {num_timesteps} timesteps)")
        print(f"Actual HEC-RAS rows: {actual_rows}")
        
        if actual_rows > expected_rows:
            rows_to_remove = actual_rows - expected_rows
            print(f"Trimming first {rows_to_remove} rows from HEC-RAS data")
            hecras_data_only = hecras_data_only.iloc[rows_to_remove:].reset_index(drop=True)
        elif actual_rows < expected_rows:
            print(f"Warning: HEC-RAS data has fewer rows than expected. Some timesteps may be missing.")
    
    # Reshape HEC-RAS data into long format for easier plotting
    # Create a dataframe with timestep, node_idx, water_level
    hecras_long = []
    if num_timesteps is not None:
        for timestep in range(num_timesteps):
            for node_idx in sorted(hecras_data_only.columns):
                row_idx = timestep
                if row_idx < len(hecras_data_only):
                    water_level = hecras_data_only.iloc[row_idx][node_idx]
                    hecras_long.append({
                        'timestep': timestep,
                        'node_idx': node_idx,
                        'water_level': water_level
                    })
        hecras_df_long = pd.DataFrame(hecras_long)
        print(f"Processed HEC-RAS data into long format: {len(hecras_df_long)} rows")
    else:
        hecras_df_long = pd.DataFrame(columns=['timestep', 'node_idx', 'water_level'])
        print("Warning: Could not process HEC-RAS data into long format")
    
    # Process each event
    for event_idx, event_folder in enumerate(event_folders):
        csv_file_original = event_folder / "1d_nodes_dynamic_all.csv"
        csv_file_remapped = event_folder / "1d_nodes_dynamic_all.csv"
        
        if not csv_file_original.exists():
            print(f"Warning: {csv_file_original} not found, skipping...")
            continue
        
        if not csv_file_remapped.exists():
            print(f"Warning: {csv_file_remapped} not found, skipping...")
            continue
        
        print(f"\nProcessing {event_folder.name}...")
        
        # Read both CSVs
        df_original = pd.read_csv(csv_file_original)
        df_remapped = pd.read_csv(csv_file_remapped)
        
        # Create output directory for plots
        output_dir = event_folder / "plots"
        output_dir.mkdir(exist_ok=True)
        
        # Plot for each node
        for node_idx in unique_nodes:
            # Filter data for this node from both datasets
            node_data_original = df_original[df_original['node_idx'] == node_idx].copy()
            node_data_remapped = df_remapped[df_remapped['node_idx'] == node_idx].copy()
            node_data_hecras = hecras_df_long[hecras_df_long['node_idx'] == node_idx].copy()
            
            if node_data_original.empty and node_data_remapped.empty:
                print(f"  Warning: No data for node {node_idx}, skipping...")
                continue
            
            # Sort by timestep
            node_data_original = node_data_original.sort_values('timestep')
            node_data_remapped = node_data_remapped.sort_values('timestep')
            if not node_data_hecras.empty:
                node_data_hecras = node_data_hecras.sort_values('timestep')
            
            # Get invert elevation for this node
            invert_elev = node_invert_elevation.get(node_idx, None)
            
            if invert_elev is None:
                print(f"  Warning: No invert elevation for node {node_idx}, skipping...")
                continue
            
            # Create side-by-side plot
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 6))
            
            # Calculate common y-axis limits
            y_min_list = [invert_elev]
            y_max_list = [invert_elev]
            
            if not node_data_original.empty:
                y_min_list.append(node_data_original['water_level'].min())
                y_max_list.append(node_data_original['water_level'].max())
            
            if not node_data_remapped.empty:
                y_min_list.append(node_data_remapped['water_level'].min())
                y_max_list.append(node_data_remapped['water_level'].max())
            
            if not node_data_hecras.empty:
                y_min_list.append(node_data_hecras['water_level'].min())
                y_max_list.append(node_data_hecras['water_level'].max())
            
            y_min_common = min(y_min_list) - 1
            y_max_common = max(y_max_list) + 1
            
            # LEFT PLOT: Original data
            if not node_data_original.empty:
                ax1.plot(node_data_original['timestep'], node_data_original['water_level'], 
                        label='Water Level', color='steelblue', linewidth=2, marker='o', markersize=3)
                ax1.axhline(y=invert_elev, color='red', linestyle='--', 
                           linewidth=2, label=f'Invert Elevation ({invert_elev:.2f})')
                
                # Add HEC-RAS ground truth line
                if not node_data_hecras.empty:
                    ax1.plot(node_data_hecras['timestep'], node_data_hecras['water_level'], 
                            label='HEC-RAS', color='orange', linewidth=2, linestyle='-.', marker='s', markersize=2, alpha=0.7)
                
                ax1.set_xlabel('Timestep', fontsize=12)
                ax1.set_ylabel('Elevation (m)', fontsize=12)
                ax1.set_title(f'Original - Node {node_idx}', fontsize=14, fontweight='bold')
                ax1.legend(loc='best', fontsize=10)
                ax1.grid(True, alpha=0.3)
                ax1.set_ylim([y_min_common, y_max_common])
            else:
                ax1.text(0.5, 0.5, 'No Original Data', ha='center', va='center', 
                        transform=ax1.transAxes, fontsize=14)
                ax1.set_title(f'Original - Node {node_idx}', fontsize=14, fontweight='bold')
                ax1.set_ylim([y_min_common, y_max_common])
            
            # RIGHT PLOT: Remapped data
            if not node_data_remapped.empty:
                ax2.plot(node_data_remapped['timestep'], node_data_remapped['water_level'], 
                        label='Water Level', color='darkgreen', linewidth=2, marker='o', markersize=3)
                ax2.axhline(y=invert_elev, color='red', linestyle='--', 
                           linewidth=2, label=f'Invert Elevation ({invert_elev:.2f})')
                
                # Add HEC-RAS ground truth line
                if not node_data_hecras.empty:
                    ax2.plot(node_data_hecras['timestep'], node_data_hecras['water_level'], 
                            label='HEC-RAS', color='orange', linewidth=2, linestyle='-.', marker='s', markersize=2, alpha=0.7)
                
                ax2.set_xlabel('Timestep', fontsize=12)
                ax2.set_ylabel('Elevation (m)', fontsize=12)
                ax2.set_title(f'Remapped - Node {node_idx}', fontsize=14, fontweight='bold')
                ax2.legend(loc='best', fontsize=10)
                ax2.grid(True, alpha=0.3)
                ax2.set_ylim([y_min_common, y_max_common])
            else:
                ax2.text(0.5, 0.5, 'No Remapped Data', ha='center', va='center', 
                        transform=ax2.transAxes, fontsize=14)
                ax2.set_title(f'Remapped - Node {node_idx}', fontsize=14, fontweight='bold')
                ax2.set_ylim([y_min_common, y_max_common])
            
            # Overall title
            fig.suptitle(f'{event_folder.name} - Node {node_idx}: Original vs Remapped', 
                        fontsize=16, fontweight='bold', y=1.02)
            
            # Save plot
            output_file = output_dir / f"node_{node_idx:03d}_comparison.png"
            plt.tight_layout()
            plt.savefig(output_file, dpi=150, bbox_inches='tight')
            plt.close()
            
            if node_idx == unique_nodes[0]:  # Print progress for first node only
                print(f"  ✓ Created comparison plots in {output_dir}/")
        
        print(f"  ✓ Completed {num_nodes} comparison plots for {event_folder.name}")
    
    print("\nPlotting complete!")

def plot_water_levels_all(train_dir, static_csv_path, event_names=None):
    """
    Plot water level across timesteps for each node, comparing with invert elevation.
    
    Parameters:
    -----------
    train_dir : str or Path
        Path to the parent 'train' directory containing event_* folders
    static_csv_path : str or Path
        Path to the static CSV file with node information (invert_elevation, etc.)
    event_names : list of str, optional
        List of event folder names to plot (e.g., ['event_1', 'event_5']). 
        If None, plots all events.
    """
    train_path = Path(train_dir)
    static_path = Path(static_csv_path)
    
    # Read static node data
    print(f"Reading static node data from {static_path}...")
    static_df = pd.read_csv(static_path)
    
    # Create a dictionary mapping node_idx to invert_elevation
    node_invert_elevation = dict(zip(static_df['node_idx'], static_df['invert_elevation']))
    
    # Find event folders
    if event_names is not None:
        # Use specified event names
        event_folders = []
        for event_name in event_names:
            event_path = train_path / event_name
            if event_path.exists() and event_path.is_dir():
                event_folders.append(event_path)
            else:
                print(f"Warning: Event folder '{event_name}' not found, skipping...")
    else:
        # Plot all events
        event_folders = sorted([f for f in train_path.iterdir() 
                               if f.is_dir() and f.name.startswith('event_')])
    
    if not event_folders:
        print(f"No event folders found in {train_path}")
        return
    
    print(f"Found {len(event_folders)} event folders to plot")
    
    # Get unique nodes from static data
    unique_nodes = sorted(static_df['node_idx'].unique())
    num_nodes = len(unique_nodes)
    
    print(f"Number of nodes to plot: {num_nodes}")
    
    # Process each event
    for event_idx, event_folder in enumerate(event_folders):
        csv_file_original = event_folder / "1d_nodes_dynamic_all.csv"
        csv_file_remapped = event_folder / "1d_nodes_dynamic_all.csv"
        
        if not csv_file_original.exists():
            print(f"Warning: {csv_file_original} not found, skipping...")
            continue
        
        if not csv_file_remapped.exists():
            print(f"Warning: {csv_file_remapped} not found, skipping...")
            continue
        
        print(f"\nProcessing {event_folder.name} ({event_idx + 1}/{len(event_folders)})...")
        
        # Read both CSVs
        df_original = pd.read_csv(csv_file_original)
        df_remapped = pd.read_csv(csv_file_remapped)
        
        # Create output directory for plots
        output_dir = event_folder / "plots"
        output_dir.mkdir(exist_ok=True)
        
        # Plot for each node
        for node_idx in unique_nodes:
            # Filter data for this node from both datasets
            node_data_original = df_original[df_original['node_idx'] == node_idx].copy()
            node_data_remapped = df_remapped[df_remapped['node_idx'] == node_idx].copy()
            
            if node_data_original.empty and node_data_remapped.empty:
                print(f"  Warning: No data for node {node_idx}, skipping...")
                continue
            
            # Sort by timestep
            node_data_original = node_data_original.sort_values('timestep')
            node_data_remapped = node_data_remapped.sort_values('timestep')
            
            # Get invert elevation for this node
            invert_elev = node_invert_elevation.get(node_idx, None)
            
            if invert_elev is None:
                print(f"  Warning: No invert elevation for node {node_idx}, skipping...")
                continue
            
            # Create side-by-side plot
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 6))
            
            # Calculate common y-axis limits
            y_min_list = [invert_elev]
            y_max_list = [invert_elev]
            
            if not node_data_original.empty:
                y_min_list.append(node_data_original['water_level'].min())
                y_max_list.append(node_data_original['water_level'].max())
            
            if not node_data_remapped.empty:
                y_min_list.append(node_data_remapped['water_level'].min())
                y_max_list.append(node_data_remapped['water_level'].max())
            
            y_min_common = min(y_min_list) - 1
            y_max_common = max(y_max_list) + 1
            
            # LEFT PLOT: Original data
            if not node_data_original.empty:
                ax1.plot(node_data_original['timestep'], node_data_original['water_level'], 
                        label='Water Level', color='steelblue', linewidth=2, marker='o', markersize=3)
                ax1.axhline(y=invert_elev, color='red', linestyle='--', 
                           linewidth=2, label=f'Invert Elevation ({invert_elev:.2f})')
                
                ax1.set_xlabel('Timestep', fontsize=12)
                ax1.set_ylabel('Elevation (m)', fontsize=12)
                ax1.set_title(f'Original - Node {node_idx}', fontsize=14, fontweight='bold')
                ax1.legend(loc='best', fontsize=10)
                ax1.grid(True, alpha=0.3)
                ax1.set_ylim([y_min_common, y_max_common])
            else:
                ax1.text(0.5, 0.5, 'No Original Data', ha='center', va='center', 
                        transform=ax1.transAxes, fontsize=14)
                ax1.set_title(f'Original - Node {node_idx}', fontsize=14, fontweight='bold')
                ax1.set_ylim([y_min_common, y_max_common])
            
            # RIGHT PLOT: Remapped data
            if not node_data_remapped.empty:
                ax2.plot(node_data_remapped['timestep'], node_data_remapped['water_level'], 
                        label='Water Level', color='darkgreen', linewidth=2, marker='o', markersize=3)
                ax2.axhline(y=invert_elev, color='red', linestyle='--', 
                           linewidth=2, label=f'Invert Elevation ({invert_elev:.2f})')
                
                ax2.set_xlabel('Timestep', fontsize=12)
                ax2.set_ylabel('Elevation (m)', fontsize=12)
                ax2.set_title(f'Remapped - Node {node_idx}', fontsize=14, fontweight='bold')
                ax2.legend(loc='best', fontsize=10)
                ax2.grid(True, alpha=0.3)
                ax2.set_ylim([y_min_common, y_max_common])
            else:
                ax2.text(0.5, 0.5, 'No Remapped Data', ha='center', va='center', 
                        transform=ax2.transAxes, fontsize=14)
                ax2.set_title(f'Remapped - Node {node_idx}', fontsize=14, fontweight='bold')
                ax2.set_ylim([y_min_common, y_max_common])
            
            # Overall title
            fig.suptitle(f'{event_folder.name} - Node {node_idx}: Original vs Remapped', 
                        fontsize=16, fontweight='bold', y=1.02)
            
            # Save plot
            output_file = output_dir / f"node_{node_idx:03d}_comparison.png"
            plt.tight_layout()
            plt.savefig(output_file, dpi=150, bbox_inches='tight')
            plt.close()
            
            if node_idx == unique_nodes[0]:  # Print progress for first node only
                print(f"  ✓ Created comparison plots in {output_dir}/")
        
        print(f"  ✓ Completed {num_nodes} comparison plots for {event_folder.name}")
    
    print("\nPlotting complete!")

def remove_original_csv(train_dir):
    """
    Remove all 1d_nodes_dynamic_all.csv files from all event_* folders.
    
    Parameters:
    -----------
    train_dir : str or Path
        Path to the parent 'train' directory containing event_* folders
    """
    train_path = Path(train_dir)
    
    if not train_path.exists():
        print(f"Error: Directory {train_path} does not exist")
        return
    
    # Find all event_* folders
    event_folders = sorted([f for f in train_path.iterdir() 
                           if f.is_dir() and f.name.startswith('event_')])
    
    if not event_folders:
        print(f"No event_* folders found in {train_path}")
        return
    
    print(f"Found {len(event_folders)} event folders")
    removed_count = 0
    
    for event_folder in event_folders:
        csv_file = event_folder / "1d_nodes_dynamic_all.csv"
        
        if csv_file.exists():
            try:
                csv_file.unlink()  # Delete the file
                print(f"  ✓ Removed: {csv_file}")
                removed_count += 1
            except Exception as e:
                print(f"  ✗ Error removing {csv_file}: {str(e)}")
        else:
            print(f"  - Not found: {csv_file}")
    
    print(f"\nRemoved {removed_count} files out of {len(event_folders)} folders")


def rename_remapped_to_original(train_dir):
    """
    Rename all 1d_nodes_dynamic_all_remapped.csv files to 1d_nodes_dynamic_all.csv
    in all event_* folders.
    
    Parameters:
    -----------
    train_dir : str or Path
        Path to the parent 'train' directory containing event_* folders
    """
    train_path = Path(train_dir)
    
    if not train_path.exists():
        print(f"Error: Directory {train_path} does not exist")
        return
    
    # Find all event_* folders
    event_folders = sorted([f for f in train_path.iterdir() 
                           if f.is_dir() and f.name.startswith('event_')])
    
    if not event_folders:
        print(f"No event_* folders found in {train_path}")
        return
    
    print(f"Found {len(event_folders)} event folders")
    renamed_count = 0
    
    for event_folder in event_folders:
        old_file = event_folder / "1d_nodes_dynamic_all_remapped.csv"
        new_file = event_folder / "1d_nodes_dynamic_all.csv"
        
        if old_file.exists():
            # Check if destination already exists
            if new_file.exists():
                print(f"  ⚠ Warning: {new_file} already exists, skipping rename for {event_folder.name}")
                continue
            
            try:
                old_file.rename(new_file)  # Rename the file
                print(f"  ✓ Renamed: {old_file.name} → {new_file.name} in {event_folder.name}")
                renamed_count += 1
            except Exception as e:
                print(f"  ✗ Error renaming in {event_folder.name}: {str(e)}")
        else:
            print(f"  - Not found: {old_file}")
    
    print(f"\nRenamed {renamed_count} files out of {len(event_folders)} folders")

def remap_combined_dataset(csv_path, output_path=None):
    """
    Remap node_id values in a combined dataset CSV based on model_id and node_type.
    
    This function processes rows where node_type == 1 (1D nodes) and remaps the node_id
    according to the appropriate mapping based on model_id:
    - model_id == 1: uses model1_1d_mapping
    - model_id == 2: uses model2_1d_mapping
    
    The mapping interpretation: mapping[FID] = HEC-RAS_column means:
    "output position FID gets data from input position HEC-RAS_column"
    
    Parameters:
    -----------
    csv_path : str or Path
        Path to the input CSV file
    output_path : str or Path, optional
        Path for the output CSV file. If None, appends '_remapped' to the input filename.
    
    Returns:
    --------
    pd.DataFrame : The remapped dataframe
    """
    csv_path = Path(csv_path)
    
    if not csv_path.exists():
        print(f"Error: File {csv_path} does not exist")
        return None
    
    print(f"Reading CSV from {csv_path}...")
    df = pd.read_csv(csv_path)
    
    # Verify required columns exist
    required_cols = ['row_id', 'model_id', 'event_id', 'node_type', 'node_id', 'water_level', 'Usage']
    if not all(col in df.columns for col in required_cols):
        print(f"Error: Missing required columns. Expected: {required_cols}")
        print(f"Found: {df.columns.tolist()}")
        return None
    
    print(f"Original data shape: {df.shape}")
    print(f"Unique model_ids: {sorted(df['model_id'].unique())}")
    print(f"Unique node_types: {sorted(df['node_type'].unique())}")
    
    # Create a copy for remapping
    df_remapped = df.copy()
    
    # Process model 1, node_type 1
    mask_m1 = (df_remapped['model_id'] == 1) & (df_remapped['node_type'] == 1)
    count_m1 = mask_m1.sum()
    if count_m1 > 0:
        print(f"\nProcessing {count_m1} rows for model_id=1, node_type=1...")
        df_m1 = df_remapped[mask_m1].copy()
        
        # Check which node_ids are present
        unique_node_ids = sorted(df_m1['node_id'].unique())
        print(f"  Unique node_ids found: {unique_node_ids}")
        print(f"  Expected node_ids in mapping: {sorted(model1_1d_mapping.values())}")
        
        # Create reverse mapping for direct application
        reverse_mapping = {v: k for k, v in model1_1d_mapping.items()}
        
        # Check for node_ids not in mapping
        unmapped_ids = set(unique_node_ids) - set(reverse_mapping.keys())
        if unmapped_ids:
            print(f"  WARNING: Found node_ids not in mapping: {sorted(unmapped_ids)}")
            print(f"  These {len(df_m1[df_m1['node_id'].isin(unmapped_ids)])} rows will be dropped!")
        
        # Apply the reverse mapping
        df_m1['node_id'] = df_m1['node_id'].map(reverse_mapping)
        
        # Check for unmapped values
        if df_m1['node_id'].isna().any():
            unmapped_count = df_m1['node_id'].isna().sum()
            print(f"  Warning: {unmapped_count} rows could not be mapped and will be dropped")
            df_m1 = df_m1.dropna(subset=['node_id'])
        
        # Convert to integer
        df_m1['node_id'] = df_m1['node_id'].astype(int)
        
        # Replace in main dataframe
        df_remapped = df_remapped[~mask_m1]  # Remove original model 1 rows
        df_remapped = pd.concat([df_remapped, df_m1], ignore_index=True)
        print(f"  ✓ Remapped {len(df_m1)} rows for model 1")
    
    # Process model 2, node_type 1
    mask_m2 = (df_remapped['model_id'] == 2) & (df_remapped['node_type'] == 1)
    count_m2 = mask_m2.sum()
    if count_m2 > 0:
        print(f"\nProcessing {count_m2} rows for model_id=2, node_type=1...")
        df_m2 = df_remapped[mask_m2].copy()
        
        # Check which node_ids are present
        unique_node_ids = sorted(df_m2['node_id'].unique())
        print(f"  Unique node_ids found: {len(unique_node_ids)} unique values")
        print(f"  Expected node_ids in mapping: {len(model2_1d_mapping.values())} unique values")
        
        # Create reverse mapping for direct application
        reverse_mapping = {v: k for k, v in model2_1d_mapping.items()}
        
        # Check for node_ids not in mapping
        unmapped_ids = set(unique_node_ids) - set(reverse_mapping.keys())
        if unmapped_ids:
            print(f"  WARNING: Found {len(unmapped_ids)} node_ids not in mapping")
            print(f"  Sample unmapped IDs: {sorted(list(unmapped_ids))[:10]}")
            print(f"  These {len(df_m2[df_m2['node_id'].isin(unmapped_ids)])} rows will be dropped!")
        
        # Apply the reverse mapping
        df_m2['node_id'] = df_m2['node_id'].map(reverse_mapping)
        
        # Check for unmapped values
        if df_m2['node_id'].isna().any():
            unmapped_count = df_m2['node_id'].isna().sum()
            print(f"  Warning: {unmapped_count} rows could not be mapped and will be dropped")
            df_m2 = df_m2.dropna(subset=['node_id'])
        
        # Convert to integer
        df_m2['node_id'] = df_m2['node_id'].astype(int)
        
        # Replace in main dataframe
        df_remapped = df_remapped[~mask_m2]  # Remove original model 2 rows
        df_remapped = pd.concat([df_remapped, df_m2], ignore_index=True)
        print(f"  ✓ Remapped {len(df_m2)} rows for model 2")
    
    # Sort by original column order and row_id
    df_remapped = df_remapped.sort_values('row_id').reset_index(drop=True)
    
    # Determine output path
    if output_path is None:
        output_path = csv_path.parent / f"{csv_path.stem}_remapped{csv_path.suffix}"
    else:
        output_path = Path(output_path)
    
    # Save the remapped CSV
    df_remapped.to_csv(output_path, index=False)
    print(f"\n✓ Saved remapped file to {output_path}")
    print(f"Output data shape: {df_remapped.shape}")
    
    return df_remapped

if __name__ == "__main__":
    # train_directory = "/Users/jiayulim/Documents/GitHub/flood_pi_gnn/data/model1/processed/features_csv/test"
    # remap_node_indices(train_directory, mapping=model1_1d_mapping)
    # train_directory = "/Users/jiayulim/Documents/GitHub/flood_pi_gnn/data/model1/processed/features_csv/test"
    # static_csv = "/Users/jiayulim/Documents/GitHub/flood_pi_gnn/data/model1/processed/features_csv/test/1d_nodes_static.csv"
    # hecras_csv = "/Users/jiayulim/Documents/GitHub/flood_pi_gnn/data/model1/model1_event1_1D.csv"
    # remove_original_csv(train_directory)
    # rename_remapped_to_original(train_directory)
    # plot_water_levels(train_directory, static_csv, hecras_csv, event_names=["event_1"])
    # plot_water_levels_all(train_directory, static_csv, event_names=None)  # Plot all events together

    train_directory_2 = "/Users/jiayulim/Documents/GitHub/flood_pi_gnn/data/model2/processed/features_csv/train"
    remap_node_indices(train_directory_2, mapping=model2_1d_mapping)
    train_directory_2 = "/Users/jiayulim/Documents/GitHub/flood_pi_gnn/data/model2/processed/features_csv/train"
    static_csv_2 = "/Users/jiayulim/Documents/GitHub/flood_pi_gnn/data/model2/processed/features_csv/train/1d_nodes_static.csv"
    hecras_csv_2 = "/Users/jiayulim/Documents/GitHub/flood_pi_gnn/data/model2/model2_event43_1D.csv"
    remove_original_csv(train_directory_2)
    rename_remapped_to_original(train_directory_2)
    plot_water_levels(train_directory_2, static_csv_2, hecras_csv_2, event_names=["event_43"])
    

    # # Remap solutions.csv
    # solutions_csv = "/Users/jiayulim/Documents/GitHub/flood_pi_gnn/solutions.csv"
    # updated_solutions_csv = "/Users/jiayulim/Documents/GitHub/flood_pi_gnn/solutions_remapped.csv"
    # remap_combined_dataset(solutions_csv, output_path=updated_solutions_csv)