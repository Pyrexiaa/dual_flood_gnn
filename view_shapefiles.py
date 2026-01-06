from typing import List, Optional, Tuple, Union
import numpy as np
import geopandas as gpd
import os
import json
import pandas as pd
from pathlib import Path


def visualize_boundary_condition_masks(
    nodes_2d_shp_file: str,
    edges_2d_shp_file: str,
    boundary_condition_npz_file: str = "boundary_condition_masks.npz",
    constant_values_file: str = "constant_value.npz",
    output_dir: str = None,
):
    """
    Load boundary condition masks from NPZ file and create filtered shapefiles for visualization.

    This function filters the original shapefiles to:
    - Remove ghost nodes
    - Remove edges connected to ghost nodes (but keep boundary edges)
    - Preserve all boundary nodes and edges

    Creates 4 shapefiles:
    1. All valid nodes (regular + boundary, no ghosts)
    2. Boundary nodes only
    3. All valid edges (regular + boundary, no edges to ghost nodes)
    4. Boundary edges only

    Args:
        nodes_2d_shp_file: Path to 2D nodes shapefile
        edges_2d_shp_file: Path to 2D edges shapefile
        boundary_condition_npz_file: Name of boundary condition NPZ file
        constant_values_file: Name of constant values NPZ file
        output_dir: Output directory for shapefiles (default: boundary_viz2)
    """

    # Load boundary condition masks
    if not os.path.exists(boundary_condition_npz_file):
        print(f"❌ Boundary condition file not found: {boundary_condition_npz_file}")
        return

    bc_data = np.load(boundary_condition_npz_file)

    print("=" * 80)
    print("BOUNDARY CONDITION MASKS CONTENT")
    print("=" * 80)
    print(f"\nFile: {boundary_condition_npz_file}\n")

    # Print all arrays in the NPZ file
    print("Available arrays:")
    for key in bc_data.files:
        arr = bc_data[key]
        print(f"  - {key:30s} Shape: {str(arr.shape):20s} Dtype: {arr.dtype}")
        if arr.dtype == bool:
            print(f"    True count: {np.sum(arr)}")

    # Extract masks
    boundary_nodes_mask = bc_data["boundary_nodes_mask"]
    boundary_edges_mask = bc_data["boundary_edges_mask"]
    inflow_edges_mask = bc_data["inflow_edges_mask"]
    outflow_edges_mask = bc_data["outflow_edges_mask"]

    # Load constant values to get processed node positions and edge index
    if not os.path.exists(constant_values_file):
        print(f"\n❌ Constant values file not found: {constant_values_file}")
        print("Cannot create shapefiles without processed node data.")
        return

    constant_values = np.load(constant_values_file)
    static_nodes = constant_values["static_nodes"]
    edge_index = constant_values["edge_index"]

    print("\nConstant values arrays:")
    for file in constant_values.files:
        print(
            f"  - {file:30s} Shape: {str(constant_values[file].shape):20s} Dtype: {constant_values[file].dtype}"
        )

    # Check if valid masks are available
    if "valid_nodes_mask" in constant_values.files:
        valid_nodes_mask = constant_values["valid_nodes_mask"]
        valid_edges_mask = constant_values["valid_edges_mask"]
        print("\n✓ Found valid masks in constant values")
        print(f"  Valid nodes (regular + boundary): {np.sum(valid_nodes_mask)}")
        print(f"  Ghost nodes: {np.sum(~valid_nodes_mask)}")
        print(f"  Boundary nodes: {np.sum(boundary_nodes_mask)}")
        print(f"  Valid edges (regular + boundary): {np.sum(valid_edges_mask)}")
        print(f"  Ghost edges: {np.sum(~valid_edges_mask)}")
        print(f"  Boundary edges: {np.sum(boundary_edges_mask)}")
    else:
        print("\n⚠ No valid masks found - assuming all data is valid")
        valid_nodes_mask = np.ones(len(static_nodes), dtype=bool)
        valid_edges_mask = np.ones(edge_index.shape[1], dtype=bool)

    # Load original shapefiles
    print("\n" + "=" * 80)
    print("LOADING ORIGINAL SHAPEFILES")
    print("=" * 80)

    if not os.path.exists(nodes_2d_shp_file):
        print(f"❌ Nodes shapefile not found: {nodes_2d_shp_file}")
        return

    if not os.path.exists(edges_2d_shp_file):
        print(f"❌ Edges shapefile not found: {edges_2d_shp_file}")
        return

    original_nodes_gdf = gpd.read_file(nodes_2d_shp_file)
    original_edges_gdf = gpd.read_file(edges_2d_shp_file)
    crs = original_nodes_gdf.crs

    print(f"\n✓ Loaded nodes shapefile: {nodes_2d_shp_file}")
    print(f"  Total nodes: {len(original_nodes_gdf)}")
    print(f"  Columns: {list(original_nodes_gdf.columns)}")

    print(f"\n✓ Loaded edges shapefile: {edges_2d_shp_file}")
    print(f"  Total edges: {len(original_edges_gdf)}")
    print(f"  Columns: {list(original_edges_gdf.columns)}")

    # Set output directory
    if output_dir is None:
        output_dir = os.path.join("boundary_viz2")
    os.makedirs(output_dir, exist_ok=True)

    print(f"\n{'=' * 80}")
    print("FILTERING AND CREATING SHAPEFILES")
    print("=" * 80)

    # Get valid node indices (nodes to keep)
    valid_node_indices = np.where(valid_nodes_mask)[0]
    valid_node_indices_set = set(valid_node_indices)

    # Get boundary node indices
    boundary_node_indices = np.where(boundary_nodes_mask)[0]
    boundary_node_indices_set = set(boundary_node_indices)

    # Get boundary edge indices
    boundary_edge_indices = np.where(boundary_edges_mask)[0]
    boundary_edge_indices_set = set(boundary_edge_indices)

    # --- 1. Filter Nodes: Keep only valid nodes (remove ghosts) ---
    print("\n1. Filtering nodes...")

    # Assuming FID in shapefile corresponds to node index
    nodes_fid_col = (
        "FID"
        if "FID" in original_nodes_gdf.columns
        else original_nodes_gdf.index.name or "index"
    )

    if nodes_fid_col == "FID":
        node_mask = original_nodes_gdf["FID"].isin(valid_node_indices_set)
    else:
        node_mask = original_nodes_gdf.index.isin(valid_node_indices_set)

    filtered_nodes_gdf = original_nodes_gdf[node_mask].copy()

    # Add boundary flag
    if nodes_fid_col == "FID":
        filtered_nodes_gdf["is_boundary"] = filtered_nodes_gdf["FID"].isin(
            boundary_node_indices_set
        )
    else:
        filtered_nodes_gdf["is_boundary"] = filtered_nodes_gdf.index.isin(
            boundary_node_indices_set
        )

    nodes_output = os.path.join(output_dir, "nodes_valid_all.shp")
    filtered_nodes_gdf.to_file(nodes_output)
    print(f"✓ Saved: {nodes_output}")
    print(f"  Total nodes: {len(filtered_nodes_gdf)}")
    print(f"  - Regular nodes: {np.sum(~filtered_nodes_gdf['is_boundary'])}")
    print(f"  - Boundary nodes: {np.sum(filtered_nodes_gdf['is_boundary'])}")

    # --- 2. Boundary Nodes Only ---
    print("\n2. Creating boundary nodes shapefile...")

    if nodes_fid_col == "FID":
        boundary_node_mask = original_nodes_gdf["FID"].isin(boundary_node_indices_set)
    else:
        boundary_node_mask = original_nodes_gdf.index.isin(boundary_node_indices_set)

    boundary_nodes_gdf = original_nodes_gdf[boundary_node_mask].copy()
    boundary_nodes_output = os.path.join(output_dir, "nodes_boundary_only.shp")
    boundary_nodes_gdf.to_file(boundary_nodes_output)
    print(f"✓ Saved: {boundary_nodes_output}")
    print(f"  Boundary nodes: {len(boundary_nodes_gdf)}")

    # --- 3. Filter Edges: Remove edges to ghost nodes, but keep boundary edges ---
    print("\n3. Filtering edges...")

    edges_fid_col = (
        "FID"
        if "FID" in original_edges_gdf.columns
        else original_edges_gdf.index.name or "index"
    )

    # Get edge indices that should be kept
    # Strategy: Keep edges that are either:
    # 1. Valid edges (from valid_edges_mask), OR
    # 2. Boundary edges (even if they connect to ghost nodes)

    valid_edge_indices = np.where(valid_edges_mask)[0]
    valid_edge_indices_set = set(valid_edge_indices)

    # For each edge in original shapefile, check if it should be kept
    if edges_fid_col == "FID":
        # Keep edge if: (FID in valid_edges) OR (FID in boundary_edges)
        edge_mask = original_edges_gdf["FID"].isin(
            valid_edge_indices_set
        ) | original_edges_gdf["FID"].isin(boundary_edge_indices_set)
        filtered_edges_gdf = original_edges_gdf[edge_mask].copy()

        # Add boundary flags
        filtered_edges_gdf["is_boundary"] = filtered_edges_gdf["FID"].isin(
            boundary_edge_indices_set
        )

        # Add inflow/outflow flags for boundary edges
        inflow_edge_indices = np.where(inflow_edges_mask)[0]
        outflow_edge_indices = np.where(outflow_edges_mask)[0]
        filtered_edges_gdf["is_inflow"] = filtered_edges_gdf["FID"].isin(
            set(inflow_edge_indices)
        )
        filtered_edges_gdf["is_outflow"] = filtered_edges_gdf["FID"].isin(
            set(outflow_edge_indices)
        )
    else:
        edge_mask = original_edges_gdf.index.isin(
            valid_edge_indices_set
        ) | original_edges_gdf.index.isin(boundary_edge_indices_set)
        filtered_edges_gdf = original_edges_gdf[edge_mask].copy()

        filtered_edges_gdf["is_boundary"] = filtered_edges_gdf.index.isin(
            boundary_edge_indices_set
        )

        inflow_edge_indices = np.where(inflow_edges_mask)[0]
        outflow_edge_indices = np.where(outflow_edges_mask)[0]
        filtered_edges_gdf["is_inflow"] = filtered_edges_gdf.index.isin(
            set(inflow_edge_indices)
        )
        filtered_edges_gdf["is_outflow"] = filtered_edges_gdf.index.isin(
            set(outflow_edge_indices)
        )

    edges_output = os.path.join(output_dir, "edges_valid_all.shp")
    filtered_edges_gdf.to_file(edges_output)
    print(f"✓ Saved: {edges_output}")
    print(f"  Total edges: {len(filtered_edges_gdf)}")
    print(f"  - Regular edges: {np.sum(~filtered_edges_gdf['is_boundary'])}")
    print(f"  - Boundary edges: {np.sum(filtered_edges_gdf['is_boundary'])}")
    print(f"    - Inflow: {np.sum(filtered_edges_gdf['is_inflow'])}")
    print(f"    - Outflow: {np.sum(filtered_edges_gdf['is_outflow'])}")

    # --- 4. Boundary Edges Only ---
    print("\n4. Creating boundary edges shapefile...")

    if edges_fid_col == "FID":
        boundary_edge_mask = original_edges_gdf["FID"].isin(boundary_edge_indices_set)
        boundary_edges_gdf = original_edges_gdf[boundary_edge_mask].copy()

        # Add inflow/outflow flags
        inflow_edge_indices = np.where(inflow_edges_mask)[0]
        outflow_edge_indices = np.where(outflow_edges_mask)[0]
        boundary_edges_gdf["is_inflow"] = boundary_edges_gdf["FID"].isin(
            set(inflow_edge_indices)
        )
        boundary_edges_gdf["is_outflow"] = boundary_edges_gdf["FID"].isin(
            set(outflow_edge_indices)
        )
    else:
        boundary_edge_mask = original_edges_gdf.index.isin(boundary_edge_indices_set)
        boundary_edges_gdf = original_edges_gdf[boundary_edge_mask].copy()

        inflow_edge_indices = np.where(inflow_edges_mask)[0]
        outflow_edge_indices = np.where(outflow_edges_mask)[0]
        boundary_edges_gdf["is_inflow"] = boundary_edges_gdf.index.isin(
            set(inflow_edge_indices)
        )
        boundary_edges_gdf["is_outflow"] = boundary_edges_gdf.index.isin(
            set(outflow_edge_indices)
        )

    boundary_edges_output = os.path.join(output_dir, "edges_boundary_only.shp")
    boundary_edges_gdf.to_file(boundary_edges_output)
    print(f"✓ Saved: {boundary_edges_output}")
    print(f"  Boundary edges: {len(boundary_edges_gdf)}")
    print(f"    - Inflow: {np.sum(boundary_edges_gdf['is_inflow'])}")
    print(f"    - Outflow: {np.sum(boundary_edges_gdf['is_outflow'])}")

    print(f"\n{'=' * 80}")
    print("SUMMARY")
    print("=" * 80)
    print(f"\nOutput directory: {output_dir}")
    print("\nShapefiles created:")
    print(
        f"  1. nodes_valid_all.shp - {len(filtered_nodes_gdf)} nodes (regular + boundary, no ghosts)"
    )
    print(f"  2. nodes_boundary_only.shp - {len(boundary_nodes_gdf)} boundary nodes")
    print(
        f"  3. edges_valid_all.shp - {len(filtered_edges_gdf)} edges (regular + boundary, no ghost connections)"
    )
    print(f"  4. edges_boundary_only.shp - {len(boundary_edges_gdf)} boundary edges")

    print(f"\n{'=' * 80}")
    print("FILTERING SUMMARY")
    print("=" * 80)
    print("\nNodes:")
    print(f"  Original: {len(original_nodes_gdf)}")
    print(f"  Filtered: {len(filtered_nodes_gdf)}")
    print(
        f"  Removed:  {len(original_nodes_gdf) - len(filtered_nodes_gdf)} (ghost nodes)"
    )

    print("\nEdges:")
    print(f"  Original: {len(original_edges_gdf)}")
    print(f"  Filtered: {len(filtered_edges_gdf)}")
    print(
        f"  Removed:  {len(original_edges_gdf) - len(filtered_edges_gdf)} (edges to ghost nodes)"
    )
    print("  Note: All boundary edges preserved even if connected to ghost nodes")

    print(f"\n{'=' * 80}")
    print("QGIS VISUALIZATION TIPS")
    print("=" * 80)
    print("\n1. Load all 4 shapefiles")
    print("\n2. Style 'nodes_valid_all.shp':")
    print("   - Categorized by 'is_boundary'")
    print("   - False (regular): Small green circles")
    print("   - True (boundary): Larger red circles")
    print("\n3. Style 'edges_valid_all.shp':")
    print("   - Categorized by 'is_boundary'")
    print("   - False (regular): Thin gray lines")
    print("   - True (boundary): Thick orange lines")
    print("\n4. Layer order (bottom to top):")
    print("   - edges_valid_all.shp")
    print("   - edges_boundary_only.shp")
    print("   - nodes_valid_all.shp")
    print("   - nodes_boundary_only.shp")

    print(f"\n{'=' * 80}\n")

    return {
        "nodes_valid": filtered_nodes_gdf,
        "nodes_boundary": boundary_nodes_gdf,
        "edges_valid": filtered_edges_gdf,
        "edges_boundary": boundary_edges_gdf,
    }


def diagnose_boundary_condition_npz(
    boundary_condition_npz_file: str = "boundary_condition_masks.npz",
    constant_values_npz_file: str = "constant_values.npz",
):
    """
    Detailed diagnostic of boundary condition NPZ file to identify issues.
    Works with either boundary_nodes_mask or non_boundary_nodes_mask.
    """

    print("=" * 80)
    print("BOUNDARY CONDITION NPZ DIAGNOSTIC")
    print("=" * 80)

    # Load boundary condition masks
    if not os.path.exists(boundary_condition_npz_file):
        print(f"❌ File not found: {boundary_condition_npz_file}")
        return

    bc_data = np.load(boundary_condition_npz_file)

    print(f"\n📁 Loading: {boundary_condition_npz_file}\n")

    # Load constant values
    if not os.path.exists(constant_values_npz_file):
        print(f"❌ File not found: {constant_values_npz_file}")
        return

    constant_values = np.load(constant_values_npz_file)
    static_nodes = constant_values["static_nodes"]

    print(f"📁 Loading: {constant_values_npz_file}\n")

    # Extract masks - check what's available
    print("=" * 80)
    print("AVAILABLE ARRAYS IN NPZ FILE")
    print("=" * 80)
    print(f"\nArrays in {boundary_condition_npz_file}:")
    for key in bc_data.files:
        print(f"  - {key}")

    # Determine which mask is available and get both versions
    has_boundary_mask = "boundary_nodes_mask" in bc_data
    has_non_boundary_mask = "non_boundary_nodes_mask" in bc_data

    if not has_boundary_mask and not has_non_boundary_mask:
        print(
            "\n❌ ERROR: Neither boundary_nodes_mask nor non_boundary_nodes_mask found in file!"
        )
        return

    # Get the masks (whichever is available, derive the other)
    if has_boundary_mask:
        boundary_nodes_mask = bc_data["boundary_nodes_mask"]
        non_boundary_nodes_mask = ~boundary_nodes_mask
        print("\n✓ Found boundary_nodes_mask, derived non_boundary_nodes_mask")
    else:
        non_boundary_nodes_mask = bc_data["non_boundary_nodes_mask"]
        boundary_nodes_mask = ~non_boundary_nodes_mask
        print("\n✓ Found non_boundary_nodes_mask, derived boundary_nodes_mask")

    # Get edge masks if they exist
    boundary_edges_mask = bc_data.get("boundary_edges_mask", None)
    inflow_edges_mask = bc_data.get("inflow_edges_mask", None)
    outflow_edges_mask = bc_data.get("outflow_edges_mask", None)

    print("\n" + "=" * 80)
    print("1. MASK ARRAYS INFO")
    print("=" * 80)

    print("\nboundary_nodes_mask:")
    print(f"  Shape: {boundary_nodes_mask.shape}")
    print(f"  Dtype: {boundary_nodes_mask.dtype}")
    print(f"  True count: {np.sum(boundary_nodes_mask)}")
    print(f"  Min value: {boundary_nodes_mask.min()}")
    print(f"  Max value: {boundary_nodes_mask.max()}")

    print("\nnon_boundary_nodes_mask:")
    print(f"  Shape: {non_boundary_nodes_mask.shape}")
    print(f"  Dtype: {non_boundary_nodes_mask.dtype}")
    print(f"  True count: {np.sum(non_boundary_nodes_mask)}")
    print(f"  Min value: {non_boundary_nodes_mask.min()}")
    print(f"  Max value: {non_boundary_nodes_mask.max()}")

    if boundary_edges_mask is not None:
        print("\nboundary_edges_mask:")
        print(f"  Shape: {boundary_edges_mask.shape}")
        print(f"  Dtype: {boundary_edges_mask.dtype}")
        print(f"  True count: {np.sum(boundary_edges_mask)}")

    if inflow_edges_mask is not None:
        print("\ninflow_edges_mask:")
        print(f"  Shape: {inflow_edges_mask.shape}")
        print(f"  Dtype: {inflow_edges_mask.dtype}")
        print(f"  True count: {np.sum(inflow_edges_mask)}")

    if outflow_edges_mask is not None:
        print("\noutflow_edges_mask:")
        print(f"  Shape: {outflow_edges_mask.shape}")
        print(f"  Dtype: {outflow_edges_mask.dtype}")
        print(f"  True count: {np.sum(outflow_edges_mask)}")

    # Get node indices
    boundary_node_indices = np.where(boundary_nodes_mask)[0]
    non_boundary_node_indices = np.where(non_boundary_nodes_mask)[0]

    print("\n" + "=" * 80)
    print("2. NODE INDICES")
    print("=" * 80)

    print("\nBoundary nodes:")
    print(f"  Count: {len(boundary_node_indices)}")
    if len(boundary_node_indices) == 0:
        print("  ✓ No boundary nodes (expected for models without boundaries)")
    elif len(boundary_node_indices) <= 20:
        print(f"  Indices: {boundary_node_indices}")
    else:
        print(f"  First 10: {boundary_node_indices[:10]}")
        print(f"  Last 10: {boundary_node_indices[-10:]}")

    print("\nNon-boundary nodes:")
    print(f"  Count: {len(non_boundary_node_indices)}")
    if len(non_boundary_node_indices) <= 20:
        print(f"  Indices: {non_boundary_node_indices}")
    else:
        print(f"  First 10: {non_boundary_node_indices[:10]}")
        print(f"  Last 10: {non_boundary_node_indices[-10:]}")

    print("\n" + "=" * 80)
    print("3. STATIC NODE FEATURES")
    print("=" * 80)

    print("\nstatic_nodes array:")
    print(f"  Shape: {static_nodes.shape}")
    print(f"  Dtype: {static_nodes.dtype}")
    print(f"  Number of nodes: {static_nodes.shape[0]}")
    print(f"  Number of features: {static_nodes.shape[1]}")

    # Assuming standard feature order
    feature_names = [
        "position_x",
        "position_y",
        "area",
        "roughness",
        "elevation",
        "aspect",
        "curvature",
        "flow_accumulation",
    ]

    print(f"\n  Assumed feature order: {feature_names}")

    # Validate that all nodes are accounted for
    print("\n" + "=" * 80)
    print("4. VALIDATION: ALL NODES ACCOUNTED FOR?")
    print("=" * 80)

    total_nodes = len(static_nodes)
    accounted_nodes = len(non_boundary_node_indices) + len(boundary_node_indices)

    print(f"\nTotal nodes in static_nodes: {total_nodes}")
    print(f"Non-boundary nodes: {len(non_boundary_node_indices)}")
    print(f"Boundary nodes: {len(boundary_node_indices)}")
    print(f"Sum of both: {accounted_nodes}")

    if total_nodes == accounted_nodes:
        print("✓ All nodes are accounted for")
    else:
        print(f"❌ ERROR: {total_nodes - accounted_nodes} nodes are unaccounted for!")

    # Show details based on which nodes exist
    if len(boundary_node_indices) > 0:
        print("\n" + "=" * 80)
        print("5. BOUNDARY NODE DETAILS")
        print("=" * 80)

        num_to_show = min(10, len(boundary_node_indices))
        for i, node_idx in enumerate(boundary_node_indices[:num_to_show]):
            print(f"\n--- Boundary Node {i + 1}/{num_to_show} ---")
            print(f"  Node Index: {node_idx}")

            if node_idx >= len(static_nodes):
                print(f"  ❌ ERROR: Index {node_idx} is out of bounds!")
                print(
                    f"  Static nodes array only has {len(static_nodes)} nodes (indices 0-{len(static_nodes) - 1})"
                )
                continue

            node_features = static_nodes[node_idx]
            print("  Features:")
            for feat_idx, feat_name in enumerate(feature_names):
                if feat_idx < len(node_features):
                    print(f"    {feat_name:20s}: {node_features[feat_idx]:.6f}")

            # Check if position is zero
            pos_x = node_features[0]
            pos_y = node_features[1]

            if pos_x == 0.0 and pos_y == 0.0:
                print("  ⚠️  WARNING: Position is (0, 0) - likely incorrect!")

            if np.isnan(pos_x) or np.isnan(pos_y):
                print("  ❌ ERROR: Position contains NaN values!")

        if len(boundary_node_indices) > 10:
            print(f"\n... and {len(boundary_node_indices) - 10} more boundary nodes")

    # Show sample of non-boundary nodes
    print("\n" + "=" * 80)
    print("6. SAMPLE OF NON-BOUNDARY NODES")
    print("=" * 80)

    sample_size = min(5, len(non_boundary_node_indices))
    sample_indices = non_boundary_node_indices[:sample_size]

    for i, node_idx in enumerate(sample_indices):
        print(f"\n--- Non-Boundary Node {i + 1}/{sample_size} ---")
        print(f"  Node Index: {node_idx}")
        node_features = static_nodes[node_idx]
        print("  Features:")
        for feat_idx, feat_name in enumerate(feature_names):
            if feat_idx < len(node_features):
                print(f"    {feat_name:20s}: {node_features[feat_idx]:.6f}")

    # Check for patterns in boundary nodes (if any exist)
    if len(boundary_node_indices) > 0:
        print("\n" + "=" * 80)
        print("7. CHECKING FOR PATTERNS IN BOUNDARY NODES")
        print("=" * 80)

        # Check if all boundary nodes have zero positions
        boundary_positions_x = static_nodes[boundary_node_indices, 0]
        boundary_positions_y = static_nodes[boundary_node_indices, 1]

        num_zero_x = np.sum(boundary_positions_x == 0)
        num_zero_y = np.sum(boundary_positions_y == 0)
        num_both_zero = np.sum(
            (boundary_positions_x == 0) & (boundary_positions_y == 0)
        )

        print("\nBoundary nodes with position issues:")
        print(f"  Position X = 0: {num_zero_x}/{len(boundary_node_indices)}")
        print(f"  Position Y = 0: {num_zero_y}/{len(boundary_node_indices)}")
        print(f"  Both X and Y = 0: {num_both_zero}/{len(boundary_node_indices)}")

        if num_both_zero > 0:
            print(f"\n⚠️  ISSUE: {num_both_zero} boundary nodes have (0, 0) position!")
            print(
                "   This suggests boundary nodes were added with zero/placeholder values."
            )

        # Check if boundary nodes are at the end (appended)
        print("\n" + "=" * 80)
        print("8. CHECKING IF BOUNDARY NODES WERE APPENDED")
        print("=" * 80)

        print(f"\nTotal nodes in static_nodes: {len(static_nodes)}")
        print(f"Maximum boundary node index: {boundary_node_indices.max()}")
        print(f"Minimum boundary node index: {boundary_node_indices.min()}")

        if boundary_node_indices.min() >= len(static_nodes) - len(
            boundary_node_indices
        ):
            print("\n✓ Boundary nodes appear to be appended at the end of the array")
        else:
            print("\n⚠️  Boundary nodes are scattered throughout the array")

    print("\n" + "=" * 80)
    print("9. EDGE MASK STATISTICS")
    print("=" * 80)

    if boundary_edges_mask is not None:
        print(f"\nBoundary edges: {np.sum(boundary_edges_mask)}")
    else:
        print("\nNo boundary_edges_mask found")

    if inflow_edges_mask is not None:
        print(f"Inflow edges: {np.sum(inflow_edges_mask)}")
    else:
        print("No inflow_edges_mask found")

    if outflow_edges_mask is not None:
        print(f"Outflow edges: {np.sum(outflow_edges_mask)}")
    else:
        print("No outflow_edges_mask found")

    print("\n" + "=" * 80)
    print("DIAGNOSTIC COMPLETE")
    print("=" * 80)

    # Summary
    print("\n📊 SUMMARY:")
    print(f"  Total nodes: {total_nodes}")
    print(f"  Boundary nodes: {len(boundary_node_indices)}")
    print(f"  Non-boundary nodes: {len(non_boundary_node_indices)}")

    if len(boundary_node_indices) == 0:
        print("  ✓ Model has no boundary nodes (as expected for some configurations)")
    else:
        print(f"  ℹ️  Model has {len(boundary_node_indices)} boundary nodes")

    if total_nodes == accounted_nodes:
        print("  ✓ All nodes properly classified")
    else:
        print(
            f"  ❌ Classification issue: {total_nodes - accounted_nodes} nodes unaccounted for"
        )


def filter_nodes_by_fid_match(
    source_shapefile: str,
    reference_shapefile: str,
    output_shapefile: Optional[str] = None,
    source_fid_column: str = "FID",
    reference_nodeidx_column: str = "node_idx",
    boundary_nodes: Optional[Union[List[int], np.ndarray]] = None,
    keep_all_boundary: bool = True,
    remapping_json: Optional[str] = None,
    remapping_key: str = "node_remapping",
    removed_nodes_key: str = "removed_nodes",
    add_original_fid: bool = True,
) -> gpd.GeoDataFrame:
    """
    Filter a source shapefile by removing specified nodes and applying ID remapping.
    Optionally preserves boundary nodes (with remapped IDs).

    Correct logic flow:
    1. Load remapping JSON (contains removed_nodes and node_remapping)
    2. Remove nodes whose FID is in removed_nodes list
    3. Apply node_remapping to remap FIDs
    4. If boundary_nodes provided, remap them and ensure they're preserved
    5. Optionally filter by reference shapefile node_idx values

    Args:
        source_shapefile: Path to the shapefile to be filtered (contains FID)
        reference_shapefile: Path to the reference shapefile (contains node_idx)
        output_shapefile: Optional path to save filtered shapefile.
                         If None, returns GeoDataFrame without saving
        source_fid_column: Column name for FID in source shapefile (default: "FID")
        reference_nodeidx_column: Column name for node indices in reference (default: "node_idx")
        boundary_nodes: List or array of ORIGINAL FID values for boundary nodes to preserve
        keep_all_boundary: If True, keeps boundary nodes regardless of other filters (default: True)
        remapping_json: Path to JSON file containing removed_nodes and node_remapping
        remapping_key: Key in JSON for the remapping dict (default: "node_remapping")
        removed_nodes_key: Key in JSON for removed nodes list (default: "removed_nodes")
        add_original_fid: If True, adds 'original_fid' column before remapping (default: True)

    Returns:
        GeoDataFrame: Filtered and remapped nodes
    """

    # Validate input files exist
    if not os.path.exists(source_shapefile):
        raise FileNotFoundError(f"Source shapefile not found: {source_shapefile}")
    if not os.path.exists(reference_shapefile):
        raise FileNotFoundError(f"Reference shapefile not found: {reference_shapefile}")

    print("=" * 80)
    print("FILTERING NODES WITH CORRECT LOGIC")
    print("=" * 80)

    # Load shapefiles
    print(f"\nLoading source shapefile: {source_shapefile}")
    source_gdf = gpd.read_file(source_shapefile)
    print(f"  Total nodes in source: {len(source_gdf)}")
    print(f"  Columns: {list(source_gdf.columns)}")

    print(f"\nLoading reference shapefile: {reference_shapefile}")
    reference_gdf = gpd.read_file(reference_shapefile)
    print(f"  Total nodes in reference: {len(reference_gdf)}")
    print(f"  Columns: {list(reference_gdf.columns)}")

    # Validate required columns exist
    if source_fid_column not in source_gdf.columns:
        raise ValueError(
            f"Column '{source_fid_column}' not found in source shapefile. "
            f"Available columns: {list(source_gdf.columns)}"
        )

    if reference_nodeidx_column not in reference_gdf.columns:
        raise ValueError(
            f"Column '{reference_nodeidx_column}' not found in reference shapefile. "
            f"Available columns: {list(reference_gdf.columns)}"
        )

    # =========================================================================
    # STEP 1: Load remapping data (removed_nodes and node_remapping)
    # =========================================================================
    removed_nodes_list = []
    node_remapping = {}

    if remapping_json:
        print("\n" + "=" * 80)
        print("STEP 1: LOADING REMAPPING DATA")
        print("=" * 80)

        if not os.path.exists(remapping_json):
            print(f"⚠ Warning: Remapping file not found: {remapping_json}")
            print("  Proceeding without remapping")
        else:
            print(f"\nLoading from: {remapping_json}")

            with open(remapping_json, "r") as f:
                remapping_data = json.load(f)

            # Load removed_nodes
            if removed_nodes_key in remapping_data:
                removed_nodes_list = remapping_data[removed_nodes_key]
                print(
                    f"\n✓ Loaded '{removed_nodes_key}': {len(removed_nodes_list)} nodes to remove"
                )
                print(f"  Sample removed nodes: {removed_nodes_list[:10]}")
            else:
                print(f"\n⚠ Warning: Key '{removed_nodes_key}' not found in JSON")

            # Load node_remapping
            if remapping_key in remapping_data:
                node_remapping = remapping_data[remapping_key]
                print(
                    f"\n✓ Loaded '{remapping_key}': {len(node_remapping)} remapping entries"
                )
                print(f"  Sample mappings: {dict(list(node_remapping.items())[:5])}")
            else:
                print(f"\n⚠ Warning: Key '{remapping_key}' not found in JSON")

    # =========================================================================
    # STEP 2: Remove nodes in removed_nodes list
    # =========================================================================
    print("\n" + "=" * 80)
    print("STEP 2: REMOVING SPECIFIED NODES")
    print("=" * 80)

    original_count = len(source_gdf)

    if removed_nodes_list:
        # Convert removed_nodes to set for efficient lookup
        removed_nodes_set = set(removed_nodes_list)

        # Convert FID column to comparable type
        source_fids = source_gdf[source_fid_column].astype(str)
        removed_nodes_str = {str(node) for node in removed_nodes_set}

        # Create mask: keep nodes NOT in removed list
        keep_mask = ~source_fids.isin(removed_nodes_str)

        filtered_gdf = source_gdf[keep_mask].copy()

        removed_count = original_count - len(filtered_gdf)
        print(f"\nRemoved {removed_count:,} nodes from removed_nodes list")
        print(f"Remaining nodes: {len(filtered_gdf):,}")
    else:
        print("\nNo removed_nodes list provided - keeping all nodes")
        filtered_gdf = source_gdf.copy()

    # =========================================================================
    # STEP 3: Remap boundary nodes (if provided)
    # =========================================================================
    remapped_boundary_set = set()

    if boundary_nodes is not None and len(boundary_nodes) > 0:
        print("\n" + "=" * 80)
        print("STEP 3: REMAPPING BOUNDARY NODES")
        print("=" * 80)

        boundary_nodes_array = np.array(boundary_nodes)
        print(f"\nOriginal boundary nodes: {len(boundary_nodes_array)} unique FIDs")
        print(f"  Sample original: {list(boundary_nodes_array[:10])}")

        if node_remapping:
            # Remap each boundary node
            remapped_boundary = []
            for orig_fid in boundary_nodes_array:
                orig_fid_str = str(orig_fid)
                if orig_fid_str in node_remapping:
                    remapped_fid = node_remapping[orig_fid_str]
                    remapped_boundary.append(remapped_fid)
                else:
                    # Keep original if no mapping found
                    remapped_boundary.append(orig_fid)

            remapped_boundary_set = set(str(x) for x in remapped_boundary)
            print(
                f"\n✓ Remapped boundary nodes to: {len(remapped_boundary_set)} unique FIDs"
            )
            print(f"  Sample remapped: {list(remapped_boundary_set)[:10]}")
        else:
            # No remapping available, use original boundary nodes
            remapped_boundary_set = set(str(x) for x in boundary_nodes_array)
            print("\n⚠ No node_remapping available - using original boundary node IDs")

    # =========================================================================
    # STEP 4: Apply node remapping to FID column
    # =========================================================================
    print("\n" + "=" * 80)
    print("STEP 4: APPLYING NODE ID REMAPPING")
    print("=" * 80)

    if node_remapping:
        # Save original FID if requested
        if add_original_fid:
            filtered_gdf["original_fid"] = filtered_gdf[source_fid_column].copy()
            print("\n✓ Saved original FIDs to 'original_fid' column")

        # Apply remapping
        remapped_count = 0
        unmapped_count = 0
        unmapped_samples = []

        new_fids = []
        for old_fid in filtered_gdf[source_fid_column]:
            old_fid_str = str(old_fid)

            if old_fid_str in node_remapping:
                new_fid = node_remapping[old_fid_str]
                new_fids.append(new_fid)
                remapped_count += 1
            else:
                # Keep original FID if no mapping found
                new_fids.append(old_fid)
                unmapped_count += 1
                if len(unmapped_samples) < 10:
                    unmapped_samples.append(old_fid)

        # Update FID column with remapped values
        filtered_gdf[source_fid_column] = new_fids

        print("\nRemapping results:")
        print(f"  Nodes remapped:     {remapped_count:,}")
        print(f"  Nodes not in map:   {unmapped_count:,}")
        if unmapped_samples:
            print(f"  Sample unmapped:    {unmapped_samples}")
    else:
        print("\nNo node_remapping provided - keeping original FIDs")

    # =========================================================================
    # STEP 5: Ensure boundary nodes are preserved (after remapping)
    # =========================================================================
    if keep_all_boundary and remapped_boundary_set:
        print("\n" + "=" * 80)
        print("STEP 5: ENSURING BOUNDARY NODES ARE PRESERVED")
        print("=" * 80)

        # Check which remapped boundary nodes are in the current filtered set
        current_fids_str = set(str(x) for x in filtered_gdf[source_fid_column])
        boundary_in_filtered = remapped_boundary_set & current_fids_str
        boundary_missing = remapped_boundary_set - current_fids_str

        print("\nBoundary nodes status (using remapped IDs):")
        print(f"  Already in filtered set: {len(boundary_in_filtered):,}")
        print(f"  Missing from filtered set: {len(boundary_missing):,}")

        if boundary_missing:
            print(f"  Sample missing: {list(boundary_missing)[:10]}")

            # Find these nodes in the original (after step 2 removal)
            # Need to search in source_gdf before remapping was applied
            # This is tricky - we need the original data before remapping
            print("\n⚠ Some boundary nodes were removed in previous steps")
            print(
                "  These nodes cannot be recovered as they were in removed_nodes list"
            )

    # =========================================================================
    # STEP 6: Optional reference shapefile filtering
    # =========================================================================
    print("\n" + "=" * 80)
    print("STEP 6: OPTIONAL REFERENCE FILTERING")
    print("=" * 80)

    # Get valid node indices from reference shapefile
    reference_values = reference_gdf[reference_nodeidx_column].astype(str)
    valid_reference_set = set(reference_values)

    print(f"\nValid node_idx values in reference: {len(valid_reference_set)}")

    # Check how many of our nodes match the reference
    current_fids_str = filtered_gdf[source_fid_column].astype(str)
    nodes_in_reference = current_fids_str.isin(valid_reference_set).sum()
    nodes_not_in_reference = len(filtered_gdf) - nodes_in_reference

    print(f"Current nodes matching reference: {nodes_in_reference:,}")
    print(f"Current nodes NOT in reference: {nodes_not_in_reference:,}")

    # Optionally filter by reference (you can enable this if needed)
    # filtered_gdf = filtered_gdf[current_fids_str.isin(valid_reference_set)]

    # =========================================================================
    # FINAL STATISTICS
    # =========================================================================
    print("\n" + "=" * 80)
    print("FINAL RESULTS")
    print("=" * 80)

    final_count = len(filtered_gdf)
    total_removed = original_count - final_count

    print(f"\nOriginal nodes:     {original_count:,}")
    print(f"Final nodes:        {final_count:,}")
    print(f"Total removed:      {total_removed:,}")
    print(f"Retention rate:     {(final_count / original_count) * 100:.2f}%")

    # =========================================================================
    # SAVE OUTPUT
    # =========================================================================
    if output_shapefile:
        output_dir = os.path.dirname(output_shapefile)
        if output_dir and not os.path.exists(output_dir):
            os.makedirs(output_dir, exist_ok=True)

        print(f"\nSaving filtered shapefile to: {output_shapefile}")
        filtered_gdf.to_file(output_shapefile)
        print("✓ File saved successfully")

        if node_remapping:
            print(f"  {source_fid_column} column: Contains remapped node IDs")
            if add_original_fid:
                print("  original_fid column: Contains original FIDs")

    print("\n" + "=" * 80 + "\n")

    return filtered_gdf


def filter_edges_by_node_existence(
    source_edges_shapefile: str,
    reference_nodes_shapefile: str,
    output_shapefile: Optional[str] = None,
    edge_from_node_column: str = "from_node",
    edge_to_node_column: str = "to_node",
    node_id_column: str = "FID",
    edge_fid_column: str = "FID",
    remapping_json: Optional[str] = None,
    edge_remapping_key: str = "edge_remapping",
    node_remapping_key: str = "node_remapping",
    removed_edges_key: str = "removed_edges",
    add_original_ids: bool = True,
) -> gpd.GeoDataFrame:
    """
    Filter edges by removing specified edges, remapping node IDs, filtering by node existence,
    and finally remapping edge IDs.

    Correct logic flow:
    1. Load remapping JSON (contains removed_edges, node_remapping, edge_remapping)
    2. Remove edges whose FID is in removed_edges list
    3. Remap from_node and to_node using node_remapping
    4. Filter edges to keep only those where BOTH nodes exist in reference
    5. Remap edge FIDs using edge_remapping
    6. Save result

    Args:
        source_edges_shapefile: Path to the edges shapefile to be filtered
        reference_nodes_shapefile: Path to the nodes shapefile containing valid nodes
        output_shapefile: Optional path to save filtered edges shapefile
        edge_from_node_column: Column name for source node in edges (default: "from_node")
        edge_to_node_column: Column name for target node in edges (default: "to_node")
        node_id_column: Column name for node identifier in nodes shapefile (default: "FID")
        edge_fid_column: Column name for edge FID in edges shapefile (default: "FID")
        remapping_json: Path to JSON file containing removed_edges, node_remapping, edge_remapping
        edge_remapping_key: Key in JSON for edge remapping (default: "edge_remapping")
        node_remapping_key: Key in JSON for node remapping (default: "node_remapping")
        removed_edges_key: Key in JSON for removed edges list (default: "removed_edges")
        add_original_ids: If True, adds columns for original IDs (default: True)

    Returns:
        GeoDataFrame: Filtered and remapped edges
    """

    # Validate input files exist
    if not os.path.exists(source_edges_shapefile):
        raise FileNotFoundError(
            f"Source edges shapefile not found: {source_edges_shapefile}"
        )
    if not os.path.exists(reference_nodes_shapefile):
        raise FileNotFoundError(
            f"Reference nodes shapefile not found: {reference_nodes_shapefile}"
        )

    print("=" * 80)
    print("FILTERING EDGES WITH CORRECT LOGIC")
    print("=" * 80)

    # Load shapefiles
    print(f"\nLoading edges shapefile: {source_edges_shapefile}")
    edges_gdf = gpd.read_file(source_edges_shapefile)
    print(f"  Total edges: {len(edges_gdf)}")
    print(f"  Columns: {list(edges_gdf.columns)}")

    print(f"\nLoading nodes shapefile: {reference_nodes_shapefile}")
    nodes_gdf = gpd.read_file(reference_nodes_shapefile)
    print(f"  Total nodes: {len(nodes_gdf)}")
    print(f"  Columns: {list(nodes_gdf.columns)}")

    # Validate required columns exist
    if edge_from_node_column not in edges_gdf.columns:
        raise ValueError(
            f"Column '{edge_from_node_column}' not found in edges shapefile. "
            f"Available columns: {list(edges_gdf.columns)}"
        )

    if edge_to_node_column not in edges_gdf.columns:
        raise ValueError(
            f"Column '{edge_to_node_column}' not found in edges shapefile. "
            f"Available columns: {list(edges_gdf.columns)}"
        )

    if node_id_column not in nodes_gdf.columns:
        raise ValueError(
            f"Column '{node_id_column}' not found in nodes shapefile. "
            f"Available columns: {list(nodes_gdf.columns)}"
        )

    if edge_fid_column not in edges_gdf.columns:
        raise ValueError(
            f"Column '{edge_fid_column}' not found in edges shapefile. "
            f"Available columns: {list(edges_gdf.columns)}"
        )

    # =========================================================================
    # STEP 1: Load remapping data
    # =========================================================================
    removed_edges_list = []
    node_remapping = {}
    edge_remapping = {}

    if remapping_json:
        print("\n" + "=" * 80)
        print("STEP 1: LOADING REMAPPING DATA")
        print("=" * 80)

        if not os.path.exists(remapping_json):
            print(f"⚠ Warning: Remapping file not found: {remapping_json}")
            print("  Proceeding without remapping")
        else:
            print(f"\nLoading from: {remapping_json}")

            with open(remapping_json, "r") as f:
                remapping_data = json.load(f)

            # Load removed_edges
            if removed_edges_key in remapping_data:
                removed_edges_list = remapping_data[removed_edges_key]
                print(
                    f"\n✓ Loaded '{removed_edges_key}': {len(removed_edges_list)} edges to remove"
                )
                print(f"  Sample removed edges: {removed_edges_list[:10]}")
            else:
                print(f"\n⚠ Warning: Key '{removed_edges_key}' not found in JSON")

            # Load node_remapping
            if node_remapping_key in remapping_data:
                node_remapping = remapping_data[node_remapping_key]
                print(
                    f"\n✓ Loaded '{node_remapping_key}': {len(node_remapping)} node remapping entries"
                )
                print(
                    f"  Sample node mappings: {dict(list(node_remapping.items())[:5])}"
                )
            else:
                print(f"\n⚠ Warning: Key '{node_remapping_key}' not found in JSON")

            # Load edge_remapping
            if edge_remapping_key in remapping_data:
                edge_remapping = remapping_data[edge_remapping_key]
                print(
                    f"\n✓ Loaded '{edge_remapping_key}': {len(edge_remapping)} edge remapping entries"
                )
                print(
                    f"  Sample edge mappings: {dict(list(edge_remapping.items())[:5])}"
                )
            else:
                print(f"\n⚠ Warning: Key '{edge_remapping_key}' not found in JSON")

    # =========================================================================
    # STEP 2: Remove edges in removed_edges list
    # =========================================================================
    print("\n" + "=" * 80)
    print("STEP 2: REMOVING SPECIFIED EDGES")
    print("=" * 80)

    original_count = len(edges_gdf)

    if removed_edges_list:
        # Convert removed_edges to set for efficient lookup
        removed_edges_set = set(removed_edges_list)

        # Convert edge FID column to comparable type
        edge_fids = edges_gdf[edge_fid_column].astype(str)
        removed_edges_str = {str(edge) for edge in removed_edges_set}

        # Create mask: keep edges NOT in removed list
        keep_mask = ~edge_fids.isin(removed_edges_str)

        filtered_edges_gdf = edges_gdf[keep_mask].copy()

        removed_count = original_count - len(filtered_edges_gdf)
        print(f"\nRemoved {removed_count:,} edges from removed_edges list")
        print(f"Remaining edges: {len(filtered_edges_gdf):,}")
    else:
        print("\nNo removed_edges list provided - keeping all edges")
        filtered_edges_gdf = edges_gdf.copy()

    # =========================================================================
    # STEP 3: Remap from_node and to_node using node_remapping
    # =========================================================================
    print("\n" + "=" * 80)
    print("STEP 3: REMAPPING NODE IDs (from_node and to_node)")
    print("=" * 80)

    if node_remapping:
        # Save original node IDs if requested
        if add_original_ids:
            filtered_edges_gdf["orig_from"] = filtered_edges_gdf[
                edge_from_node_column
            ].copy()
            filtered_edges_gdf["orig_to"] = filtered_edges_gdf[
                edge_to_node_column
            ].copy()
            print("\n✓ Saved original node IDs to 'orig_from' and 'orig_to' columns")

        # Remap from_node
        from_remapped_count = 0
        from_unmapped_count = 0
        from_unmapped_samples = []
        new_from_nodes = []

        for old_node in filtered_edges_gdf[edge_from_node_column]:
            old_node_str = str(old_node)

            if old_node_str in node_remapping:
                new_node = node_remapping[old_node_str]
                new_from_nodes.append(new_node)
                from_remapped_count += 1
            else:
                new_from_nodes.append(old_node)
                from_unmapped_count += 1
                if len(from_unmapped_samples) < 10:
                    from_unmapped_samples.append(old_node)

        filtered_edges_gdf[edge_from_node_column] = new_from_nodes

        # Remap to_node
        to_remapped_count = 0
        to_unmapped_count = 0
        to_unmapped_samples = []
        new_to_nodes = []

        for old_node in filtered_edges_gdf[edge_to_node_column]:
            old_node_str = str(old_node)

            if old_node_str in node_remapping:
                new_node = node_remapping[old_node_str]
                new_to_nodes.append(new_node)
                to_remapped_count += 1
            else:
                new_to_nodes.append(old_node)
                to_unmapped_count += 1
                if len(to_unmapped_samples) < 10:
                    to_unmapped_samples.append(old_node)

        filtered_edges_gdf[edge_to_node_column] = new_to_nodes

        print("\nNode ID remapping results:")
        print(f"  from_node remapped: {from_remapped_count:,}")
        print(f"  from_node unmapped: {from_unmapped_count:,}")
        if from_unmapped_samples:
            print(f"    Sample unmapped from_node: {from_unmapped_samples}")
        print(f"  to_node remapped:   {to_remapped_count:,}")
        print(f"  to_node unmapped:   {to_unmapped_count:,}")
        if to_unmapped_samples:
            print(f"    Sample unmapped to_node: {to_unmapped_samples}")
    else:
        print("\nNo node_remapping provided - keeping original node IDs")

    # =========================================================================
    # STEP 4: Filter edges by node existence (both endpoints must exist)
    # =========================================================================
    print("\n" + "=" * 80)
    print("STEP 4: FILTERING EDGES BY NODE EXISTENCE")
    print("=" * 80)

    # Get set of valid node IDs from reference (these should already be remapped)
    valid_node_ids = nodes_gdf[node_id_column].values
    print(f"\nValid node IDs in reference: {len(valid_node_ids)}")

    # Get edge connectivity (now with remapped node IDs)
    from_nodes = filtered_edges_gdf[edge_from_node_column].values
    to_nodes = filtered_edges_gdf[edge_to_node_column].values

    print("\nColumn types:")
    print(f"  Edges {edge_from_node_column}: {from_nodes.dtype}")
    print(f"  Edges {edge_to_node_column}: {to_nodes.dtype}")
    print(f"  Nodes {node_id_column}: {valid_node_ids.dtype}")

    # Convert to consistent types for comparison
    try:
        valid_node_ids_int = valid_node_ids.astype("int64")
        from_nodes_int = from_nodes.astype("int64")
        to_nodes_int = to_nodes.astype("int64")
        print("  ✓ Successfully converted all to int64 for comparison")
    except (ValueError, TypeError):
        # If int conversion fails, use string comparison
        print("  ⚠ Int conversion failed, using string comparison")
        valid_node_ids_int = valid_node_ids.astype(str)
        from_nodes_int = from_nodes.astype(str)
        to_nodes_int = to_nodes.astype(str)

    valid_node_ids_set = set(valid_node_ids_int)

    # Create masks for valid connections
    print("\nChecking edge connectivity...")
    from_node_exists = np.isin(from_nodes_int, list(valid_node_ids_set))
    to_node_exists = np.isin(to_nodes_int, list(valid_node_ids_set))

    # Keep edge only if BOTH from_node AND to_node exist
    both_nodes_exist = from_node_exists & to_node_exists

    # Statistics before filtering
    edges_before_filter = len(filtered_edges_gdf)
    print("\nConnectivity analysis:")
    print(f"  Edges with valid from_node: {np.sum(from_node_exists):,}")
    print(f"  Edges with valid to_node:   {np.sum(to_node_exists):,}")
    print(f"  Edges with BOTH nodes valid: {np.sum(both_nodes_exist):,}")

    # Identify problematic edges
    only_from_invalid = ~from_node_exists & to_node_exists
    only_to_invalid = from_node_exists & ~to_node_exists
    both_invalid = ~from_node_exists & ~to_node_exists

    print("\nEdges to be removed:")
    print(f"  Missing from_node only:  {np.sum(only_from_invalid):,}")
    print(f"  Missing to_node only:    {np.sum(only_to_invalid):,}")
    print(f"  Missing both nodes:      {np.sum(both_invalid):,}")
    print(f"  Total to remove:         {np.sum(~both_nodes_exist):,}")

    # Show sample of problematic edges
    if np.sum(~both_nodes_exist) > 0:
        problematic_edges = filtered_edges_gdf[~both_nodes_exist]
        print("\nSample of edges being removed (first 5):")
        sample_size = min(5, len(problematic_edges))
        for i in range(sample_size):
            edge_idx = problematic_edges.index[i]
            from_node = problematic_edges.iloc[i][edge_from_node_column]
            to_node = problematic_edges.iloc[i][edge_to_node_column]
            edge_fid = problematic_edges.iloc[i][edge_fid_column]
            from_exists = from_node in valid_node_ids_set or str(from_node) in {
                str(x) for x in valid_node_ids_set
            }
            to_exists = to_node in valid_node_ids_set or str(to_node) in {
                str(x) for x in valid_node_ids_set
            }
            print(
                f"  Edge FID={edge_fid}: from={from_node} (exists: {from_exists}), to={to_node} (exists: {to_exists})"
            )

    # Filter edges
    filtered_edges_gdf = filtered_edges_gdf[both_nodes_exist].copy()

    edges_after_filter = len(filtered_edges_gdf)
    edges_removed_by_connectivity = edges_before_filter - edges_after_filter

    print(f"\nEdges before connectivity filter: {edges_before_filter:,}")
    print(f"Edges after connectivity filter:  {edges_after_filter:,}")
    print(f"Edges removed by connectivity:    {edges_removed_by_connectivity:,}")

    # =========================================================================
    # STEP 5: Remap edge FIDs using edge_remapping
    # =========================================================================
    print("\n" + "=" * 80)
    print("STEP 5: REMAPPING EDGE FIDs")
    print("=" * 80)

    if edge_remapping:
        # Save original edge FID if requested
        if add_original_ids:
            filtered_edges_gdf["original_fid"] = filtered_edges_gdf[
                edge_fid_column
            ].copy()
            print("\n✓ Saved original edge FIDs to 'original_fid' column")

        # Apply edge remapping
        remapped_count = 0
        unmapped_count = 0
        unmapped_samples = []

        new_edge_fids = []
        for old_fid in filtered_edges_gdf[edge_fid_column]:
            old_fid_str = str(old_fid)

            if old_fid_str in edge_remapping:
                new_fid = edge_remapping[old_fid_str]
                new_edge_fids.append(new_fid)
                remapped_count += 1
            else:
                new_edge_fids.append(old_fid)
                unmapped_count += 1
                if len(unmapped_samples) < 10:
                    unmapped_samples.append(old_fid)

        filtered_edges_gdf[edge_fid_column] = new_edge_fids

        print("\nEdge FID remapping results:")
        print(f"  Edges remapped:     {remapped_count:,}")
        print(f"  Edges not in map:   {unmapped_count:,}")
        if unmapped_samples:
            print(f"  Sample unmapped:    {unmapped_samples}")
    else:
        print("\nNo edge_remapping provided - keeping original edge FIDs")

    # =========================================================================
    # FINAL STATISTICS
    # =========================================================================
    print("\n" + "=" * 80)
    print("FINAL RESULTS")
    print("=" * 80)

    final_count = len(filtered_edges_gdf)
    total_removed = original_count - final_count

    print(f"\nOriginal edges:         {original_count:,}")
    print(f"Final edges:            {final_count:,}")
    print(f"Total removed:          {total_removed:,}")
    print(f"Retention rate:         {(final_count / original_count) * 100:.2f}%")

    if removed_edges_list:
        removed_by_list = original_count - (
            len(edges_gdf)
            if "edges_before_filter" not in locals()
            else edges_before_filter + edges_removed_by_connectivity
        )
        print("\nBreakdown:")
        print(
            f"  Removed by removed_edges list: {original_count - len(edges_gdf) if removed_edges_list else 0:,}"
        )
        print(f"  Removed by connectivity check: {edges_removed_by_connectivity:,}")

    # =========================================================================
    # SAVE OUTPUT
    # =========================================================================
    if output_shapefile:
        output_dir = os.path.dirname(output_shapefile)
        if output_dir and not os.path.exists(output_dir):
            os.makedirs(output_dir, exist_ok=True)

        print("\n" + "=" * 80)
        print("SAVING RESULTS")
        print("=" * 80)
        print(f"\nSaving filtered shapefile to: {output_shapefile}")
        filtered_edges_gdf.to_file(output_shapefile)
        print("✓ File saved successfully")

        print("\nFinal column mapping:")
        print(f"  {edge_fid_column}: Remapped edge IDs")
        print(f"  {edge_from_node_column}: Remapped source node IDs")
        print(f"  {edge_to_node_column}: Remapped target node IDs")

        if add_original_ids and (node_remapping or edge_remapping):
            print("\nOriginal IDs preserved in:")
            if "original_fid" in filtered_edges_gdf.columns:
                print("  - original_fid: Original edge FIDs")
            if "orig_from" in filtered_edges_gdf.columns:
                print("  - orig_from: Original from_node IDs")
            if "orig_to" in filtered_edges_gdf.columns:
                print("  - orig_to: Original to_node IDs")

    print("\n" + "=" * 80 + "\n")

    return filtered_edges_gdf


def amend_csv_indices(
    csv_path: str,
    remapping_path: str,
    output_path: Optional[str] = None,
    node_column: str = "node_idx",
    edge_column: Optional[str] = "edge_idx",
) -> str:
    """
    Load a CSV file, amend node and edge indices using remapping info, and save the result.

    Correct logic flow:
    1. Load CSV and remapping data
    2. Remap node indices using node_remapping
    3. Remap edge indices using edge_remapping
    4. Remove rows where remapped node_idx is in removed_nodes
    5. Remove rows where remapped edge_idx is in removed_edges
    6. Save result

    The key insight: removed_nodes and removed_edges contain ORIGINAL indices,
    so we must remap first, then filter based on which original indices were removed.

    Args:
        csv_path: Path to input CSV file containing node_idx and/or edge_idx columns
        remapping_path: Path to JSON file containing remapping information
        output_path: Optional custom output path. If None, adds '_amended' suffix to input filename
        node_column: Name of the node index column (default: 'node_idx')
        edge_column: Name of the edge index column (default: 'edge_idx'). Set to None if not present.

    Returns:
        Path to the saved amended CSV file

    Raises:
        FileNotFoundError: If CSV or remapping file not found
        ValueError: If required columns not found or indices can't be remapped
    """

    print("\n" + "=" * 80)
    print("AMENDING CSV INDICES WITH CORRECT LOGIC")
    print("=" * 80)

    # =========================================================================
    # STEP 1: Load remapping info and CSV
    # =========================================================================
    print(f"\nLoading remapping from: {remapping_path}")
    with open(remapping_path, "r") as f:
        remapping_info = json.load(f)

    # Convert string keys to integers for lookups
    node_remapping = {
        int(k): v for k, v in remapping_info.get("node_remapping", {}).items()
    }
    edge_remapping = {
        int(k): v for k, v in remapping_info.get("edge_remapping", {}).items()
    }
    removed_nodes = set(remapping_info.get("removed_nodes", []))
    removed_edges = set(remapping_info.get("removed_edges", []))

    print(f"  ✓ Loaded {len(node_remapping)} node mappings")
    print(f"  ✓ Loaded {len(edge_remapping)} edge mappings")
    print(f"  ✓ {len(removed_nodes)} removed nodes")
    print(f"  ✓ {len(removed_edges)} removed edges")

    print(f"\nLoading CSV from: {csv_path}")
    df = pd.read_csv(csv_path)
    original_row_count = len(df)
    print(f"  ✓ Loaded {original_row_count} rows")
    print(f"  Columns: {list(df.columns)}")

    # Check if node and edge columns exist
    has_nodes = node_column in df.columns
    has_edges = edge_column is not None and edge_column in df.columns

    if not has_nodes and not has_edges:
        raise ValueError(
            f"CSV must contain at least one of '{node_column}' or '{edge_column}' columns. "
            f"Found columns: {list(df.columns)}"
        )

    # =========================================================================
    # STEP 2: Remap node indices (before filtering)
    # =========================================================================
    if has_nodes:
        print("\n" + "=" * 80)
        print(f"STEP 2: REMAPPING NODE INDICES (column: '{node_column}')")
        print("=" * 80)

        # Track original values for filtering later
        df["_original_node_idx"] = df[node_column].copy()

        # Remap node indices
        unmapped_nodes = []
        remapped_count = 0

        def remap_node(old_idx):
            nonlocal remapped_count
            if old_idx in node_remapping:
                remapped_count += 1
                return node_remapping[old_idx]
            else:
                unmapped_nodes.append(old_idx)
                return old_idx  # Keep original if not in mapping

        df[node_column] = df["_original_node_idx"].apply(remap_node)

        print("\nNode remapping results:")
        print(f"  Remapped: {remapped_count}")
        print(f"  Unmapped: {len(unmapped_nodes)}")

        if unmapped_nodes:
            unique_unmapped = set(unmapped_nodes)
            print(f"  Unique unmapped nodes: {len(unique_unmapped)}")
            print(f"    Sample: {sorted(list(unique_unmapped))[:10]}")
            if len(unique_unmapped) > 10:
                print(f"    ... and {len(unique_unmapped) - 10} more")

    # =========================================================================
    # STEP 3: Remap edge indices (before filtering)
    # =========================================================================
    if has_edges:
        print("\n" + "=" * 80)
        print(f"STEP 3: REMAPPING EDGE INDICES (column: '{edge_column}')")
        print("=" * 80)

        # Track original values for filtering later
        df["_original_edge_idx"] = df[edge_column].copy()

        # Remap edge indices
        unmapped_edges = []
        remapped_count = 0

        def remap_edge(old_idx):
            nonlocal remapped_count
            if old_idx in edge_remapping:
                remapped_count += 1
                return edge_remapping[old_idx]
            else:
                unmapped_edges.append(old_idx)
                return old_idx  # Keep original if not in mapping

        df[edge_column] = df["_original_edge_idx"].apply(remap_edge)

        print("\nEdge remapping results:")
        print(f"  Remapped: {remapped_count}")
        print(f"  Unmapped: {len(unmapped_edges)}")

        if unmapped_edges:
            unique_unmapped = set(unmapped_edges)
            print(f"  Unique unmapped edges: {len(unique_unmapped)}")
            print(f"    Sample: {sorted(list(unique_unmapped))[:10]}")
            if len(unique_unmapped) > 10:
                print(f"    ... and {len(unique_unmapped) - 10} more")

    # =========================================================================
    # STEP 4: Filter out rows with removed nodes (based on ORIGINAL indices)
    # =========================================================================
    if has_nodes and removed_nodes:
        print("\n" + "=" * 80)
        print("STEP 4: FILTERING REMOVED NODES")
        print("=" * 80)

        # Check which ORIGINAL node indices are in the removed list
        rows_before = len(df)
        original_nodes_in_csv = set(df["_original_node_idx"].unique())
        removed_in_csv = original_nodes_in_csv & removed_nodes

        if removed_in_csv:
            print(
                f"\nFound {len(removed_in_csv)} removed nodes (original indices) in CSV"
            )
            print(f"  Sample removed nodes: {sorted(list(removed_in_csv))[:10]}")
            if len(removed_in_csv) > 10:
                print(f"  ... and {len(removed_in_csv) - 10} more")

            # Filter out rows where ORIGINAL node index is in removed_nodes
            df = df[~df["_original_node_idx"].isin(removed_nodes)].copy()
            rows_after = len(df)
            rows_removed = rows_before - rows_after

            print("\nFiltering results:")
            print(f"  Rows before: {rows_before:,}")
            print(f"  Rows after:  {rows_after:,}")
            print(f"  Rows removed: {rows_removed:,}")
        else:
            print("\nNo removed nodes found in this CSV - keeping all rows")

    # =========================================================================
    # STEP 5: Filter out rows with removed edges (based on ORIGINAL indices)
    # =========================================================================
    if has_edges and removed_edges:
        print("\n" + "=" * 80)
        print("STEP 5: FILTERING REMOVED EDGES")
        print("=" * 80)

        # Check which ORIGINAL edge indices are in the removed list
        rows_before = len(df)
        original_edges_in_csv = set(df["_original_edge_idx"].unique())
        removed_in_csv = original_edges_in_csv & removed_edges

        if removed_in_csv:
            print(
                f"\nFound {len(removed_in_csv)} removed edges (original indices) in CSV"
            )
            print(f"  Sample removed edges: {sorted(list(removed_in_csv))[:10]}")
            if len(removed_in_csv) > 10:
                print(f"  ... and {len(removed_in_csv) - 10} more")

            # Filter out rows where ORIGINAL edge index is in removed_edges
            df = df[~df["_original_edge_idx"].isin(removed_edges)].copy()
            rows_after = len(df)
            rows_removed = rows_before - rows_after

            print("\nFiltering results:")
            print(f"  Rows before: {rows_before:,}")
            print(f"  Rows after:  {rows_after:,}")
            print(f"  Rows removed: {rows_removed:,}")
        else:
            print("\nNo removed edges found in this CSV - keeping all rows")

    # =========================================================================
    # STEP 6: Clean up temporary columns
    # =========================================================================
    # Remove temporary tracking columns
    if "_original_node_idx" in df.columns:
        df = df.drop(columns=["_original_node_idx"])
    if "_original_edge_idx" in df.columns:
        df = df.drop(columns=["_original_edge_idx"])

    # =========================================================================
    # STEP 7: Save results
    # =========================================================================
    # Determine output path
    if output_path is None:
        csv_file = Path(csv_path)
        output_path = csv_file.parent / f"{csv_file.stem}_amended{csv_file.suffix}"

    output_path = str(output_path)

    print("\n" + "=" * 80)
    print("SAVING RESULTS")
    print("=" * 80)
    print(f"\nSaving amended CSV to: {output_path}")
    df.to_csv(output_path, index=False)
    print(f"  ✓ Saved {len(df)} rows")

    # =========================================================================
    # FINAL SUMMARY
    # =========================================================================
    print("\n" + "=" * 80)
    print("FINAL SUMMARY")
    print("=" * 80)
    print(f"Input file:  {csv_path}")
    print(f"Output file: {output_path}")
    print(
        f"Rows: {original_row_count:,} -> {len(df):,} (removed {original_row_count - len(df):,})"
    )

    if has_nodes:
        print(f"✓ Node indices remapped (column: '{node_column}')")
    if has_edges:
        print(f"✓ Edge indices remapped (column: '{edge_column}')")

    print("=" * 80 + "\n")

    return output_path


def batch_amend_csv_indices(
    main_folder: str,
    filename_pattern: str,
    remapping_path: str,
    node_column: str = "node_idx",
    edge_column: Optional[str] = "edge_idx",
    recursive: bool = True,
) -> list[str]:
    """
    Search for CSV files matching a pattern across multiple folders and amend their indices.

    Args:
        main_folder: Root folder to search in
        filename_pattern: Filename pattern to match (supports wildcards like '*.csv' or 'results_*.csv')
        remapping_path: Path to JSON file containing remapping information
        node_column: Name of the node index column (default: 'node_idx')
        edge_column: Name of the edge index column (default: 'edge_idx')
        recursive: If True, search in all subdirectories; if False, only search main_folder

    Returns:
        List of paths to saved amended CSV files

    Example:
        # Search for all 'simulation_results.csv' files in all subdirectories
        output_paths = bc.batch_amend_csv_indices(
            main_folder='data/',
            filename_pattern='simulation_results.csv',
            remapping_path='remapping_info.json'
        )

        # Search for all CSV files starting with 'timestep_'
        output_paths = bc.batch_amend_csv_indices(
            main_folder='data/',
            filename_pattern='timestep_*.csv',
            remapping_path='remapping_info.json'
        )
    """

    print("\n" + "=" * 80)
    print("BATCH CSV INDEX AMENDMENT WITH FILE SEARCH")
    print("=" * 80)

    # Convert to Path object
    main_path = Path(main_folder)

    if not main_path.exists():
        raise FileNotFoundError(f"Main folder not found: {main_folder}")

    if not main_path.is_dir():
        raise ValueError(f"Path is not a directory: {main_folder}")

    # Search for matching files
    print(f"\nSearching for files matching: '{filename_pattern}'")
    print(f"In folder: {main_path.absolute()}")
    print(f"Recursive search: {recursive}")
    print("-" * 80)

    if recursive:
        # Search recursively using rglob
        csv_paths = list(main_path.rglob(filename_pattern))
    else:
        # Search only in main folder using glob
        csv_paths = list(main_path.glob(filename_pattern))

    # Convert to strings and sort
    csv_paths = sorted([str(p) for p in csv_paths])

    if not csv_paths:
        print(f"⚠ WARNING: No files found matching pattern '{filename_pattern}'")
        print("=" * 80 + "\n")
        return []

    print(f"\n✓ Found {len(csv_paths)} matching file(s):")
    for i, path in enumerate(csv_paths, 1):
        rel_path = Path(path).relative_to(main_path)
        print(f"  [{i}] {rel_path}")

    # Process each file
    print("\n" + "=" * 80)
    print(f"PROCESSING {len(csv_paths)} CSV FILES")
    print("=" * 80)

    output_paths = []

    for i, csv_path in enumerate(csv_paths, 1):
        print(f"\n[{i}/{len(csv_paths)}] Processing: {Path(csv_path).name}")
        print(f"Location: {Path(csv_path).parent}")
        print("-" * 80)

        try:
            output_path = amend_csv_indices(
                csv_path=csv_path,
                remapping_path=remapping_path,
                node_column=node_column,
                edge_column=edge_column,
            )
            output_paths.append(output_path)
            print(f"✓ Success: {Path(output_path).name}")

        except Exception as e:
            print(f"✗ Error: {e}")
            output_paths.append(None)

    # Final summary
    print("\n" + "=" * 80)
    print("BATCH PROCESSING COMPLETE")
    print("=" * 80)
    successful = sum(1 for p in output_paths if p is not None)
    print(f"Files found: {len(csv_paths)}")
    print(f"Successful: {successful}/{len(csv_paths)}")
    print(f"Failed: {len(csv_paths) - successful}/{len(csv_paths)}")

    if successful > 0:
        print(
            "\nAmended files saved with '_amended' suffix in their original locations"
        )

    print("=" * 80 + "\n")

    return output_paths


def preview_csv_amendment(
    csv_path: str,
    remapping_path: str,
    node_column: str = "node_idx",
    edge_column: Optional[str] = "edge_idx",
    num_rows: int = 10,
) -> None:
    """
    Preview how indices will be amended without saving.

    Args:
        csv_path: Path to input CSV file
        remapping_path: Path to JSON file containing remapping information
        node_column: Name of the node index column
        edge_column: Name of the edge index column
        num_rows: Number of rows to display in preview
    """

    print("\n" + "=" * 80)
    print("PREVIEW: CSV INDEX AMENDMENT")
    print("=" * 80)

    # Load remapping
    with open(remapping_path, "r") as f:
        remapping_info = json.load(f)

    node_remapping = {int(k): v for k, v in remapping_info["node_remapping"].items()}
    edge_remapping = {int(k): v for k, v in remapping_info["edge_remapping"].items()}
    removed_nodes = set(remapping_info["removed_nodes"])
    removed_edges = set(remapping_info["removed_edges"])

    # Load CSV
    df = pd.read_csv(csv_path)

    print(f"\nFile: {csv_path}")
    print(f"Total rows: {len(df)}")
    print(f"Columns: {list(df.columns)}")

    # Create preview DataFrame
    preview_df = df.head(num_rows).copy()

    has_nodes = node_column in preview_df.columns
    has_edges = edge_column is not None and edge_column in preview_df.columns

    # Show before/after for nodes
    if has_nodes:
        print(f"\n{node_column} PREVIEW (first {num_rows} rows):")
        print("-" * 60)

        preview_df[f"{node_column}_NEW"] = preview_df[node_column].apply(
            lambda x: node_remapping.get(x, f"UNMAPPED({x})")
        )
        preview_df[f"{node_column}_STATUS"] = preview_df[node_column].apply(
            lambda x: "REMOVED" if x in removed_nodes else "OK"
        )

        print(
            preview_df[
                [node_column, f"{node_column}_NEW", f"{node_column}_STATUS"]
            ].to_string(index=False)
        )

    # Show before/after for edges
    if has_edges:
        print(f"\n{edge_column} PREVIEW (first {num_rows} rows):")
        print("-" * 60)

        preview_df[f"{edge_column}_NEW"] = preview_df[edge_column].apply(
            lambda x: edge_remapping.get(x, f"UNMAPPED({x})")
        )
        preview_df[f"{edge_column}_STATUS"] = preview_df[edge_column].apply(
            lambda x: "REMOVED" if x in removed_edges else "OK"
        )

        print(
            preview_df[
                [edge_column, f"{edge_column}_NEW", f"{edge_column}_STATUS"]
            ].to_string(index=False)
        )

    # Statistics
    if has_nodes:
        nodes_in_csv = set(df[node_column].unique())
        will_be_removed = len(nodes_in_csv & removed_nodes)
        will_be_kept = len(nodes_in_csv - removed_nodes)

        print("\nNode Statistics:")
        print(f"  Unique nodes in CSV: {len(nodes_in_csv)}")
        print(f"  Will be removed: {will_be_removed}")
        print(f"  Will be kept: {will_be_kept}")

    if has_edges:
        edges_in_csv = set(df[edge_column].unique())
        will_be_removed = len(edges_in_csv & removed_edges)
        will_be_kept = len(edges_in_csv - removed_edges)

        print("\nEdge Statistics:")
        print(f"  Unique edges in CSV: {len(edges_in_csv)}")
        print(f"  Will be removed: {will_be_removed}")
        print(f"  Will be kept: {will_be_kept}")

    print("\n" + "=" * 80 + "\n")


def remove_duplicate_edge_pairs(
    csv_path: str,
    output_path: Optional[str] = None,
    edge_idx_column: str = "edge_idx",
    from_node_column: str = "from_node",
    to_node_column: str = "to_node",
) -> tuple[pd.DataFrame, list]:
    """
    Remove duplicate edges that have the same from_node and to_node pair.
    Keeps the first occurrence and removes subsequent duplicates.

    Args:
        csv_path: Path to input CSV file
        output_path: Optional path to save cleaned CSV. If None, adds '_cleaned' suffix
        edge_idx_column: Name of edge index column (default: "edge_idx")
        from_node_column: Name of source node column (default: "from_node")
        to_node_column: Name of target node column (default: "to_node")

    Returns:
        Tuple of (cleaned_dataframe, removed_edge_indices_list)
        - cleaned_dataframe: DataFrame with duplicate edges removed
        - removed_edge_indices_list: List of edge_idx values that were removed
    """

    print("\n" + "=" * 80)
    print("REMOVING DUPLICATE EDGE PAIRS")
    print("=" * 80)

    # Load CSV
    print(f"\nLoading CSV from: {csv_path}")
    df = pd.read_csv(csv_path)
    original_count = len(df)
    print(f"  ✓ Loaded {original_count} rows")
    print(f"  Columns: {list(df.columns)}")

    # Validate columns exist
    required_cols = [edge_idx_column, from_node_column, to_node_column]
    missing_cols = [col for col in required_cols if col not in df.columns]
    
    if missing_cols:
        raise ValueError(
            f"Missing required columns: {missing_cols}. "
            f"Available columns: {list(df.columns)}"
        )

    print(f"\nChecking for duplicate (from_node, to_node) pairs...")

    # Create a column representing the edge pair for easy duplicate detection
    df['_edge_pair'] = list(zip(df[from_node_column], df[to_node_column]))

    # Find duplicates
    duplicate_mask = df.duplicated(subset='_edge_pair', keep='first')
    duplicate_count = duplicate_mask.sum()

    if duplicate_count == 0:
        print("  ✓ No duplicate edge pairs found!")
        df = df.drop(columns=['_edge_pair'])
        return df

    print(f"\n⚠ Found {duplicate_count} duplicate edge(s)")
    print("\n" + "-" * 80)
    print("DUPLICATE EDGE PAIRS DETECTED")
    print("-" * 80)

    # Get all rows that are part of duplicate pairs (both kept and removed)
    duplicate_pairs = df[df['_edge_pair'].duplicated(keep=False)]
    
    # Group by edge pair to show all duplicates together
    grouped = duplicate_pairs.groupby('_edge_pair')
    
    for pair, group in grouped:
        from_node, to_node = pair
        edge_indices = group[edge_idx_column].tolist()
        
        print(f"\nDuplicate pair: from_node={from_node}, to_node={to_node}")
        print(f"  Found {len(edge_indices)} edges with this pair:")
        
        for idx, (row_idx, row) in enumerate(group.iterrows()):
            edge_idx = row[edge_idx_column]
            status = "KEEPING" if idx == 0 else "REMOVING"
            print(f"    edge_idx={edge_idx:3d} [{status}]")

    # Remove duplicate rows (keep first occurrence)
    print("\n" + "-" * 80)
    print("REMOVING DUPLICATES")
    print("-" * 80)
    
    rows_to_remove = df[duplicate_mask]
    removed_edge_indices = rows_to_remove[edge_idx_column].tolist()
    
    print(f"\nRemoving {len(removed_edge_indices)} duplicate edge(s):")
    print(f"  Edge indices being removed: {removed_edge_indices}")

    # Keep only non-duplicate rows
    df_cleaned = df[~duplicate_mask].copy()
    df_cleaned = df_cleaned.drop(columns=['_edge_pair'])

    # Summary
    final_count = len(df_cleaned)
    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)
    print(f"Original edges:     {original_count}")
    print(f"Duplicates removed: {duplicate_count}")
    print(f"Final edges:        {final_count}")
    print(f"Retention rate:     {(final_count / original_count) * 100:.2f}%")

    # Save if output path provided
    if output_path is None:
        from pathlib import Path
        csv_file = Path(csv_path)
        output_path = csv_file.parent / f"{csv_file.stem}_cleaned{csv_file.suffix}"
    
    output_path = str(output_path)
    
    print(f"\nSaving cleaned CSV to: {output_path}")
    df_cleaned.to_csv(output_path, index=False)
    print("  ✓ File saved successfully")
    
    print("=" * 80 + "\n")

    return df_cleaned, removed_edge_indices


def remove_zero_feature_edges(
    csv_path: str,
    output_path: Optional[str] = None,
    edge_idx_column: str = "edge_idx",
    exclude_columns: Optional[List[str]] = None,
) -> pd.DataFrame:
    """
    Remove edges where all feature columns (excluding edge_idx) are zero.

    Args:
        csv_path: Path to input CSV file
        output_path: Optional path to save cleaned CSV. If None, adds '_no_zeros' suffix
        edge_idx_column: Name of edge index column (default: "edge_idx")
        exclude_columns: Optional list of additional columns to exclude from zero check.
                        By default, only edge_idx_column is excluded.

    Returns:
        DataFrame with zero-feature edges removed
    """

    print("\n" + "=" * 80)
    print("REMOVING ZERO-FEATURE EDGES")
    print("=" * 80)

    # Load CSV
    print(f"\nLoading CSV from: {csv_path}")
    df = pd.read_csv(csv_path)
    original_count = len(df)
    print(f"  ✓ Loaded {original_count} rows")
    print(f"  Columns: {list(df.columns)}")

    # Validate edge_idx column exists
    if edge_idx_column not in df.columns:
        raise ValueError(
            f"Column '{edge_idx_column}' not found. "
            f"Available columns: {list(df.columns)}"
        )

    # Determine which columns to check for zeros
    if exclude_columns is None:
        exclude_columns = []

    exclude_columns = [edge_idx_column] + exclude_columns
    feature_columns = [col for col in df.columns if col not in exclude_columns]

    print(f"\nChecking {len(feature_columns)} feature columns for all-zero rows:")
    print(f"  Feature columns: {feature_columns}")
    print(f"  Excluded from check: {exclude_columns}")

    # Find rows where ALL feature columns are zero
    print("\nScanning for edges with all features = 0...")

    # Create mask for rows where all features are zero
    all_zeros_mask = (df[feature_columns] == 0).all(axis=1)
    zero_count = all_zeros_mask.sum()

    if zero_count == 0:
        print("  ✓ No edges with all-zero features found!")
        return df

    print(f"\n⚠ Found {zero_count} edge(s) with all features = 0")

    # Get the rows to be removed
    rows_to_remove = df[all_zeros_mask]

    print("\n" + "-" * 80)
    print("EDGES TO BE REMOVED")
    print("-" * 80)

    for idx, row in rows_to_remove.iterrows():
        edge_idx = row[edge_idx_column]
        print(f"\nRemoving edge_idx = {edge_idx}")
        print("  Features: ", end="")
        feature_values = [f"{col}={row[col]}" for col in feature_columns]
        print(", ".join(feature_values))

        # Verify all are indeed zero
        all_zero = all(row[col] == 0 for col in feature_columns)
        print(f"  All features zero: {all_zero}")

    # Remove zero-feature rows
    print("\n" + "-" * 80)
    print("REMOVING EDGES")
    print("-" * 80)

    removed_edge_indices = rows_to_remove[edge_idx_column].tolist()
    print(f"\nEdge indices being removed: {removed_edge_indices}")

    # Keep only non-zero rows
    df_cleaned = df[~all_zeros_mask].copy()

    # Summary
    final_count = len(df_cleaned)
    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)
    print(f"Original edges:     {original_count}")
    print(f"Zero-feature edges: {zero_count}")
    print(f"Final edges:        {final_count}")
    print(f"Retention rate:     {(final_count / original_count) * 100:.2f}%")

    # Save if output path provided
    if output_path is None:
        from pathlib import Path

        csv_file = Path(csv_path)
        output_path = csv_file.parent / f"{csv_file.stem}_no_zeros{csv_file.suffix}"

    output_path = str(output_path)

    print(f"\nSaving cleaned CSV to: {output_path}")
    df_cleaned.to_csv(output_path, index=False)
    print("  ✓ File saved successfully")

    print("=" * 80 + "\n")

    return df_cleaned

def remove_amended_files(
    root_folder: str,
    dry_run: bool = True,
    case_sensitive: bool = False,
) -> Tuple[List[str], List[str]]:
    """
    Remove all files containing 'amended' in their filename from a folder and all subfolders.

    Args:
        root_folder: Path to the root folder to search
        dry_run: If True, only shows what would be deleted without actually deleting (default: True)
        case_sensitive: If True, only matches exact case 'amended'. If False, matches any case (default: False)

    Returns:
        Tuple of (successfully_removed, failed_to_remove) file paths
    """

    print("\n" + "=" * 80)
    print("REMOVING FILES WITH 'AMENDED' IN FILENAME")
    print("=" * 80)

    if not os.path.exists(root_folder):
        raise FileNotFoundError(f"Folder not found: {root_folder}")

    print(f"\nRoot folder: {root_folder}")
    print(f"Dry run: {dry_run}")
    print(f"Case sensitive: {case_sensitive}")

    # Find all files with 'amended' in filename
    print("\n" + "-" * 80)
    print("SCANNING FOR FILES")
    print("-" * 80)

    amended_files = []
    
    # Walk through all directories and subdirectories
    for dirpath, dirnames, filenames in os.walk(root_folder):
        for filename in filenames:
            # Check if 'amended' is in the filename
            if case_sensitive:
                match = 'amended' in filename
            else:
                match = 'amended' in filename.lower()
            
            if match:
                full_path = os.path.join(dirpath, filename)
                amended_files.append(full_path)

    if not amended_files:
        print("\n✓ No files with 'amended' in filename found!")
        print("=" * 80 + "\n")
        return [], []

    print(f"\nFound {len(amended_files)} file(s) with 'amended' in filename:")
    print()

    # Display files grouped by directory
    files_by_dir = {}
    for filepath in amended_files:
        dirpath = os.path.dirname(filepath)
        filename = os.path.basename(filepath)
        
        if dirpath not in files_by_dir:
            files_by_dir[dirpath] = []
        files_by_dir[dirpath].append(filename)

    for dirpath in sorted(files_by_dir.keys()):
        rel_dir = os.path.relpath(dirpath, root_folder)
        if rel_dir == '.':
            rel_dir = '(root)'
        
        print(f"  Directory: {rel_dir}")
        for filename in sorted(files_by_dir[dirpath]):
            file_size = os.path.getsize(os.path.join(dirpath, filename))
            size_kb = file_size / 1024
            print(f"    - {filename} ({size_kb:.2f} KB)")
        print()

    # Remove files (or simulate if dry_run)
    print("-" * 80)
    if dry_run:
        print("DRY RUN - NO FILES WILL BE DELETED")
        print("Set dry_run=False to actually delete files")
    else:
        print("DELETING FILES")
    print("-" * 80)
    print()

    successfully_removed = []
    failed_to_remove = []

    for filepath in amended_files:
        rel_path = os.path.relpath(filepath, root_folder)
        
        if dry_run:
            print(f"  [DRY RUN] Would delete: {rel_path}")
            successfully_removed.append(filepath)
        else:
            try:
                os.remove(filepath)
                print(f"  ✓ Deleted: {rel_path}")
                successfully_removed.append(filepath)
            except Exception as e:
                print(f"  ✗ Failed to delete: {rel_path}")
                print(f"    Error: {e}")
                failed_to_remove.append(filepath)

    # Summary
    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)
    print(f"Files found:             {len(amended_files)}")
    
    if dry_run:
        print(f"Files that would be deleted: {len(successfully_removed)}")
    else:
        print(f"Files successfully deleted:  {len(successfully_removed)}")
        print(f"Files failed to delete:      {len(failed_to_remove)}")

    if failed_to_remove:
        print("\nFailed files:")
        for filepath in failed_to_remove:
            print(f"  - {os.path.relpath(filepath, root_folder)}")

    print("=" * 80 + "\n")

    return successfully_removed, failed_to_remove

if __name__ == "__main__":
    model_name = "Model4"

    model1_boundary_nodes = [
        3741,
        3742,
        3745,
        3748,
        3749,
        3751,
        3756,
        3757,
        3761,
        3762,
        3763,
        3764,
    ]
    model2_boundary_nodes = []
    model4_boundary_nodes = [
        8940,
        8964,
        8965,
        8967,
        8968,
        8969,
        8970,
        8972,
        8973,
        8974,
        8975,
        8977,
        8982,
        8986,
        8991,
        8998,
        9003,
        9005,
        9006,
        9007,
        9008,
        9009,
        9010,
        9011,
        9012,
        9014,
        9018,
        9021,
        9022,
        9025,
        9026,
        9029,
        9030,
        9034,
        9038,
        9040,
        9041,
        9044,
        9048,
        9050,
        9051,
        9052,
        9053,
        9055,
        9057,
        9059,
        9060,
        9061,
        9063,
        9065,
        9067,
        9068,
        9069,
        9071,
        9074,
        9078,
        9087,
        9090,
        9093,
        9094,
        9095,
        9097,
        9100,
        9101,
        9107,
        9110,
        9118,
        9124,
        9128,
        9131,
        9137,
        9140,
        9144,
        9150,
        9154,
        9158,
        9167,
        9171,
        9172,
        9173,
        9174,
        9175,
        9176,
        9177,
        9178,
        9179,
        9180,
        9181,
        9182,
        9183,
        9186,
        9187,
        9188,
        9189,
        9190,
        9191,
        9192,
        9193,
        9194,
        9195,
        9198,
        9199,
        9212,
        9219,
    ]

    if model_name == "Model1":
        selected_boundary_nodes = model1_boundary_nodes
    elif model_name == "Model2":
        selected_boundary_nodes = model2_boundary_nodes
    elif model_name == "Model4":
        selected_boundary_nodes = model4_boundary_nodes

    # visualize_boundary_condition_masks(
    #     nodes_2d_shp_file=f"/Users/jiayulim/Documents/GitHub/dual_flood_gnn/data/{model_name}/raw/Geometry/Nodes_2D.shp",
    #     edges_2d_shp_file=f"/Users/jiayulim/Documents/GitHub/dual_flood_gnn/data/{model_name}/raw/Geometry/Links_2D.shp",
    #     boundary_condition_npz_file=f"/Users/jiayulim/Documents/GitHub/dual_flood_gnn/data/{model_name}/processed/boundary_condition_masks.npz",
    #     constant_values_file=f"/Users/jiayulim/Documents/GitHub/dual_flood_gnn/data/{model_name}/processed/constant_values.npz",
    # )

    # # diagnose_boundary_condition_npz(
    # #     boundary_condition_npz_file=f"/Users/jiayulim/Documents/GitHub/dual_flood_gnn/data/{model_name}/processed/boundary_condition_masks.npz",
    # #     constant_values_npz_file=f"/Users/jiayulim/Documents/GitHub/dual_flood_gnn/data/{model_name}/processed/constant_values.npz",
    # # )

    # Make sure all files are original before editing
    # remove_amended_files(
    #     root_folder=f"/Users/jiayulim/Documents/GitHub/dual_flood_gnn/data/{model_name}/processed/features_csv/test",
    #     dry_run=False,
    # )

    # Edit shape files
    filter_nodes_by_fid_match(
        source_shapefile=f"/Users/jiayulim/Documents/GitHub/dual_flood_gnn/data/{model_name}/raw/Geometry/Nodes_2D.shp",
        reference_shapefile=f"/Users/jiayulim/Documents/GitHub/dual_flood_gnn/{model_name.lower()}_removed_ghost/Nodes_2D.shp",
        output_shapefile=f"/Users/jiayulim/Documents/GitHub/dual_flood_gnn/{model_name.lower()}_removed_ghost/Nodes_2D_processed.shp",
        source_fid_column="FID",
        reference_nodeidx_column="FID",  # either node_idx or FID depending on how Nodes_2D_processed.shp was created
        boundary_nodes=selected_boundary_nodes,
        remapping_json=f"/Users/jiayulim/Documents/GitHub/dual_flood_gnn/data/{model_name}/processed/node_edge_remapping/train.json",
        remapping_key="node_remapping",
        add_original_fid=False,
    )

    filter_edges_by_node_existence(
        source_edges_shapefile=f"/Users/jiayulim/Documents/GitHub/dual_flood_gnn/data/{model_name.lower()}/raw/Geometry/Links_2D.shp",
        reference_nodes_shapefile=f"/Users/jiayulim/Documents/GitHub/dual_flood_gnn/{model_name.lower()}_removed_ghost/Nodes_2D_processed.shp",
        output_shapefile=f"/Users/jiayulim/Documents/GitHub/dual_flood_gnn/{model_name.lower()}_removed_ghost/Links_2D_processed.shp",
        edge_from_node_column="from_node",
        edge_to_node_column="to_node",
        node_id_column="FID",
        remapping_json=f"/Users/jiayulim/Documents/GitHub/dual_flood_gnn/data/{model_name}/processed/node_edge_remapping/train.json",
        add_original_ids=False,
    )