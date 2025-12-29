from typing import List, Optional, Union
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
    add_original_fid: bool = False,
) -> gpd.GeoDataFrame:
    """
    Filter a source shapefile to keep only nodes whose FID matches node_idx values
    in the reference shapefile. Optionally keeps specified boundary nodes regardless of FID match.
    Can also remap node IDs based on a JSON mapping file.

    This function:
    1. Loads both shapefiles
    2. Gets the set of valid node_idx values from the reference shapefile
    3. Filters the source shapefile to keep only rows where FID is in that set
    4. If keep_all_boundary=True and boundary_nodes list provided, also keeps those nodes
    5. Optionally applies node ID remapping from JSON file
    6. Preserves all columns and structure from the source shapefile
    7. Optionally saves the filtered result to a new shapefile

    Args:
        source_shapefile: Path to the shapefile to be filtered (contains FID)
        reference_shapefile: Path to the reference shapefile (contains node_idx)
        output_shapefile: Optional path to save filtered shapefile.
                         If None, returns GeoDataFrame without saving
        source_fid_column: Column name for FID in source shapefile (default: "FID")
        reference_nodeidx_column: Column name for node indices in reference (default: "node_idx")
        boundary_nodes: List or array of FID values for boundary nodes to preserve (default: None)
        keep_all_boundary: If True, keeps nodes in boundary_nodes list regardless of FID match (default: True)
        remapping_json: Optional path to JSON file containing node ID remapping
        remapping_key: Key in JSON file containing the remapping dict (default: "node_remapping")
        add_original_fid: If True and remapping applied, adds 'original_fid' column (default: True)

    Returns:
        GeoDataFrame: Filtered nodes matching the FID criteria and/or boundary nodes
    """

    # Validate input files exist
    if not os.path.exists(source_shapefile):
        raise FileNotFoundError(f"Source shapefile not found: {source_shapefile}")
    if not os.path.exists(reference_shapefile):
        raise FileNotFoundError(f"Reference shapefile not found: {reference_shapefile}")

    print("=" * 80)
    print("FILTERING NODES BY FID MATCH")
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

    # Get valid node indices from reference shapefile and convert to consistent type
    reference_values = reference_gdf[reference_nodeidx_column].values
    source_values = source_gdf[source_fid_column].values

    print("\nColumn types:")
    print(f"  Source {source_fid_column}: {source_values.dtype}")
    print(f"  Reference {reference_nodeidx_column}: {reference_values.dtype}")

    # Convert both to int64 for comparison (handles int, float, string representations)
    try:
        reference_values_int = reference_values.astype("int64")
        source_values_int = source_values.astype("int64")
        print("  ✓ Successfully converted both to int64 for comparison")
    except (ValueError, TypeError):
        # If int conversion fails, try string comparison
        print("  ⚠ Int conversion failed, using string comparison")
        reference_values_int = reference_values.astype(str)
        source_values_int = source_values.astype(str)

    valid_node_indices = set(reference_values_int)
    print(f"\nValid node indices from reference: {len(valid_node_indices)}")

    # Handle boundary nodes list
    has_boundary_list = boundary_nodes is not None and len(boundary_nodes) > 0
    if keep_all_boundary and has_boundary_list:
        # Convert boundary_nodes to the same type as source values
        boundary_nodes_array = np.array(boundary_nodes)
        try:
            boundary_nodes_int = boundary_nodes_array.astype(source_values_int.dtype)
        except (ValueError, TypeError):
            boundary_nodes_int = boundary_nodes_array.astype(str)

        boundary_nodes_set = set(boundary_nodes_int)
        print(
            f"\n✓ Boundary nodes list provided: {len(boundary_nodes_set)} unique FIDs"
        )
        print(f"  Sample boundary FIDs: {list(boundary_nodes_set)[:10]}")

        # Check how many boundary nodes exist in source
        boundary_in_source = (
            source_gdf[source_fid_column]
            .astype(source_values_int.dtype)
            .isin(boundary_nodes_set)
            .sum()
        )
        print(f"  Boundary nodes found in source: {boundary_in_source}")

    elif keep_all_boundary and not has_boundary_list:
        print("\n⚠ Warning: keep_all_boundary=True but no boundary_nodes list provided")
        print("  Will only filter by FID matching")

    # Filter source shapefile using converted values
    print("\nFiltering source shapefile...")
    print(
        f"  Keeping rows where {source_fid_column} is in reference {reference_nodeidx_column}"
    )

    # Create mask for FID matching
    source_fids_converted = source_gdf[source_fid_column].astype(
        source_values_int.dtype
    )
    fid_match_mask = source_fids_converted.isin(valid_node_indices)

    # Create mask for boundary nodes if applicable
    if keep_all_boundary and has_boundary_list:
        print("  AND keeping all nodes with FID in boundary_nodes list")
        boundary_mask = source_fids_converted.isin(boundary_nodes_set)

        # Combine masks: keep if FID matches OR is in boundary list
        final_mask = fid_match_mask | boundary_mask
    else:
        final_mask = fid_match_mask

    filtered_gdf = source_gdf[final_mask].copy()

    # Statistics
    original_count = len(source_gdf)
    filtered_count = len(filtered_gdf)
    removed_count = original_count - filtered_count

    print("\n" + "=" * 80)
    print("FILTERING RESULTS")
    print("=" * 80)
    print(f"\nOriginal nodes:  {original_count:,}")
    print(f"Filtered nodes:  {filtered_count:,}")
    print(f"Removed nodes:   {removed_count:,}")
    print(f"Retention rate:  {(filtered_count / original_count) * 100:.2f}%")

    # Detailed breakdown if boundary list provided
    if keep_all_boundary and has_boundary_list:
        fid_match_count = fid_match_mask.sum()
        boundary_match_count = boundary_mask.sum()
        both_match_count = (fid_match_mask & boundary_mask).sum()
        only_boundary_count = boundary_match_count - both_match_count

        print("\nBreakdown:")
        print(f"  Nodes with matching FID:           {fid_match_count:,}")
        print(f"  Nodes from boundary list:          {boundary_match_count:,}")
        print(f"  Nodes in both categories:          {both_match_count:,}")
        print(f"  Nodes ONLY from boundary list:     {only_boundary_count:,}")

    # Load and apply remapping if provided
    if remapping_json:
        print("\n" + "=" * 80)
        print("APPLYING NODE ID REMAPPING")
        print("=" * 80)

        if not os.path.exists(remapping_json):
            print(f"⚠ Warning: Remapping file not found: {remapping_json}")
            print("  Skipping remapping step")
        else:
            print(f"\nLoading remapping from: {remapping_json}")

            with open(remapping_json, "r") as f:
                remapping_data = json.load(f)

            if remapping_key not in remapping_data:
                print(f"⚠ Warning: Key '{remapping_key}' not found in JSON file")
                print(f"  Available keys: {list(remapping_data.keys())}")
                print("  Skipping remapping step")
            else:
                node_remapping = remapping_data[remapping_key]
                print(f"  Loaded {len(node_remapping)} remapping entries")
                print(f"  Sample mappings: {dict(list(node_remapping.items())[:5])}")

                # Save original FID if requested
                if add_original_fid:
                    filtered_gdf["original_fid"] = filtered_gdf[
                        source_fid_column
                    ].copy()

                # Apply remapping
                remapped_count = 0
                unmapped_count = 0
                unmapped_nodes = []

                # Create new FID column with remapped values
                new_fids = []
                for old_fid in filtered_gdf[source_fid_column]:
                    # Convert to string for JSON key lookup
                    old_fid_str = str(old_fid)

                    if old_fid_str in node_remapping:
                        new_fid = node_remapping[old_fid_str]
                        new_fids.append(new_fid)
                        remapped_count += 1
                    else:
                        # Keep original FID if no mapping found
                        new_fids.append(old_fid)
                        unmapped_count += 1
                        if len(unmapped_nodes) < 10:  # Store first 10 for reporting
                            unmapped_nodes.append(old_fid)

                # Update FID column
                filtered_gdf[source_fid_column] = new_fids

                print("\nRemapping results:")
                print(f"  Nodes remapped:     {remapped_count:,}")
                print(f"  Nodes not in map:   {unmapped_count:,}")
                if unmapped_nodes:
                    print(f"  Sample unmapped:    {unmapped_nodes}")

                if add_original_fid:
                    print("\n✓ Original FIDs preserved in 'original_fid' column")

    # Save if output path provided
    if output_shapefile:
        # Create output directory if needed
        output_dir = os.path.dirname(output_shapefile)
        if output_dir and not os.path.exists(output_dir):
            os.makedirs(output_dir, exist_ok=True)

        print(f"\nSaving filtered shapefile to: {output_shapefile}")
        filtered_gdf.to_file(output_shapefile)
        print("✓ File saved successfully")

        if remapping_json and os.path.exists(remapping_json):
            print("  FID column: Contains remapped node IDs")
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
    add_original_ids: bool = False,
) -> gpd.GeoDataFrame:
    """
    Filter edges to keep only those where BOTH from_node and to_node exist in the reference nodes shapefile.
    Applies node ID remapping BEFORE filtering, then edge ID remapping AFTER filtering.

    Processing order:
    1. Load shapefiles
    2. Remap from_node and to_node (if node_remapping provided)
    3. Filter edges based on remapped node IDs
    4. Remap edge FIDs (if edge_remapping provided)
    5. Save

    Args:
        source_edges_shapefile: Path to the edges shapefile to be filtered
        reference_nodes_shapefile: Path to the nodes shapefile containing valid nodes
        output_shapefile: Optional path to save filtered edges shapefile
        edge_from_node_column: Column name for source node in edges (default: "from_node")
        edge_to_node_column: Column name for target node in edges (default: "to_node")
        node_id_column: Column name for node identifier in nodes shapefile (default: "FID")
        edge_fid_column: Column name for edge FID in edges shapefile (default: "FID")
        remapping_json: Optional path to JSON file containing remapping
        edge_remapping_key: Key in JSON for edge remapping (default: "edge_remapping")
        node_remapping_key: Key in JSON for node remapping (default: "node_remapping")
        add_original_ids: If True, adds columns for original IDs (default: True)

    Returns:
        GeoDataFrame: Filtered edges where both endpoints exist in nodes shapefile
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
    print("FILTERING EDGES BY NODE EXISTENCE")
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

    # STEP 1: Apply node remapping to from_node and to_node BEFORE filtering
    if remapping_json and os.path.exists(remapping_json):
        print("\n" + "=" * 80)
        print("STEP 1: REMAPPING NODE IDs (BEFORE FILTERING)")
        print("=" * 80)
        
        print(f"\nLoading remapping from: {remapping_json}")
        
        with open(remapping_json, "r") as f:
            remapping_data = json.load(f)
        
        if node_remapping_key in remapping_data:
            node_remapping = remapping_data[node_remapping_key]
            print(f"\n✓ Node remapping found: {len(node_remapping)} entries")
            print(f"  Sample node mappings: {dict(list(node_remapping.items())[:5])}")
            
            # Save original node IDs if requested
            if add_original_ids:
                edges_gdf["orig_from"] = edges_gdf[edge_from_node_column].copy()
                edges_gdf["orig_to"] = edges_gdf[edge_to_node_column].copy()
            
            # Remap from_node
            from_remapped_count = 0
            from_unmapped_count = 0
            new_from_nodes = []
            
            for old_node in edges_gdf[edge_from_node_column]:
                old_node_str = str(old_node)
                
                if old_node_str in node_remapping:
                    new_node = node_remapping[old_node_str]
                    new_from_nodes.append(new_node)
                    from_remapped_count += 1
                else:
                    new_from_nodes.append(old_node)
                    from_unmapped_count += 1
            
            edges_gdf[edge_from_node_column] = new_from_nodes
            
            # Remap to_node
            to_remapped_count = 0
            to_unmapped_count = 0
            new_to_nodes = []
            
            for old_node in edges_gdf[edge_to_node_column]:
                old_node_str = str(old_node)
                
                if old_node_str in node_remapping:
                    new_node = node_remapping[old_node_str]
                    new_to_nodes.append(new_node)
                    to_remapped_count += 1
                else:
                    new_to_nodes.append(old_node)
                    to_unmapped_count += 1
            
            edges_gdf[edge_to_node_column] = new_to_nodes
            
            print("\nNode ID remapping results:")
            print(f"  from_node remapped: {from_remapped_count:,}")
            print(f"  from_node unmapped: {from_unmapped_count:,}")
            print(f"  to_node remapped:   {to_remapped_count:,}")
            print(f"  to_node unmapped:   {to_unmapped_count:,}")
            
            if add_original_ids:
                print("\n✓ Original node IDs preserved in columns:")
                print("  - orig_from: Original from_node IDs")
                print("  - orig_to: Original to_node IDs")
        else:
            print(f"\n⚠ Warning: Key '{node_remapping_key}' not found in JSON")
            print(f"  Available keys: {list(remapping_data.keys())}")
            print("  Proceeding with original node IDs")

    # STEP 2: Filter edges based on (remapped) node existence
    print("\n" + "=" * 80)
    print("STEP 2: FILTERING EDGES BY NODE EXISTENCE")
    print("=" * 80)
    
    # Get set of valid node IDs from reference
    valid_node_ids = set(nodes_gdf[node_id_column].values)
    print(f"\nValid node IDs in reference: {len(valid_node_ids)}")

    # Get edge connectivity (now with remapped node IDs if remapping was applied)
    from_nodes = edges_gdf[edge_from_node_column].values
    to_nodes = edges_gdf[edge_to_node_column].values

    print("\nColumn types:")
    print(f"  Edges {edge_from_node_column}: {from_nodes.dtype}")
    print(f"  Edges {edge_to_node_column}: {to_nodes.dtype}")
    print(f"  Nodes {node_id_column}: {nodes_gdf[node_id_column].dtype}")

    # Convert to consistent types for comparison
    try:
        valid_node_ids_int = set(np.array(list(valid_node_ids)).astype("int64"))
        from_nodes_int = from_nodes.astype("int64")
        to_nodes_int = to_nodes.astype("int64")
        print("  ✓ Successfully converted all to int64 for comparison")
    except (ValueError, TypeError):
        # If int conversion fails, use string comparison
        print("  ⚠ Int conversion failed, using string comparison")
        valid_node_ids_int = set(np.array(list(valid_node_ids)).astype(str))
        from_nodes_int = from_nodes.astype(str)
        to_nodes_int = to_nodes.astype(str)

    # Create masks for valid connections
    print("\nChecking edge connectivity...")
    from_node_exists = np.isin(from_nodes_int, list(valid_node_ids_int))
    to_node_exists = np.isin(to_nodes_int, list(valid_node_ids_int))

    # Keep edge only if BOTH from_node AND to_node exist
    both_nodes_exist = from_node_exists & to_node_exists

    # Statistics before filtering
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

    # Filter edges
    filtered_edges_gdf = edges_gdf[both_nodes_exist].copy()

    # Summary statistics
    original_count = len(edges_gdf)
    filtered_count = len(filtered_edges_gdf)
    removed_count = original_count - filtered_count

    print("\nFiltering results:")
    print(f"  Original edges:  {original_count:,}")
    print(f"  Filtered edges:  {filtered_count:,}")
    print(f"  Removed edges:   {removed_count:,}")
    print(f"  Retention rate:  {(filtered_count / original_count) * 100:.2f}%")

    # Show sample of removed edges (if any)
    if removed_count > 0:
        removed_edges = edges_gdf[~both_nodes_exist]
        print("\nSample of removed edges (first 5):")
        sample_size = min(5, len(removed_edges))
        for idx in range(sample_size):
            edge_idx = removed_edges.index[idx]
            from_node = removed_edges.iloc[idx][edge_from_node_column]
            to_node = removed_edges.iloc[idx][edge_to_node_column]
            from_exists = (
                from_node in valid_node_ids or int(from_node) in valid_node_ids_int
            )
            to_exists = to_node in valid_node_ids or int(to_node) in valid_node_ids_int
            print(
                f"  Edge {edge_idx}: from_node={from_node} (exists: {from_exists}), to_node={to_node} (exists: {to_exists})"
            )

    # STEP 3: Apply edge remapping AFTER filtering
    if remapping_json and os.path.exists(remapping_json):
        print("\n" + "=" * 80)
        print("STEP 3: REMAPPING EDGE IDs (AFTER FILTERING)")
        print("=" * 80)
        
        with open(remapping_json, "r") as f:
            remapping_data = json.load(f)
        
        if edge_remapping_key in remapping_data:
            edge_remapping = remapping_data[edge_remapping_key]
            print(f"\n✓ Edge remapping found: {len(edge_remapping)} entries")
            print(f"  Sample edge mappings: {dict(list(edge_remapping.items())[:5])}")
            
            # Save original edge FID if requested
            if add_original_ids:
                filtered_edges_gdf["original_fid"] = filtered_edges_gdf[
                    edge_fid_column
                ].copy()
            
            # Apply edge remapping
            remapped_count = 0
            unmapped_count = 0
            unmapped_edges = []
            
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
                    if len(unmapped_edges) < 10:
                        unmapped_edges.append(old_fid)
            
            filtered_edges_gdf[edge_fid_column] = new_edge_fids
            
            print("\nEdge FID remapping results:")
            print(f"  Edges remapped:     {remapped_count:,}")
            print(f"  Edges not in map:   {unmapped_count:,}")
            if unmapped_edges:
                print(f"  Sample unmapped:    {unmapped_edges}")
            
            if add_original_ids:
                print("\n✓ Original edge FIDs preserved in 'original_fid' column")
        else:
            print(f"\n⚠ Warning: Key '{edge_remapping_key}' not found in JSON")
            print(f"  Available keys: {list(remapping_data.keys())}")
            print("  Proceeding with original edge FIDs")

    # Save if output path provided
    if output_shapefile:
        # Create output directory if needed
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
        
        if add_original_ids and remapping_json:
            print("\nOriginal IDs preserved in:")
            if 'original_fid' in filtered_edges_gdf.columns:
                print("  - original_fid: Original edge FIDs")
            if 'orig_from' in filtered_edges_gdf.columns:
                print("  - orig_from: Original from_node IDs")
            if 'orig_to' in filtered_edges_gdf.columns:
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
    print("AMENDING CSV INDICES")
    print("=" * 80)

    # Load remapping info
    print(f"\nLoading remapping from: {remapping_path}")
    with open(remapping_path, "r") as f:
        remapping_info = json.load(f)

    # Convert string keys to integers
    node_remapping = {int(k): v for k, v in remapping_info["node_remapping"].items()}
    edge_remapping = {int(k): v for k, v in remapping_info["edge_remapping"].items()}
    removed_nodes = set(remapping_info["removed_nodes"])
    removed_edges = set(remapping_info["removed_edges"])

    print(f"  ✓ Loaded {len(node_remapping)} node mappings")
    print(f"  ✓ Loaded {len(edge_remapping)} edge mappings")
    print(f"  ✓ {len(removed_nodes)} removed nodes")
    print(f"  ✓ {len(removed_edges)} removed edges")

    # Load CSV
    print(f"\nLoading CSV from: {csv_path}")
    df = pd.read_csv(csv_path)
    print(f"  ✓ Loaded {len(df)} rows")
    print(f"  Columns: {list(df.columns)}")

    original_row_count = len(df)

    # Check if node column exists
    has_nodes = node_column in df.columns
    has_edges = edge_column is not None and edge_column in df.columns

    if not has_nodes and not has_edges:
        raise ValueError(
            f"CSV must contain at least one of '{node_column}' or '{edge_column}' columns. "
            f"Found columns: {list(df.columns)}"
        )

    # Amend node indices
    if has_nodes:
        print(f"\nAmending node indices (column: '{node_column}')...")

        # Check for removed nodes
        nodes_in_csv = set(df[node_column].unique())
        removed_in_csv = nodes_in_csv & removed_nodes

        if removed_in_csv:
            print(f"  ⚠ WARNING: Found {len(removed_in_csv)} removed nodes in CSV")
            print("    These rows will be FILTERED OUT")
            print(f"    Removed nodes: {sorted(list(removed_in_csv))[:10]}")
            if len(removed_in_csv) > 10:
                print(f"    ... and {len(removed_in_csv) - 10} more")

            # Filter out rows with removed nodes
            df = df[~df[node_column].isin(removed_nodes)]
            print(f"  ✓ Filtered: {original_row_count} -> {len(df)} rows")

        # Remap node indices
        unmapped_nodes = []

        def remap_node(old_idx):
            if old_idx in node_remapping:
                return node_remapping[old_idx]
            else:
                unmapped_nodes.append(old_idx)
                return old_idx  # Keep original if not in mapping

        df[node_column] = df[node_column].apply(remap_node)

        if unmapped_nodes:
            unique_unmapped = set(unmapped_nodes)
            print(
                f"  ⚠ WARNING: {len(unique_unmapped)} node indices not found in remapping"
            )
            print(
                f"    Unmapped nodes (kept original): {sorted(list(unique_unmapped))[:10]}"
            )
        else:
            print("  ✓ All node indices successfully remapped")

    # Amend edge indices
    if has_edges:
        print(f"\nAmending edge indices (column: '{edge_column}')...")

        # Check for removed edges
        edges_in_csv = set(df[edge_column].unique())
        removed_in_csv = edges_in_csv & removed_edges

        if removed_in_csv:
            print(f"  ⚠ WARNING: Found {len(removed_in_csv)} removed edges in CSV")
            print("    These rows will be FILTERED OUT")
            print(f"    Removed edges: {sorted(list(removed_in_csv))[:10]}")
            if len(removed_in_csv) > 10:
                print(f"    ... and {len(removed_in_csv) - 10} more")

            # Filter out rows with removed edges
            rows_before = len(df)
            df = df[~df[edge_column].isin(removed_edges)]
            print(f"  ✓ Filtered: {rows_before} -> {len(df)} rows")

        # Remap edge indices
        unmapped_edges = []

        def remap_edge(old_idx):
            if old_idx in edge_remapping:
                return edge_remapping[old_idx]
            else:
                unmapped_edges.append(old_idx)
                return old_idx  # Keep original if not in mapping

        df[edge_column] = df[edge_column].apply(remap_edge)

        if unmapped_edges:
            unique_unmapped = set(unmapped_edges)
            print(
                f"  ⚠ WARNING: {len(unique_unmapped)} edge indices not found in remapping"
            )
            print(
                f"    Unmapped edges (kept original): {sorted(list(unique_unmapped))[:10]}"
            )
        else:
            print("  ✓ All edge indices successfully remapped")

    # Determine output path
    if output_path is None:
        csv_file = Path(csv_path)
        output_path = csv_file.parent / f"{csv_file.stem}_amended{csv_file.suffix}"

    output_path = str(output_path)

    # Save amended CSV
    print(f"\nSaving amended CSV to: {output_path}")
    df.to_csv(output_path, index=False)
    print(f"  ✓ Saved {len(df)} rows")

    # Summary
    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)
    print(f"Input file:  {csv_path}")
    print(f"Output file: {output_path}")
    print(
        f"Rows: {original_row_count} -> {len(df)} (removed {original_row_count - len(df)})"
    )

    if has_nodes:
        print(f"✓ Node indices amended (column: '{node_column}')")
    if has_edges:
        print(f"✓ Edge indices amended (column: '{edge_column}')")

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


if __name__ == "__main__":
    model_name = "Model1"

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

    visualize_boundary_condition_masks(
        nodes_2d_shp_file=f"/Users/jiayulim/Documents/GitHub/dual_flood_gnn/data/{model_name}/raw/Geometry/Nodes_2D.shp",
        edges_2d_shp_file=f"/Users/jiayulim/Documents/GitHub/dual_flood_gnn/data/{model_name}/raw/Geometry/Links_2D.shp",
        boundary_condition_npz_file=f"/Users/jiayulim/Documents/GitHub/dual_flood_gnn/data/{model_name}/processed/boundary_condition_masks.npz",
        constant_values_file=f"/Users/jiayulim/Documents/GitHub/dual_flood_gnn/data/{model_name}/processed/constant_values.npz",
    )

    # diagnose_boundary_condition_npz(
    #     boundary_condition_npz_file=f"/Users/jiayulim/Documents/GitHub/dual_flood_gnn/data/{model_name}/processed/boundary_condition_masks.npz",
    #     constant_values_npz_file=f"/Users/jiayulim/Documents/GitHub/dual_flood_gnn/data/{model_name}/processed/constant_values.npz",
    # )

    filter_nodes_by_fid_match(
        source_shapefile=f"/Users/jiayulim/Documents/GitHub/dual_flood_gnn/data/{model_name}/raw/Geometry/Nodes_2D.shp",
        reference_shapefile=f"/Users/jiayulim/Documents/GitHub/dual_flood_gnn/{model_name.lower()}_removed_ghost/Nodes_2D.shp",
        output_shapefile=f"/Users/jiayulim/Documents/GitHub/dual_flood_gnn/{model_name.lower()}_removed_ghost/Nodes_2D_processed.shp",
        source_fid_column="FID",
        reference_nodeidx_column="FID",  # either node_idx or FID depending on how Nodes_2D_processed.shp was created
        boundary_nodes=selected_boundary_nodes,
        remapping_json=f"/Users/jiayulim/Documents/GitHub/dual_flood_gnn/data/{model_name}/processed/node_edge_remapping/train.json",
        remapping_key="node_remapping",
    )

    filter_edges_by_node_existence(
        source_edges_shapefile=f"/Users/jiayulim/Documents/GitHub/dual_flood_gnn/data/{model_name.lower()}/raw/Geometry/Links_2D.shp",
        reference_nodes_shapefile=f"/Users/jiayulim/Documents/GitHub/dual_flood_gnn/{model_name.lower()}_removed_ghost/Nodes_2D_processed.shp",
        output_shapefile=f"/Users/jiayulim/Documents/GitHub/dual_flood_gnn/{model_name.lower()}_removed_ghost/Links_2D_processed.shp",
        edge_from_node_column="from_node",
        edge_to_node_column="to_node",
        node_id_column="FID",
        remapping_json=f"/Users/jiayulim/Documents/GitHub/dual_flood_gnn/data/{model_name}/processed/node_edge_remapping/train.json",
    )

    batch_amend_csv_indices(
        main_folder=f"/Users/jiayulim/Documents/GitHub/dual_flood_gnn/data/{model_name}/processed/features_csv/train",
        filename_pattern="2d_nodes_dynamic_*.csv",
        remapping_path=f"/Users/jiayulim/Documents/GitHub/dual_flood_gnn/data/{model_name}/processed/node_edge_remapping/train.json",
        node_column="node_idx",
        edge_column="edge_idx",
        recursive=True,
    )

    # amend_csv_indices(
    #     csv_path=f"/Users/jiayulim/Documents/GitHub/dual_flood_gnn/data/{model_name}/processed/features_csv/train/1d2d_connections.csv",
    #     remapping_path=f"/Users/jiayulim/Documents/GitHub/dual_flood_gnn/data/{model_name}/processed/node_edge_remapping/train.json",
    #     output_path=f"/Users/jiayulim/Documents/GitHub/dual_flood_gnn/data/{model_name}/processed/features_csv/train/event_1/2d_nodes_dynamic_all_reindexed.csv",
    #     node_column="node_idx",
    #     edge_column="edge_idx",
    # )
