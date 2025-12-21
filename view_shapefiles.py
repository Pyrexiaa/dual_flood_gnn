import numpy as np
import geopandas as gpd
from shapely.geometry import Point, LineString
import os


def visualize_boundary_condition_masks(
    nodes_2d_shp_file: str,
    edges_2d_shp_file: str,
    boundary_condition_npz_file: str = "boundary_condition_masks.npz",
    constant_values_file: str = "constant_value.npz",
    output_dir: str = None,
):
    """
    Load boundary condition masks from NPZ file and create shapefiles for visualization.
    
    Creates 4 shapefiles:
    1. All valid nodes (regular + boundary, no ghosts)
    2. Boundary nodes only
    3. All valid edges (regular + boundary, no ghost edges)
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
    
    for file in constant_values.files:
        print(f"  - {file:30s} Shape: {str(constant_values[file].shape):20s} Dtype: {constant_values[file].dtype}")

    # Check if valid masks are available
    if "_valid_nodes_mask" in constant_values.files:
        valid_nodes_mask = constant_values["valid_nodes_mask"]
        valid_edges_mask = constant_values["valid_edges_mask"]
        print(f"\n✓ Found valid masks in constant values")
        print(f"  Valid nodes (regular + boundary): {np.sum(valid_nodes_mask)}")
        print(f"  Ghost nodes: {np.sum(~valid_nodes_mask)}")
        print(f"  Boundary nodes: {np.sum(boundary_nodes_mask)}")
        print(f"  Valid edges (regular + boundary): {np.sum(valid_edges_mask)}")
        print(f"  Ghost edges: {np.sum(~valid_edges_mask)}")
        print(f"  Boundary edges: {np.sum(boundary_edges_mask)}")
    else:
        print(f"\n⚠ No valid masks found - assuming all data is valid")
        valid_nodes_mask = np.ones(len(static_nodes), dtype=bool)
        valid_edges_mask = np.ones(edge_index.shape[1], dtype=bool)

    # Assuming these are the indices in STATIC_NODE_FEATURES
    position_x_idx = 0  # Usually position_x is first
    position_y_idx = 1  # Usually position_y is second

    # Load original shapefiles to get CRS
    original_nodes_gdf = gpd.read_file(nodes_2d_shp_file)
    crs = original_nodes_gdf.crs

    # Set output directory
    if output_dir is None:
        output_dir = os.path.join("boundary_viz2")
    os.makedirs(output_dir, exist_ok=True)

    print(f"\n{'=' * 80}")
    print("CREATING SHAPEFILES")
    print("=" * 80)

    # --- 1. All Valid Nodes (Regular + Boundary, No Ghosts) ---
    valid_node_indices = np.where(valid_nodes_mask)[0]
    
    nodes_data = {
        "node_idx": valid_node_indices,
        "position_x": static_nodes[valid_nodes_mask, position_x_idx],
        "position_y": static_nodes[valid_nodes_mask, position_y_idx],
        "is_boundary": boundary_nodes_mask[valid_nodes_mask],
    }

    # Create Point geometries
    geometries = [
        Point(x, y) for x, y in zip(nodes_data["position_x"], nodes_data["position_y"])
    ]

    nodes_gdf = gpd.GeoDataFrame(nodes_data, geometry=geometries, crs=crs)
    nodes_output = os.path.join(output_dir, "nodes_valid_all.shp")
    nodes_gdf.to_file(nodes_output)
    print(f"\n✓ Saved: {nodes_output}")
    print(f"  Total nodes: {len(nodes_gdf)}")
    print(f"  - Regular nodes: {np.sum(~nodes_data['is_boundary'])}")
    print(f"  - Boundary nodes: {np.sum(nodes_data['is_boundary'])}")

    # --- 2. Boundary Nodes Only ---
    boundary_only_mask = valid_nodes_mask & boundary_nodes_mask
    boundary_node_indices = np.where(boundary_only_mask)[0]
    
    boundary_nodes_data = {
        "node_idx": boundary_node_indices,
        "position_x": static_nodes[boundary_only_mask, position_x_idx],
        "position_y": static_nodes[boundary_only_mask, position_y_idx],
    }

    geometries = [
        Point(x, y) for x, y in zip(boundary_nodes_data["position_x"], boundary_nodes_data["position_y"])
    ]

    boundary_nodes_gdf = gpd.GeoDataFrame(boundary_nodes_data, geometry=geometries, crs=crs)
    boundary_nodes_output = os.path.join(output_dir, "nodes_boundary_only.shp")
    boundary_nodes_gdf.to_file(boundary_nodes_output)
    print(f"\n✓ Saved: {boundary_nodes_output}")
    print(f"  Boundary nodes: {len(boundary_nodes_gdf)}")

    # --- 3. All Valid Edges (Regular + Boundary, No Ghost Edges) ---
    valid_edge_indices = np.where(valid_edges_mask)[0]
    
    from_nodes = edge_index[0, valid_edges_mask]
    to_nodes = edge_index[1, valid_edges_mask]

    edges_data = {
        "edge_idx": valid_edge_indices,
        "from_node": from_nodes,
        "to_node": to_nodes,
        "is_boundary": boundary_edges_mask[valid_edges_mask],
        "is_inflow": inflow_edges_mask[valid_edges_mask],
        "is_outflow": outflow_edges_mask[valid_edges_mask],
    }

    # Create LineString geometries
    geometries = []
    for from_idx, to_idx in zip(from_nodes, to_nodes):
        from_x = static_nodes[from_idx, position_x_idx]
        from_y = static_nodes[from_idx, position_y_idx]
        to_x = static_nodes[to_idx, position_x_idx]
        to_y = static_nodes[to_idx, position_y_idx]
        geometries.append(LineString([(from_x, from_y), (to_x, to_y)]))

    edges_gdf = gpd.GeoDataFrame(edges_data, geometry=geometries, crs=crs)
    edges_output = os.path.join(output_dir, "edges_valid_all.shp")
    edges_gdf.to_file(edges_output)
    print(f"\n✓ Saved: {edges_output}")
    print(f"  Total edges: {len(edges_gdf)}")
    print(f"  - Regular edges: {np.sum(~edges_data['is_boundary'])}")
    print(f"  - Boundary edges: {np.sum(edges_data['is_boundary'])}")
    print(f"    - Inflow: {np.sum(edges_data['is_inflow'])}")
    print(f"    - Outflow: {np.sum(edges_data['is_outflow'])}")

    # --- 4. Boundary Edges Only ---
    boundary_edges_only_mask = valid_edges_mask & boundary_edges_mask
    boundary_edge_indices = np.where(boundary_edges_only_mask)[0]
    
    from_nodes_boundary = edge_index[0, boundary_edges_only_mask]
    to_nodes_boundary = edge_index[1, boundary_edges_only_mask]

    boundary_edges_data = {
        "edge_idx": boundary_edge_indices,
        "from_node": from_nodes_boundary,
        "to_node": to_nodes_boundary,
        "is_inflow": inflow_edges_mask[boundary_edges_only_mask],
        "is_outflow": outflow_edges_mask[boundary_edges_only_mask],
    }

    # Create LineString geometries
    geometries = []
    for from_idx, to_idx in zip(from_nodes_boundary, to_nodes_boundary):
        from_x = static_nodes[from_idx, position_x_idx]
        from_y = static_nodes[from_idx, position_y_idx]
        to_x = static_nodes[to_idx, position_x_idx]
        to_y = static_nodes[to_idx, position_y_idx]
        geometries.append(LineString([(from_x, from_y), (to_x, to_y)]))

    boundary_edges_gdf = gpd.GeoDataFrame(boundary_edges_data, geometry=geometries, crs=crs)
    boundary_edges_output = os.path.join(output_dir, "edges_boundary_only.shp")
    boundary_edges_gdf.to_file(boundary_edges_output)
    print(f"\n✓ Saved: {boundary_edges_output}")
    print(f"  Boundary edges: {len(boundary_edges_gdf)}")
    print(f"    - Inflow: {np.sum(boundary_edges_data['is_inflow'])}")
    print(f"    - Outflow: {np.sum(boundary_edges_data['is_outflow'])}")

    print(f"\n{'=' * 80}")
    print("SUMMARY")
    print("=" * 80)
    print(f"\nOutput directory: {output_dir}")
    print(f"\nShapefiles created:")
    print(f"  1. nodes_valid_all.shp - {len(nodes_gdf)} nodes (regular + boundary)")
    print(f"  2. nodes_boundary_only.shp - {len(boundary_nodes_gdf)} boundary nodes")
    print(f"  3. edges_valid_all.shp - {len(edges_gdf)} edges (regular + boundary)")
    print(f"  4. edges_boundary_only.shp - {len(boundary_edges_gdf)} boundary edges")

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

def diagnose_boundary_condition_npz(
    boundary_condition_npz_file: str = "boundary_condition_masks.npz",
    constant_values_npz_file: str = "constant_values.npz",
):
    """
    Detailed diagnostic of boundary condition NPZ file to identify issues.
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

    # Extract masks
    boundary_nodes_mask = bc_data["boundary_nodes_mask"]
    boundary_edges_mask = bc_data["boundary_edges_mask"]
    inflow_edges_mask = bc_data["inflow_edges_mask"]
    outflow_edges_mask = bc_data["outflow_edges_mask"]

    print("=" * 80)
    print("1. MASK ARRAYS INFO")
    print("=" * 80)

    print("\nboundary_nodes_mask:")
    print(f"  Shape: {boundary_nodes_mask.shape}")
    print(f"  Dtype: {boundary_nodes_mask.dtype}")
    print(f"  True count: {np.sum(boundary_nodes_mask)}")
    print(f"  Min value: {boundary_nodes_mask.min()}")
    print(f"  Max value: {boundary_nodes_mask.max()}")

    print("\nboundary_edges_mask:")
    print(f"  Shape: {boundary_edges_mask.shape}")
    print(f"  Dtype: {boundary_edges_mask.dtype}")
    print(f"  True count: {np.sum(boundary_edges_mask)}")

    print("\ninflow_edges_mask:")
    print(f"  Shape: {inflow_edges_mask.shape}")
    print(f"  Dtype: {inflow_edges_mask.dtype}")
    print(f"  True count: {np.sum(inflow_edges_mask)}")

    print("\noutflow_edges_mask:")
    print(f"  Shape: {outflow_edges_mask.shape}")
    print(f"  Dtype: {outflow_edges_mask.dtype}")
    print(f"  True count: {np.sum(outflow_edges_mask)}")

    # Get boundary node indices
    boundary_node_indices = np.where(boundary_nodes_mask)[0]

    print("\n" + "=" * 80)
    print("2. BOUNDARY NODE INDICES")
    print("=" * 80)

    print("\nIndices where boundary_nodes_mask is True:")
    print(f"  Count: {len(boundary_node_indices)}")
    print(f"  Indices: {boundary_node_indices}")

    if len(boundary_node_indices) == 0:
        print("\n❌ ERROR: No boundary nodes found in mask!")
        return

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

    print("\n" + "=" * 80)
    print("4. BOUNDARY NODE DETAILS (ONE BY ONE)")
    print("=" * 80)

    for i, node_idx in enumerate(boundary_node_indices):
        print(f"\n--- Boundary Node {i + 1}/{len(boundary_node_indices)} ---")
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
            else:
                print(f"    {feat_name:20s}: N/A (feature index out of range)")

        # Check if position is zero
        if node_idx < len(static_nodes):
            pos_x = node_features[0]
            pos_y = node_features[1]

            if pos_x == 0.0 and pos_y == 0.0:
                print("  ⚠️  WARNING: Position is (0, 0) - likely incorrect!")

            if np.isnan(pos_x) or np.isnan(pos_y):
                print("  ❌ ERROR: Position contains NaN values!")

    print("\n" + "=" * 80)
    print("5. SAMPLE OF REGULAR NODES (for comparison)")
    print("=" * 80)

    regular_node_indices = np.where(~boundary_nodes_mask)[0][
        :5
    ]  # First 5 regular nodes

    for i, node_idx in enumerate(regular_node_indices):
        print(f"\n--- Regular Node {i + 1}/5 ---")
        print(f"  Node Index: {node_idx}")
        node_features = static_nodes[node_idx]
        print("  Features:")
        for feat_idx, feat_name in enumerate(feature_names):
            if feat_idx < len(node_features):
                print(f"    {feat_name:20s}: {node_features[feat_idx]:.6f}")

    print("\n" + "=" * 80)
    print("6. CHECKING FOR PATTERN")
    print("=" * 80)

    # Check if all boundary nodes have zero positions
    boundary_positions_x = static_nodes[boundary_node_indices, 0]
    boundary_positions_y = static_nodes[boundary_node_indices, 1]

    num_zero_x = np.sum(boundary_positions_x == 0)
    num_zero_y = np.sum(boundary_positions_y == 0)
    num_both_zero = np.sum((boundary_positions_x == 0) & (boundary_positions_y == 0))

    print("\nBoundary nodes with position issues:")
    print(f"  Position X = 0: {num_zero_x}/{len(boundary_node_indices)}")
    print(f"  Position Y = 0: {num_zero_y}/{len(boundary_node_indices)}")
    print(f"  Both X and Y = 0: {num_both_zero}/{len(boundary_node_indices)}")

    if num_both_zero > 0:
        print(
            f"\n❌ ISSUE IDENTIFIED: {num_both_zero} boundary nodes have (0, 0) position!"
        )
        print(
            "   This suggests boundary nodes were added with zero/placeholder values."
        )
        print("   Check the boundary_condition.apply() method.")

    print("\n" + "=" * 80)
    print("7. EXPECTED vs ACTUAL NODE INDICES")
    print("=" * 80)

    print("\nYou mentioned expected indices:")
    expected_indices = [
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
    print(f"  Expected: {expected_indices}")
    print("\nActual boundary node indices from mask:")
    print(f"  Actual: {list(boundary_node_indices)}")

    print("\nDifference:")
    print(f"  Number of expected: {len(expected_indices)}")
    print(f"  Number of actual: {len(boundary_node_indices)}")

    if set(expected_indices) == set(boundary_node_indices):
        print("  ✓ Indices match!")
    else:
        print("  ❌ Indices DO NOT match!")
        print(
            f"     In expected but not in actual: {set(expected_indices) - set(boundary_node_indices)}"
        )
        print(
            f"     In actual but not in expected: {set(boundary_node_indices) - set(expected_indices)}"
        )

    print("\n" + "=" * 80)
    print("8. CHECKING IF NODES WERE APPENDED")
    print("=" * 80)

    print(f"\nTotal nodes in static_nodes: {len(static_nodes)}")
    print(
        f"Maximum boundary node index: {boundary_node_indices.max() if len(boundary_node_indices) > 0 else 'N/A'}"
    )
    print(
        f"Minimum boundary node index: {boundary_node_indices.min() if len(boundary_node_indices) > 0 else 'N/A'}"
    )

    # Check if boundary nodes are at the end (appended)
    if len(boundary_node_indices) > 0:
        if boundary_node_indices.min() >= len(static_nodes) - len(
            boundary_node_indices
        ):
            print("\n✓ Boundary nodes appear to be appended at the end of the array")
        else:
            print("\n⚠️  Boundary nodes are scattered throughout the array")

    print("\n" + "=" * 80)
    print("DIAGNOSTIC COMPLETE")
    print("=" * 80)


if __name__ == "__main__":
    model_name = "Model1"

    # visualize_boundary_condition_masks(
    #     nodes_2d_shp_file=f"/Users/jiayulim/Documents/GitHub/dual_flood_gnn/data/{model_name}/raw/Geometry/Nodes_2D.shp",
    #     edges_2d_shp_file=f"/Users/jiayulim/Documents/GitHub/dual_flood_gnn/data/{model_name}/raw/Geometry/Links_2D.shp",
    #     boundary_condition_npz_file=f"/Users/jiayulim/Documents/GitHub/dual_flood_gnn/data/{model_name}/processed/boundary_condition_masks.npz",
    #     constant_values_file=f"/Users/jiayulim/Documents/GitHub/dual_flood_gnn/data/{model_name}/processed/constant_values.npz"
    # )

    diagnose_boundary_condition_npz(
        boundary_condition_npz_file=f"/Users/jiayulim/Documents/GitHub/dual_flood_gnn/data/{model_name}/processed/boundary_condition_masks.npz",
        constant_values_npz_file=f"/Users/jiayulim/Documents/GitHub/dual_flood_gnn/data/{model_name}/processed/constant_values.npz",
    )
