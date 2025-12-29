import os
from pathlib import Path
import numpy as np

from numpy import ndarray
from typing import List, Tuple

from .hecras_1d2d_data_retrieval import get_min_cell_elevation
import json


class Boundary1d2dCondition:
    def __init__(
        self,
        root_dir: str,
        hec_ras_file: str,
        inflow_boundary_nodes: List[int],
        outflow_boundary_nodes: List[int],
        saved_npz_file: str,
        perimeter_name: str = "US Beaver",
    ):
        self.hec_ras_path = os.path.join(root_dir, "raw", hec_ras_file)
        self.saved_npz_path = os.path.join(root_dir, "processed", saved_npz_file)
        self.init_inflow_boundary_nodes = inflow_boundary_nodes
        self.init_outflow_boundary_nodes = outflow_boundary_nodes
        self.perimeter_name = perimeter_name
        self._init()
        self._init_masks()

        self._is_called = {"create": False, "remove": False, "apply": False}
        self._boundary_edge_index = None
        self._boundary_dynamic_edges = None

        print("Init inflow boundary nodes: ", self.init_inflow_boundary_nodes)
        print("Init outflow boundary nodes: ", self.init_outflow_boundary_nodes)

    def _init(self) -> None:
        min_elevation = get_min_cell_elevation(
            self.hec_ras_path, perimeter_name=self.perimeter_name
        )
        ghost_nodes = np.where(np.isnan(min_elevation))[0]

        # Handle empty boundary node lists
        inflow_array = (
            np.array(self.init_inflow_boundary_nodes)
            if self.init_inflow_boundary_nodes
            else np.array([], dtype=int)
        )
        outflow_array = (
            np.array(self.init_outflow_boundary_nodes)
            if self.init_outflow_boundary_nodes
            else np.array([], dtype=int)
        )

        boundary_nodes = np.concatenate([inflow_array, outflow_array])

        # Only validate if there are boundary nodes
        if len(boundary_nodes) > 0:
            for bn in boundary_nodes:
                assert bn in ghost_nodes, f"Boundary node {bn} is not a ghost node."

        # Reassign new indices to the boundary nodes taking into account the removal of ghost nodes
        # Ghost nodes are assumed to be the last nodes in the node feature matrix
        (num_nodes,) = min_elevation.shape
        num_non_ghost_nodes = num_nodes - len(ghost_nodes)

        if len(boundary_nodes) > 0:
            new_boundary_nodes = np.arange(
                num_non_ghost_nodes, (num_non_ghost_nodes + len(boundary_nodes))
            )
            boundary_nodes_mapping = dict(zip(boundary_nodes, new_boundary_nodes))
            new_inflow_boundary_nodes = (
                np.array(
                    [
                        boundary_nodes_mapping[bn]
                        for bn in self.init_inflow_boundary_nodes
                    ]
                )
                if self.init_inflow_boundary_nodes
                else np.array([], dtype=int)
            )
            new_outflow_boundary_nodes = (
                np.array(
                    [
                        boundary_nodes_mapping[bn]
                        for bn in self.init_outflow_boundary_nodes
                    ]
                )
                if self.init_outflow_boundary_nodes
                else np.array([], dtype=int)
            )
        else:
            boundary_nodes_mapping = {}
            new_inflow_boundary_nodes = np.array([], dtype=int)
            new_outflow_boundary_nodes = np.array([], dtype=int)

        self.ghost_nodes = ghost_nodes
        self.boundary_nodes_mapping = boundary_nodes_mapping
        self.new_inflow_boundary_nodes = new_inflow_boundary_nodes
        self.new_outflow_boundary_nodes = new_outflow_boundary_nodes

    def _init_masks(self) -> None:
        if os.path.exists(self.saved_npz_path):
            masks = np.load(self.saved_npz_path)
            self.boundary_nodes_mask = masks["boundary_nodes_mask"]
            self.boundary_edges_mask = masks["boundary_edges_mask"]
            self.inflow_edges_mask = masks["inflow_edges_mask"]
            self.outflow_edges_mask = masks["outflow_edges_mask"]
            return

        # If masks_npz_path does not exist, assume that this is the first time the boundary condition is being created and process() is being called on dataset
        self.boundary_nodes_mask = None  # Used for autoregressive prediction
        self.boundary_edges_mask = None  # Used for autoregressive prediction
        self.inflow_edges_mask = None  # Used for global physics mass conservation loss
        self.outflow_edges_mask = None  # Used for global physics mass conservation loss

    def save_data(self) -> None:
        np.savez(
            self.saved_npz_path,
            boundary_nodes_mask=self.boundary_nodes_mask,
            boundary_edges_mask=self.boundary_edges_mask,
            inflow_edges_mask=self.inflow_edges_mask,
            outflow_edges_mask=self.outflow_edges_mask,
        )

    def create(self, edge_index: ndarray, dynamic_edges: ndarray) -> None:
        print("\n" + "=" * 80)
        print("BOUNDARY CONDITION: CREATE")
        print("=" * 80)

        # Handle empty boundary node lists
        inflow_array = (
            np.array(self.init_inflow_boundary_nodes)
            if self.init_inflow_boundary_nodes
            else np.array([], dtype=int)
        )
        outflow_array = (
            np.array(self.init_outflow_boundary_nodes)
            if self.init_outflow_boundary_nodes
            else np.array([], dtype=int)
        )

        print("\nInput:")
        print(f"  Initial inflow boundary nodes: {self.init_inflow_boundary_nodes}")
        print(f"  Initial outflow boundary nodes: {self.init_outflow_boundary_nodes}")
        print(f"  Ghost nodes: {self.ghost_nodes}")
        print(f"  Edge index shape: {edge_index.shape}")
        print(f"  Dynamic edges shape: {dynamic_edges.shape}")

        boundary_nodes = np.concatenate([inflow_array, outflow_array])
        print(f"\nBoundary nodes (original indices): {boundary_nodes}")

        # If no boundary nodes, initialize empty arrays
        if len(boundary_nodes) == 0:
            print("\n⚠️  No boundary nodes - initializing empty arrays")
            self._boundary_edge_index = np.empty((2, 0), dtype=edge_index.dtype)
            self._boundary_dynamic_edges = np.empty(
                (dynamic_edges.shape[0], 0, dynamic_edges.shape[2]),
                dtype=dynamic_edges.dtype,
            )
            self._is_called["create"] = True
            return

        print("\n✓ Boundary nodes will KEEP their original indices")
        print(
            f"  Inflow boundary nodes: {self.init_inflow_boundary_nodes if self.init_inflow_boundary_nodes else []}"
        )
        print(
            f"  Outflow boundary nodes: {self.init_outflow_boundary_nodes if self.init_outflow_boundary_nodes else []}"
        )

        # Find edges connected to boundary nodes
        boundary_edges_mask = np.any(np.isin(edge_index, boundary_nodes), axis=0)
        boundary_edge_index = edge_index[:, boundary_edges_mask]
        boundary_edges = boundary_edges_mask.nonzero()[0]

        print("\nBoundary edges:")
        print(f"  Number of boundary edges: {len(boundary_edges)}")
        print(f"  Boundary edge indices: {boundary_edges}")
        print(f"  Boundary edge_index shape: {boundary_edge_index.shape}")
        print("  First 5 boundary edges:")
        for i in range(min(5, boundary_edge_index.shape[1])):
            print(
                f"    Edge {i}: {boundary_edge_index[0, i]} -> {boundary_edge_index[1, i]}"
            )

        # NO REMAPPING - use original indices
        boundary_dynamic_edges = dynamic_edges[:, boundary_edges, :].copy()

        # Ensure inflow boundary edges point away from the boundary node
        if self.init_inflow_boundary_nodes:
            inflow_boundary_array = np.array(self.init_inflow_boundary_nodes)
            inflow_to_boundary_mask = np.isin(
                boundary_edge_index[1], inflow_boundary_array
            )
            print("\nInflow edge processing:")
            print(
                f"  Edges pointing TO inflow boundary nodes: {np.sum(inflow_to_boundary_mask)}"
            )

            if np.any(inflow_to_boundary_mask):
                print("  Flipping these edges to point AWAY from boundary")
                inflow_to_boundary = boundary_edge_index[:, inflow_to_boundary_mask]
                inflow_to_boundary[[0, 1], :] = inflow_to_boundary[[1, 0], :]
                boundary_edge_index[:, inflow_to_boundary_mask] = inflow_to_boundary
                boundary_dynamic_edges[:, inflow_to_boundary_mask, :] *= -1

        # Ensure outflow boundary edges point towards the boundary node
        if self.init_outflow_boundary_nodes:
            outflow_boundary_array = np.array(self.init_outflow_boundary_nodes)
            outflow_from_boundary_mask = np.isin(
                boundary_edge_index[0], outflow_boundary_array
            )
            print("\nOutflow edge processing:")
            print(
                f"  Edges pointing FROM outflow boundary nodes: {np.sum(outflow_from_boundary_mask)}"
            )

            if np.any(outflow_from_boundary_mask):
                print("  Flipping these edges to point TOWARD boundary")
                outflow_from_boundary = boundary_edge_index[
                    :, outflow_from_boundary_mask
                ]
                outflow_from_boundary[[0, 1], :] = outflow_from_boundary[[1, 0], :]
                boundary_edge_index[:, outflow_from_boundary_mask] = (
                    outflow_from_boundary
                )
                boundary_dynamic_edges[:, outflow_from_boundary_mask, :] *= -1

        self._boundary_edge_index = boundary_edge_index
        self._boundary_dynamic_edges = boundary_dynamic_edges

        print("\nStored boundary data (NO REMAPPING):")
        print(f"  _boundary_edge_index shape: {self._boundary_edge_index.shape}")
        print(f"  _boundary_dynamic_edges shape: {self._boundary_dynamic_edges.shape}")
        print("  Boundary edges still use original node indices")

        self._is_called["create"] = True
        print("\n✓ CREATE completed")

    def remove(
        self,
        static_nodes: ndarray,
        dynamic_nodes: ndarray,
        static_edges: ndarray,
        dynamic_edges: ndarray,
        edge_index: ndarray,
    ) -> Tuple[ndarray, ndarray, ndarray, ndarray, ndarray]:
        """
        Remove ghost nodes and edges while preserving boundary nodes.
        Creates and stores comprehensive remapping information for traceability.
        """

        print("\n" + "=" * 80)
        print("BOUNDARY CONDITION: REMOVE")
        print("=" * 80)

        if not self._is_called["create"]:
            print(
                "WARNING: Attempting to remove boundary condition before it has been created."
            )

        print("\nInput shapes:")
        print(f"  static_nodes: {static_nodes.shape}")
        print(f"  dynamic_nodes: {dynamic_nodes.shape}")
        print(f"  static_edges: {static_edges.shape}")
        print(f"  dynamic_edges: {dynamic_edges.shape}")
        print(f"  edge_index: {edge_index.shape}")

        # Separate ghost nodes from boundary nodes
        all_boundary_nodes = np.concatenate(
            [
                np.array(self.init_inflow_boundary_nodes)
                if self.init_inflow_boundary_nodes
                else np.array([], dtype=int),
                np.array(self.init_outflow_boundary_nodes)
                if self.init_outflow_boundary_nodes
                else np.array([], dtype=int),
            ]
        )

        # Ghost nodes to ignore = all ghosts EXCEPT boundary nodes
        ghost_nodes_to_ignore = np.setdiff1d(self.ghost_nodes, all_boundary_nodes)

        print("\nGhost node analysis:")
        print(f"  Total ghost nodes: {len(self.ghost_nodes)}")
        print(f"  Boundary nodes (keep these): {all_boundary_nodes}")
        print(f"  Ghost nodes to remove: {len(ghost_nodes_to_ignore)}")
        print(f"  First 10 ghost nodes to remove: {ghost_nodes_to_ignore[:10]}")

        # Create masks BEFORE removal (for reference with original indices)
        num_nodes = static_nodes.shape[0]
        valid_nodes_mask_original = np.ones(num_nodes, dtype=bool)
        valid_nodes_mask_original[ghost_nodes_to_ignore] = False

        ghost_edges_mask_original = np.any(
            np.isin(edge_index, ghost_nodes_to_ignore), axis=0
        )
        valid_edges_mask_original = ~ghost_edges_mask_original

        print("\nMasks (with original indices):")
        print(f"  Total nodes: {num_nodes}")
        print(f"  Valid nodes: {np.sum(valid_nodes_mask_original)}")
        print(f"  Ghost nodes to remove: {np.sum(~valid_nodes_mask_original)}")
        print(f"  Total edges: {len(valid_edges_mask_original)}")
        print(f"  Valid edges: {np.sum(valid_edges_mask_original)}")
        print(f"  Ghost edges to remove: {np.sum(ghost_edges_mask_original)}")

        # ACTUALLY REMOVE ghost nodes and edges
        print(f"\n{'=' * 40}")
        print("REMOVING DATA")
        print(f"{'=' * 40}")

        print(f"\nRemoving {len(ghost_nodes_to_ignore)} ghost nodes...")
        print(f"  Before - static_nodes: {static_nodes.shape}")
        static_nodes_filtered = np.delete(static_nodes, ghost_nodes_to_ignore, axis=0)
        print(f"  After - static_nodes: {static_nodes_filtered.shape}")

        print(f"\n  Before - dynamic_nodes: {dynamic_nodes.shape}")
        dynamic_nodes_filtered = np.delete(dynamic_nodes, ghost_nodes_to_ignore, axis=1)
        print(f"  After - dynamic_nodes: {dynamic_nodes_filtered.shape}")

        print(f"\nRemoving {np.sum(ghost_edges_mask_original)} ghost edges...")
        ghost_edge_indices = np.where(ghost_edges_mask_original)[0]

        print(f"  Before - static_edges: {static_edges.shape}")
        static_edges_filtered = np.delete(static_edges, ghost_edge_indices, axis=0)
        print(f"  After - static_edges: {static_edges_filtered.shape}")

        print(f"\n  Before - dynamic_edges: {dynamic_edges.shape}")
        dynamic_edges_filtered = np.delete(dynamic_edges, ghost_edge_indices, axis=1)
        print(f"  After - dynamic_edges: {dynamic_edges_filtered.shape}")

        print(f"\n  Before - edge_index: {edge_index.shape}")
        edge_index_filtered = np.delete(edge_index, ghost_edge_indices, axis=1)
        print(f"  After - edge_index: {edge_index_filtered.shape}")

        # CREATE INDEX MAPPING for updating edge_index node references
        print(f"\n{'=' * 40}")
        print("UPDATING NODE INDICES IN EDGE_INDEX")
        print(f"{'=' * 40}")

        old_to_new_idx = np.full(num_nodes, -1, dtype=int)
        new_idx = 0
        for old_idx in range(num_nodes):
            if valid_nodes_mask_original[old_idx]:  # Not deleted
                old_to_new_idx[old_idx] = new_idx
                new_idx += 1

        print("\nSample index mappings (boundary nodes):")
        for bn in sorted(all_boundary_nodes)[:5]:
            print(f"  {bn} -> {old_to_new_idx[bn]}")

        # Update edge_index with new node indices
        print("\nRemapping node indices in edge_index...")
        print("  Sample BEFORE remapping (first 5 edges):")
        for i in range(min(5, edge_index_filtered.shape[1])):
            print(f"    {edge_index_filtered[0, i]} -> {edge_index_filtered[1, i]}")

        edge_index_remapped = edge_index_filtered.copy()
        for i in range(edge_index_filtered.shape[1]):
            old_from = edge_index_filtered[0, i]
            old_to = edge_index_filtered[1, i]
            edge_index_remapped[0, i] = old_to_new_idx[old_from]
            edge_index_remapped[1, i] = old_to_new_idx[old_to]

        print("\n  Sample AFTER remapping (first 5 edges):")
        for i in range(min(5, edge_index_remapped.shape[1])):
            print(f"    {edge_index_remapped[0, i]} -> {edge_index_remapped[1, i]}")

        # Update boundary edge_index
        print("\nRemapping boundary edge_index...")
        print("  Sample BEFORE (first 5 boundary edges):")
        for i in range(min(5, self._boundary_edge_index.shape[1])):
            print(
                f"    {self._boundary_edge_index[0, i]} -> {self._boundary_edge_index[1, i]}"
            )

        boundary_edge_index_remapped = self._boundary_edge_index.copy()
        for i in range(self._boundary_edge_index.shape[1]):
            old_from = self._boundary_edge_index[0, i]
            old_to = self._boundary_edge_index[1, i]
            boundary_edge_index_remapped[0, i] = old_to_new_idx[old_from]
            boundary_edge_index_remapped[1, i] = old_to_new_idx[old_to]

        self._boundary_edge_index = boundary_edge_index_remapped

        print("\n  Sample AFTER (first 5 boundary edges):")
        for i in range(min(5, self._boundary_edge_index.shape[1])):
            print(
                f"    {self._boundary_edge_index[0, i]} -> {self._boundary_edge_index[1, i]}"
            )

        # CREATE REMAPPING DICTIONARIES FOR TRACEABILITY
        print(f"\n{'=' * 40}")
        print("CREATING REMAPPING DICTIONARIES")
        print(f"{'=' * 40}")

        # Node remapping: old_idx -> new_idx (only for kept nodes)
        node_remapping = {}
        removed_nodes = []

        for old_idx in range(num_nodes):
            new_idx_val = old_to_new_idx[old_idx]
            if new_idx_val != -1:  # Node was kept
                node_remapping[int(old_idx)] = int(new_idx_val)
            else:  # Node was removed
                removed_nodes.append(int(old_idx))

        # Edge remapping: old_edge_idx -> new_edge_idx (only for kept edges)
        edge_remapping = {}
        removed_edges = []

        old_edge_idx = 0
        new_edge_idx = 0
        for i in range(len(valid_edges_mask_original)):
            if valid_edges_mask_original[i]:  # Edge was kept
                edge_remapping[old_edge_idx] = new_edge_idx
                new_edge_idx += 1
            else:  # Edge was removed
                removed_edges.append(old_edge_idx)
            old_edge_idx += 1

        # Store final dimensions
        new_num_nodes = static_nodes_filtered.shape[0]
        new_num_edges = edge_index_remapped.shape[1]

        # Create comprehensive remapping info
        remapping_info = {
            "node_remapping": node_remapping,
            "removed_nodes": removed_nodes,
            "edge_remapping": edge_remapping,
            "removed_edges": removed_edges,
            "boundary_nodes_old_to_new": {
                int(old): int(old_to_new_idx[old]) for old in all_boundary_nodes
            },
            "ghost_nodes_removed": [int(x) for x in ghost_nodes_to_ignore],
            "statistics": {
                "original_num_nodes": int(num_nodes),
                "final_num_nodes": int(new_num_nodes),
                "nodes_removed": len(removed_nodes),
                "original_num_edges": int(len(valid_edges_mask_original)),
                "final_num_edges": int(new_num_edges),
                "edges_removed": len(removed_edges),
            },
        }

        # Store in instance variable
        self._remapping_info = remapping_info

        print("\nRemapping summary:")
        print(f"  Nodes: {num_nodes} -> {new_num_nodes} (removed {len(removed_nodes)})")
        print(
            f"  Edges: {len(valid_edges_mask_original)} -> {new_num_edges} (removed {len(removed_edges)})"
        )
        print("  Remapping info stored in self._remapping_info")

        # Store masks for the NEW filtered data
        valid_nodes_mask_new = np.ones(new_num_nodes, dtype=bool)
        valid_edges_mask_new = np.ones(new_num_edges, dtype=bool)

        self._valid_nodes_mask = valid_nodes_mask_new
        self._valid_edges_mask = valid_edges_mask_new
        self._ghost_nodes_to_ignore = np.array(
            [], dtype=int
        )  # No ghosts in filtered data

        # Store mapping for reference
        self._old_to_new_node_idx = old_to_new_idx

        print(f"\n{'=' * 40}")
        print("SUMMARY")
        print(f"{'=' * 40}")
        print("\nBoundary nodes NEW indices:")
        boundary_nodes_new_indices = [old_to_new_idx[bn] for bn in all_boundary_nodes]
        print(f"  {boundary_nodes_new_indices}")

        self._is_called["remove"] = True

        print("\nFinal output shapes:")
        print(f"  static_nodes: {static_nodes_filtered.shape}")
        print(f"  dynamic_nodes: {dynamic_nodes_filtered.shape}")
        print(f"  static_edges: {static_edges_filtered.shape}")
        print(f"  dynamic_edges: {dynamic_edges_filtered.shape}")
        print(f"  edge_index: {edge_index_remapped.shape}")

        print("\n✓ Ghost nodes and edges REMOVED")
        print("✓ Boundary nodes preserved with UPDATED indices")
        print("✓ Edge indices REMAPPED to new node positions")
        print("✓ Remapping dictionaries CREATED and stored")
        print("\n✓ REMOVE completed")
        print("=" * 80 + "\n")

        return (
            static_nodes_filtered,
            dynamic_nodes_filtered,
            static_edges_filtered,
            dynamic_edges_filtered,
            edge_index_remapped,
        )

    def save_remapping(self, filepath: str) -> None:
        """
        Save node and edge remapping information to JSON file.

        Args:
            filepath: Path where JSON file will be saved

        Raises:
            ValueError: If remapping info not available (remove() not called yet)
        """
        if not hasattr(self, "_remapping_info"):
            raise ValueError("No remapping info available. Run remove() first.")

        path = Path(filepath)
        path.parent.mkdir(parents=True, exist_ok=True)

        with path.open("w") as f:
            json.dump(self._remapping_info, f, indent=2)

        print(f"✓ Remapping saved to: {filepath}")
        print(f"  - {len(self._remapping_info['node_remapping'])} node mappings")
        print(f"  - {len(self._remapping_info['removed_nodes'])} removed nodes")
        print(f"  - {len(self._remapping_info['edge_remapping'])} edge mappings")
        print(f"  - {len(self._remapping_info['removed_edges'])} removed edges")

    def load_remapping(self, filepath: str) -> dict:
        """
        Load remapping information from JSON file.

        Args:
            filepath: Path to JSON file containing remapping info

        Returns:
            Dictionary containing remapping information
        """
        with open(filepath, "r") as f:
            remapping_info = json.load(f)

        # Convert string keys back to integers for node_remapping
        remapping_info["node_remapping"] = {
            int(k): v for k, v in remapping_info["node_remapping"].items()
        }
        remapping_info["edge_remapping"] = {
            int(k): v for k, v in remapping_info["edge_remapping"].items()
        }
        remapping_info["boundary_nodes_old_to_new"] = {
            int(k): v for k, v in remapping_info["boundary_nodes_old_to_new"].items()
        }

        self._remapping_info = remapping_info

        print(f"✓ Remapping loaded from: {filepath}")
        print(f"  - {len(remapping_info['node_remapping'])} node mappings")
        print(f"  - {len(remapping_info['removed_nodes'])} removed nodes")

        return remapping_info

    def apply(
        self,
        static_nodes: ndarray,
        dynamic_nodes: ndarray,
        static_edges: ndarray,
        dynamic_edges: ndarray,
        edge_index: ndarray,
    ) -> Tuple[ndarray, ndarray, ndarray, ndarray, ndarray]:
        print("\n" + "=" * 80)
        print("BOUNDARY CONDITION: APPLY")
        print("=" * 80)

        if not self._is_called["create"] or not self._is_called["remove"]:
            raise RuntimeError(
                "Boundary condition must be created and removed before applying."
            )

        print("\nInput shapes (all data still present):")
        print(f"  static_nodes: {static_nodes.shape}")
        print(f"  dynamic_nodes: {dynamic_nodes.shape}")
        print(f"  static_edges: {static_edges.shape}")
        print(f"  dynamic_edges: {dynamic_edges.shape}")
        print(f"  edge_index: {edge_index.shape}")

        print("\n✓ Using mask-based filtering - all original indices preserved")
        print(f"  Valid nodes: {np.sum(self._valid_nodes_mask)}")
        print(f"  Valid edges: {np.sum(self._valid_edges_mask)}")

        # Only add boundary EDGES
        if (
            self._boundary_edge_index is not None
            and self._boundary_edge_index.shape[1] > 0
        ):
            _, num_static_edge_feat = static_edges.shape

            print(f"\nAdding {self._boundary_edge_index.shape[1]} boundary edges...")

            # Create zero static features for boundary edges
            boundary_static_edges = np.zeros(
                (self._boundary_edge_index.shape[1], num_static_edge_feat),
                dtype=static_edges.dtype,
            )

            print(f"  Before concatenate - static_edges: {static_edges.shape}")
            static_edges = np.concatenate([static_edges, boundary_static_edges], axis=0)
            print(f"  After concatenate - static_edges: {static_edges.shape}")

            print(f"\n  Before concatenate - dynamic_edges: {dynamic_edges.shape}")
            dynamic_edges = np.concatenate(
                [dynamic_edges, self._boundary_dynamic_edges], axis=1
            )
            print(f"  After concatenate - dynamic_edges: {dynamic_edges.shape}")

            print(f"\n  Before concatenate - edge_index: {edge_index.shape}")
            print(f"  Boundary edge_index to add: {self._boundary_edge_index.shape}")
            print("  First 5 boundary edges (original indices):")
            for i in range(min(5, self._boundary_edge_index.shape[1])):
                print(
                    f"    {self._boundary_edge_index[0, i]} -> {self._boundary_edge_index[1, i]}"
                )

            edge_index = np.concatenate([edge_index, self._boundary_edge_index], axis=1)
            print(f"  After concatenate - edge_index: {edge_index.shape}")

            # Extend masks for new boundary edges
            boundary_edges_valid_mask = np.ones(
                self._boundary_edge_index.shape[1], dtype=bool
            )
            self._valid_edges_mask = np.concatenate(
                [self._valid_edges_mask, boundary_edges_valid_mask]
            )

        new_num_nodes = static_nodes.shape[0]

        print(f"\nCreating masks for {new_num_nodes} total nodes...")
        self._create_masks(new_num_nodes, edge_index)

        # Get boundary node indices (these are the ORIGINAL indices, not shifted)
        all_boundary_nodes = np.concatenate(
            [
                np.array(self.init_inflow_boundary_nodes)
                if self.init_inflow_boundary_nodes
                else np.array([], dtype=int),
                np.array(self.init_outflow_boundary_nodes)
                if self.init_outflow_boundary_nodes
                else np.array([], dtype=int),
            ]
        )

        print("\nMasks created:")
        print(
            f"  boundary_nodes_mask: {self.boundary_nodes_mask.shape}, True count: {np.sum(self.boundary_nodes_mask)}"
        )
        print(f"  Boundary node indices (ORIGINAL): {all_boundary_nodes}")
        print(
            f"  boundary_edges_mask: {self.boundary_edges_mask.shape}, True count: {np.sum(self.boundary_edges_mask)}"
        )
        print(
            f"  inflow_edges_mask: {self.inflow_edges_mask.shape}, True count: {np.sum(self.inflow_edges_mask)}"
        )
        print(
            f"  outflow_edges_mask: {self.outflow_edges_mask.shape}, True count: {np.sum(self.outflow_edges_mask)}"
        )

        # Clear boundary condition attributes to save memory
        self._boundary_dynamic_edges = None
        self._boundary_edge_index = None

        self._is_called["apply"] = True

        print("\nFinal output shapes:")
        print(f"  static_nodes: {static_nodes.shape}")
        print(f"  dynamic_nodes: {dynamic_nodes.shape}")
        print(f"  static_edges: {static_edges.shape}")
        print(f"  dynamic_edges: {dynamic_edges.shape}")
        print(f"  edge_index: {edge_index.shape}")

        print("\n✓ All original indices preserved!")
        print("✓ Use self._valid_nodes_mask and self._valid_edges_mask to filter")
        print("\n✓ APPLY completed")
        print("=" * 80 + "\n")

        return static_nodes, dynamic_nodes, static_edges, dynamic_edges, edge_index

    def _create_masks(self, new_num_nodes: int, edge_index: ndarray) -> None:
        """Create masks using ORIGINAL boundary node indices"""

        # Use ORIGINAL boundary node indices
        all_boundary_nodes = np.concatenate(
            [
                np.array(self.init_inflow_boundary_nodes)
                if self.init_inflow_boundary_nodes
                else np.array([], dtype=int),
                np.array(self.init_outflow_boundary_nodes)
                if self.init_outflow_boundary_nodes
                else np.array([], dtype=int),
            ]
        )

        self.boundary_nodes_mask = np.isin(np.arange(new_num_nodes), all_boundary_nodes)

        # Create masks even if empty (will be all False)
        if len(all_boundary_nodes) > 0:
            self.boundary_edges_mask = np.any(
                np.isin(edge_index, all_boundary_nodes), axis=0
            )
        else:
            self.boundary_edges_mask = np.zeros(edge_index.shape[1], dtype=bool)

        if self.init_inflow_boundary_nodes and len(self.init_inflow_boundary_nodes) > 0:
            inflow_array = np.array(self.init_inflow_boundary_nodes)
            self.inflow_edges_mask = np.any(np.isin(edge_index, inflow_array), axis=0)
        else:
            self.inflow_edges_mask = np.zeros(edge_index.shape[1], dtype=bool)

        if (
            self.init_outflow_boundary_nodes
            and len(self.init_outflow_boundary_nodes) > 0
        ):
            outflow_array = np.array(self.init_outflow_boundary_nodes)
            self.outflow_edges_mask = np.any(np.isin(edge_index, outflow_array), axis=0)
        else:
            self.outflow_edges_mask = np.zeros(edge_index.shape[1], dtype=bool)

    def get_new_boundary_nodes(self) -> ndarray:
        return np.union1d(
            self.new_inflow_boundary_nodes, self.new_outflow_boundary_nodes
        )
