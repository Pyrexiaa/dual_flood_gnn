"""
Aligns 1D and 2D node/edge feature tensors to a shared schema using two
distinct strategies:

    align_common_features()
        Strict intersection — only features that exist in BOTH 1D and 2D
        with the same name (or a registered alias) are kept. No cross-domain
        value borrowing. Safe, conservative, zero assumptions.

    align_with_extrapolation()
        Starts from the common-feature intersection, then fills any feature
        that is present in 2D but missing from 1D by looking up the value
        from the spatially nearest 2D neighbour. Useful when 1D nodes are
        physically embedded in the 2D domain (e.g. manholes inside 2D cells).

    align_1d_to_full_2d_schema()
        Makes 1D nodes mirror the COMPLETE 2D node schema. Common features
        are taken from 1D's own values (with aliases); every remaining 2D
        feature — static or dynamic — that is absent from 1D is filled from
        the spatially nearest 2D neighbour. Output shape for both 2D and 1D
        is [N, F_2d_total], column-aligned to the full 2D feature order.
        Edge features are NOT extrapolated (geometry semantics differ).

Comparison
----------
Method                      | 1D node output shape    | 2D node output shape
----------------------------|-------------------------|---------------------
align_common_features       | [N_1d, F_common]        | [N_2d, F_common]
align_with_extrapolation    | [N_1d, F_common+F_dyn]  | [N_2d, F_common+F_dyn]
align_1d_to_full_2d_schema  | [N_1d, F_2d_total]      | [N_2d, F_2d_total]

Feature name aliases
--------------------
Some features represent the same physical quantity under different names
across domains. Register them in FEATURE_ALIASES so all methods treat them
as the same feature.

    FEATURE_ALIASES = {
        '1d_name': '2d_name',   # 1D name → canonical (2D) name
        ...
    }

Currently registered:
    surface_elevation → elevation   (same physical surface level)

Concrete resolved schemas (given the project feature lists)
-----------------------------------------------------------
  Common node    : position_x, position_y, elevation, water_level
  Common edge    : relative_position_x, relative_position_y, length, slope,
                   roughness, flow
  Extrap dynamic : rainfall
  Full 2D node   : position_x, position_y, area, roughness, elevation, aspect,
                   curvature, flow_accumulation, rainfall, water_level
                   (1D missing columns filled from nearest 2D neighbour)
"""

import torch
from torch import Tensor
import numpy as np
from scipy.spatial import cKDTree
from typing import Tuple, Dict, List, Optional


# ---------------------------------------------------------------------------
# Alias map: 1D feature name → canonical name used in 2D (and in output)
# ---------------------------------------------------------------------------

FEATURE_ALIASES: Dict[str, str] = {
    "surface_elevation": "elevation",  # 1D surface_elevation == 2D elevation
}


# ---------------------------------------------------------------------------
# BatchTensorAligner
# ---------------------------------------------------------------------------


class BatchTensorAligner:
    """
    Aligns 1D and 2D node/edge tensors to a shared feature schema.

    Constructed ONCE from dataset metadata. All per-batch operations are
    pure tensor index lookups — no KD-tree or Python loops at runtime.

    Parameters
    ----------
    dataset : AutoregressiveFlood1D2DDataset
        Source of feature name lists and static node position arrays.
        Expected attributes:
            STATIC_NODE_FEATURES        list[str]
            DYNAMIC_NODE_FEATURES       list[str]
            STATIC_1D_NODE_FEATURES     list[str]
            DYNAMIC_1D_NODE_FEATURES    list[str]
            STATIC_EDGE_FEATURES        list[str]
            DYNAMIC_EDGE_FEATURES       list[str]
            STATIC_1D_EDGE_FEATURES     list[str]
            DYNAMIC_1D_EDGE_FEATURES    list[str]
            num_static_node_features    int
            num_static_1d_node_features int
            num_static_edge_features    int
            num_static_1d_edge_features int
            processed_paths[3]          str  path to constant_values.npz
                                             (contains 'static_nodes' and 'static_nodes_1d')
    """

    def __init__(self, dataset):
        # ── Raw feature name lists ─────────────────────────────────────────────
        self._static_2d   = dataset.STATIC_NODE_FEATURES
        self._dynamic_2d  = dataset.DYNAMIC_NODE_FEATURES
        self._static_1d   = dataset.STATIC_1D_NODE_FEATURES
        self._dynamic_1d  = dataset.DYNAMIC_1D_NODE_FEATURES
        self._static_e2d  = dataset.STATIC_EDGE_FEATURES
        self._dynamic_e2d = dataset.DYNAMIC_EDGE_FEATURES
        self._static_e1d  = dataset.STATIC_1D_EDGE_FEATURES
        self._dynamic_e1d = dataset.DYNAMIC_1D_EDGE_FEATURES

        # ── Feature offsets in the concatenated tensor ─────────────────────────
        # Tensor layout: [static | dyn_feat_0_t-W+1 | dyn_feat_0_t-W+2 | ... | dyn_feat_0_t |
        #                          dyn_feat_1_t-W+1 | ...               | dyn_feat_1_t | ...]
        # W = previous_timesteps + 1  (sliding window width)
        # The LAST slot of each dynamic feature (index W-1 within its window) holds
        # the current timestep value — this is the one the aligner selects.
        self._n_static_2d   = dataset.num_static_node_features
        self._n_static_1d   = dataset.num_static_1d_node_features
        self._n_static_e2d  = dataset.num_static_edge_features
        self._n_static_e1d  = dataset.num_static_1d_edge_features
        self._window_size   = dataset.previous_timesteps + 1   # W

        # ── Resolve common features ────────────────────────────────────────────
        # Node
        (
            self._common_node_names,
            self._common_static_node_cols_2d,
            self._common_static_node_cols_1d,
            self._common_dynamic_node_cols_2d,
            self._common_dynamic_node_cols_1d,
        ) = self._resolve_common_features(
            static_2d=self._static_2d,
            dynamic_2d=self._dynamic_2d,
            static_1d=self._static_1d,
            dynamic_1d=self._dynamic_1d,
            n_static_2d=self._n_static_2d,
            n_static_1d=self._n_static_1d,
            window_size=self._window_size,
        )

        # Edge
        (
            self._common_edge_names,
            self._common_static_edge_cols_2d,
            self._common_static_edge_cols_1d,
            self._common_dynamic_edge_cols_2d,
            self._common_dynamic_edge_cols_1d,
        ) = self._resolve_common_features(
            static_2d=self._static_e2d,
            dynamic_2d=self._dynamic_e2d,
            static_1d=self._static_e1d,
            dynamic_1d=self._dynamic_e1d,
            n_static_2d=self._n_static_e2d,
            n_static_1d=self._n_static_e1d,
            window_size=self._window_size,
        )

        # ── Resolve extrapolatable features (dynamic only) ────────────────────
        # Features present in 2D dynamic nodes but absent from 1D dynamic nodes.
        # Used by align_with_extrapolation().
        (
            self._extrap_node_names,
            self._extrap_dynamic_node_cols_2d,
        ) = self._resolve_extrapolatable_features(
            dynamic_2d=self._dynamic_2d,
            dynamic_1d=self._dynamic_1d,
            n_static_2d=self._n_static_2d,
        )

        # ── Resolve full 2D schema — missing columns for 1D ───────────────────
        # For align_1d_to_full_2d_schema(): every 2D feature (static + dynamic)
        # that is absent from 1D needs to be filled from the nearest 2D neighbour.
        # We store:
        #   _full_2d_node_names          — ordered list of ALL 2D feature names
        #   _full_2d_own_cols_1d         — for features 1D already has: col index in raw 1D tensor
        #   _full_2d_nearest_cols_2d     — for features 1D is missing: col index in raw 2D tensor
        #   _full_2d_own_positions       — output column positions filled from 1D's own data
        #   _full_2d_nearest_positions   — output column positions filled from nearest 2D
        (
            self._full_2d_node_names,
            self._full_2d_own_cols_1d,
            self._full_2d_nearest_cols_2d,
            self._full_2d_own_positions,
            self._full_2d_nearest_positions,
        ) = self._resolve_full_2d_schema(
            static_2d=self._static_2d,
            dynamic_2d=self._dynamic_2d,
            static_1d=self._static_1d,
            dynamic_1d=self._dynamic_1d,
            n_static_2d=self._n_static_2d,
            n_static_1d=self._n_static_1d,
        )

        # ── KD-tree: nearest 2D node index for every 1D node ──────────────────
        # Needed by both align_with_extrapolation() and align_1d_to_full_2d_schema().
        # Computed once from static positions; stored as a long tensor.
        self._nearest_2d_idx: Optional[Tensor] = None
        needs_kdtree = (
            len(self._extrap_node_names) > 0
            or len(self._full_2d_nearest_cols_2d) > 0
        )
        if needs_kdtree:
            self._nearest_2d_idx = self._build_nearest_mapping(dataset)

        # Log resolved schema for transparency
        self._log_schema()

    # ── Private: schema resolution ─────────────────────────────────────────────

    def _canonical(self, name: str) -> str:
        """Map a 1D feature name to its canonical (2D) name via FEATURE_ALIASES."""
        return FEATURE_ALIASES.get(name, name)

    def _resolve_common_features(
        self,
        static_2d: List[str],
        dynamic_2d: List[str],
        static_1d: List[str],
        dynamic_1d: List[str],
        n_static_2d: int,
        n_static_1d: int,
        window_size: int,
    ) -> Tuple:
        """
        Find the intersection of 2D and 1D features (static and dynamic separately).
        Preserves the 2D ordering so the output schema is deterministic.

        The raw tensor layout for dynamic features is:
            [static | dyn_0_slot_0 | ... | dyn_0_slot_W-1 | dyn_1_slot_0 | ... | dyn_1_slot_W-1 | ...]
        where W = window_size = previous_timesteps + 1.
        This method selects slot W-1 (the current timestep) for each dynamic feature.

        Returns
        -------
        common_names        : list[str]  canonical feature names in output order
                             [static features... | dynamic features...]
        static_cols_2d      : LongTensor  column indices into raw 2D tensor
        static_cols_1d      : LongTensor  column indices into raw 1D tensor
        dynamic_cols_2d     : LongTensor  column indices into raw 2D tensor (last slot)
        dynamic_cols_1d     : LongTensor  column indices into raw 1D tensor (last slot)
        """
        # Build canonical name → 1D raw index maps
        # For dynamic: index of the LAST (current) slot = n_static + feat_i * W + (W-1)
        canonical_static_1d  = {self._canonical(f): i for i, f in enumerate(static_1d)}
        canonical_dynamic_1d = {
            self._canonical(f): n_static_1d + i * window_size + (window_size - 1)
            for i, f in enumerate(dynamic_1d)
        }

        # Static intersection (preserve 2D order)
        s_cols_2d, s_cols_1d, s_names = [], [], []
        for i, f in enumerate(static_2d):
            if f in canonical_static_1d:
                s_cols_2d.append(i)
                s_cols_1d.append(canonical_static_1d[f])
                s_names.append(f)

        # Dynamic intersection (preserve 2D order)
        # Select the LAST slot (current timestep) for each dynamic feature
        d_cols_2d, d_cols_1d, d_names = [], [], []
        for i, f in enumerate(dynamic_2d):
            col_2d_current = n_static_2d + i * window_size + (window_size - 1)
            if f in canonical_dynamic_1d:
                d_cols_2d.append(col_2d_current)
                d_cols_1d.append(canonical_dynamic_1d[f])
                d_names.append(f)

        common_names = s_names + d_names

        return (
            common_names,
            torch.tensor(s_cols_2d, dtype=torch.long),
            torch.tensor(s_cols_1d, dtype=torch.long),
            torch.tensor(d_cols_2d, dtype=torch.long),
            torch.tensor(d_cols_1d, dtype=torch.long),
        )

    def _resolve_extrapolatable_features(
        self,
        dynamic_2d: List[str],
        dynamic_1d: List[str],
        n_static_2d: int,
    ) -> Tuple:
        """
        Find dynamic node features present in 2D but absent from 1D.
        These are candidates for spatial extrapolation from nearest 2D node.

        Returns
        -------
        extrap_names    : list[str]  feature names to extrapolate
        extrap_cols_2d  : LongTensor column indices in the raw 2D tensor
        """
        canonical_dynamic_1d = {self._canonical(f) for f in dynamic_1d}

        names, cols_2d = [], []
        for i, f in enumerate(dynamic_2d):
            if f not in canonical_dynamic_1d:
                names.append(f)
                # Select the LAST (current) slot: n_static + feat_i * W + (W-1)
                cols_2d.append(n_static_2d + i * self._window_size + (self._window_size - 1))

        return names, torch.tensor(cols_2d, dtype=torch.long)

    def _resolve_full_2d_schema(
        self,
        static_2d: List[str],
        dynamic_2d: List[str],
        static_1d: List[str],
        dynamic_1d: List[str],
        n_static_2d: int,
        n_static_1d: int,
    ) -> Tuple:
        """
        Build the mapping needed to project 1D nodes onto the full 2D feature schema.

        Iterates over every 2D feature in order (static first, then dynamic).
        For each position in the output:
          - If 1D has this feature (by name or alias) → record the 1D column index
            so the 1D node's own value is used.
          - If 1D lacks this feature → record the 2D column index so the nearest
            2D neighbour's value is borrowed.

        Returns
        -------
        full_2d_names        : list[str]  all 2D feature names in output order
        own_cols_1d          : LongTensor  1D column indices for features 1D owns
        nearest_cols_2d      : LongTensor  2D column indices for features 1D lacks
        own_positions        : LongTensor  output positions filled from 1D own data
        nearest_positions    : LongTensor  output positions filled from nearest 2D
        """
        # Canonical lookup maps for 1D features (name → raw column index in 1D tensor)
        canonical_static_1d  = {self._canonical(f): i
                                 for i, f in enumerate(static_1d)}
        # Last (current) slot for each 1D dynamic feature
        canonical_dynamic_1d = {
            self._canonical(f): n_static_1d + i * self._window_size + (self._window_size - 1)
            for i, f in enumerate(dynamic_1d)
        }

        full_2d_names = list(static_2d) + list(dynamic_2d)

        own_cols_1d       = []   # col in raw 1D tensor
        nearest_cols_2d   = []   # col in raw 2D tensor
        own_positions     = []   # which output column this fills
        nearest_positions = []   # which output column this fills

        for out_pos, f in enumerate(static_2d):
            raw_2d_col = out_pos                          # static: direct index
            if f in canonical_static_1d:
                own_cols_1d.append(canonical_static_1d[f])
                own_positions.append(out_pos)
            else:
                nearest_cols_2d.append(raw_2d_col)
                nearest_positions.append(out_pos)

        n_static_out = len(static_2d)
        for dyn_i, f in enumerate(dynamic_2d):
            out_pos    = n_static_out + dyn_i
            # Last (current) slot: n_static + feat_i * W + (W-1)
            raw_2d_col = n_static_2d + dyn_i * self._window_size + (self._window_size - 1)
            if f in canonical_dynamic_1d:
                own_cols_1d.append(canonical_dynamic_1d[f])
                own_positions.append(out_pos)
            else:
                nearest_cols_2d.append(raw_2d_col)
                nearest_positions.append(out_pos)

        return (
            full_2d_names,
            torch.tensor(own_cols_1d,       dtype=torch.long),
            torch.tensor(nearest_cols_2d,   dtype=torch.long),
            torch.tensor(own_positions,     dtype=torch.long),
            torch.tensor(nearest_positions, dtype=torch.long),
        )

    def _build_nearest_mapping(self, dataset) -> Tensor:
        """
        KD-tree query over 2D node XY positions. Runs once at construction.

        Loads static node arrays from the processed constant-values npz file
        (the same source used by dataset.get()), then extracts position_x and
        position_y using the dataset feature name lists so the column indices
        are always correct regardless of feature ordering.

        Expected dataset attributes (all present in FloodEvent1D2DDataset):
            processed_paths[3]       — path to constant_values.npz
            STATIC_NODE_FEATURES     — list[str] for 2D static node features
            STATIC_1D_NODE_FEATURES  — list[str] for 1D static node features
        """
        constant_values = np.load(dataset.processed_paths[3])
        static_nodes_2d = constant_values["static_nodes"]
        static_nodes_1d = constant_values["static_nodes_1d"]

        x_col_2d = dataset.STATIC_NODE_FEATURES.index("position_x")
        y_col_2d = dataset.STATIC_NODE_FEATURES.index("position_y")
        x_col_1d = dataset.STATIC_1D_NODE_FEATURES.index("position_x")
        y_col_1d = dataset.STATIC_1D_NODE_FEATURES.index("position_y")

        pos_2d = static_nodes_2d[:, [x_col_2d, y_col_2d]]
        pos_1d = static_nodes_1d[:, [x_col_1d, y_col_1d]]

        # Store single-graph sizes for batch offset computation at runtime
        self._n_2d_single = pos_2d.shape[0]   # ← add this
        self._n_1d_single = pos_1d.shape[0]   # ← and this (for safety checks)

        tree = cKDTree(pos_2d)
        _, nearest_idx = tree.query(pos_1d, k=1)
        return torch.tensor(nearest_idx, dtype=torch.long)

    def _log_schema(self):
        """Print resolved schema at construction time for transparency."""
        missing_from_1d = [
            self._full_2d_node_names[p.item()]
            for p in self._full_2d_nearest_positions
        ]
        print("=" * 60)
        print("BatchTensorAligner — resolved feature schema")
        print("-" * 60)
        print(f"  Common node features        : {self._common_node_names}")
        print(f"  Common edge features        : {self._common_edge_names}")
        print(f"  Extrap dynamic (1D←2D)     : {self._extrap_node_names}")
        print(f"  Full 2D schema size         : {len(self._full_2d_node_names)}")
        print(f"  Missing from 1D (nearest←) : {missing_from_1d}")
        print("=" * 60)

    # ── Private: tensor extraction helpers ────────────────────────────────────

    def _extract(self, tensor: Tensor, static_cols: Tensor, dynamic_cols: Tensor) -> Tensor:
        """
        Extract and concatenate static + dynamic columns from a raw feature tensor.
        Both index tensors may be empty (zero-length), which is handled gracefully.
        """
        parts = []
        if len(static_cols) > 0:
            parts.append(tensor[:, static_cols])
        if len(dynamic_cols) > 0:
            parts.append(tensor[:, dynamic_cols])
        if not parts:
            raise ValueError("No common features found — check feature name lists.")
        return torch.cat(parts, dim=-1)

    # ── Public API ─────────────────────────────────────────────────────────────
    def to(self, device):
        """Move all precomputed index tensors to the target device."""
        self._common_static_node_cols_2d  = self._common_static_node_cols_2d.to(device)
        self._common_static_node_cols_1d  = self._common_static_node_cols_1d.to(device)
        self._common_dynamic_node_cols_2d = self._common_dynamic_node_cols_2d.to(device)
        self._common_dynamic_node_cols_1d = self._common_dynamic_node_cols_1d.to(device)
        self._common_static_edge_cols_2d  = self._common_static_edge_cols_2d.to(device)
        self._common_static_edge_cols_1d  = self._common_static_edge_cols_1d.to(device)
        self._common_dynamic_edge_cols_2d = self._common_dynamic_edge_cols_2d.to(device)
        self._common_dynamic_edge_cols_1d = self._common_dynamic_edge_cols_1d.to(device)
        self._extrap_dynamic_node_cols_2d = self._extrap_dynamic_node_cols_2d.to(device)
        self._full_2d_own_cols_1d         = self._full_2d_own_cols_1d.to(device)
        self._full_2d_nearest_cols_2d     = self._full_2d_nearest_cols_2d.to(device)
        self._full_2d_own_positions       = self._full_2d_own_positions.to(device)
        self._full_2d_nearest_positions   = self._full_2d_nearest_positions.to(device)
        if self._nearest_2d_idx is not None:
            self._nearest_2d_idx = self._nearest_2d_idx.to(device)
        return self

    def align_common_features(
        self,
        x: Tensor,
        x_1d: Tensor,
        edge_attr: Tensor,
        edge_attr_1d: Tensor,
    ) -> Tuple[Tensor, Tensor, Tensor, Tensor]:
        """
        Common static features + full dynamic features for both domains.

        Static  : intersection of 2D and 1D static schemas (no borrowing).
        Dynamic : common dynamic features + any 2D-only dynamic features
                injected into 1D via nearest-neighbour lookup.
                This ensures both domains always receive the same dynamic
                forcing inputs (e.g. rainfall), which is physically required
                since 1D sub-surface nodes are driven by the same rainfall
                field as the 2D surface domain.

        Edge features: strict intersection, no extrapolation.
        """
        # ── Static: strict intersection ───────────────────────────────────────
        x_static    = x[:, self._common_static_node_cols_2d]
        x_1d_static = x_1d[:, self._common_static_node_cols_1d]

        # ── Dynamic: common features from each domain's own values ────────────
        x_dyn_common    = x[:, self._common_dynamic_node_cols_2d]
        x_1d_dyn_common = x_1d[:, self._common_dynamic_node_cols_1d]

        # ── Dynamic: extrapolated features (e.g. rainfall) always included ────
        if len(self._extrap_dynamic_node_cols_2d) > 0:
            extrap_from_2d = x[:, self._extrap_dynamic_node_cols_2d]   # [N_2d*B, F_extrap]

            batch_size    = x_1d.size(0) // self._n_1d_single
            offsets       = torch.arange(batch_size, device=x.device) * self._n_2d_single
            nearest_tiled = (
                self._nearest_2d_idx.unsqueeze(0) + offsets.unsqueeze(1)
            ).reshape(-1)
            extrap_for_1d = extrap_from_2d[nearest_tiled]              # [N_1d*B, F_extrap]

            x_aligned    = torch.cat([x_static,    x_dyn_common,    extrap_from_2d], dim=-1)
            x_1d_aligned = torch.cat([x_1d_static, x_1d_dyn_common, extrap_for_1d],  dim=-1)
        else:
            x_aligned    = torch.cat([x_static,    x_dyn_common],    dim=-1)
            x_1d_aligned = torch.cat([x_1d_static, x_1d_dyn_common], dim=-1)

        # ── Edges: strict intersection ────────────────────────────────────────
        edge_attr_aligned = self._extract(
            edge_attr, self._common_static_edge_cols_2d, self._common_dynamic_edge_cols_2d
        )
        edge_attr_1d_aligned = self._extract(
            edge_attr_1d, self._common_static_edge_cols_1d, self._common_dynamic_edge_cols_1d
        )

        return x_aligned, x_1d_aligned, edge_attr_aligned, edge_attr_1d_aligned

    def align_common_features_no_rainfall_1d(
        self,
        x: Tensor,
        x_1d: Tensor,
        edge_attr: Tensor,
        edge_attr_1d: Tensor,
    ) -> Tuple[Tensor, Tensor, Tensor, Tensor]:
        """
        Common static features + full dynamic features for both domains,
        but WITHOUT injecting extrapolated features (e.g. rainfall) into 1D nodes.

        Identical to align_common_features() except:
        - 2D nodes still receive [common_static | common_dynamic | extrapolated_dynamic]
        - 1D nodes receive only  [common_static | common_dynamic]
            (no nearest-neighbour extrapolation appended)

        Use this when:
        - You want to ablate or isolate the effect of rainfall forcing on 1D nodes.
        - Your 1D sub-model handles rainfall separately (e.g. via a dedicated input head).
        - You need asymmetric feature sizes between 1D and 2D intentionally.

        Edge features: strict intersection, no extrapolation (same as align_common_features).

        Args:
            x            : [N_2d, F_2d]   raw 2D node features
            x_1d         : [N_1d, F_1d]   raw 1D node features
            edge_attr    : [E_2d, Fe_2d]  raw 2D edge features
            edge_attr_1d : [E_1d, Fe_1d]  raw 1D edge features

        Returns:
            x_aligned            : [N_2d, F_common_node + F_extrap]   2D nodes (with extrapolated)
            x_1d_aligned         : [N_1d, F_common_node]              1D nodes (WITHOUT extrapolated)
            edge_attr_aligned    : [E_2d, F_common_edge]
            edge_attr_1d_aligned : [E_1d, F_common_edge]

        Output node feature order:
            2D: [common_static | common_dynamic | extrapolated_dynamic]
            1D: [common_static | common_dynamic]
        """
        # ── Static: strict intersection ───────────────────────────────────────
        x_static    = x[:, self._common_static_node_cols_2d]
        x_1d_static = x_1d[:, self._common_static_node_cols_1d]

        # ── Dynamic: common features from each domain's own values ────────────
        x_dyn_common    = x[:, self._common_dynamic_node_cols_2d]
        x_1d_dyn_common = x_1d[:, self._common_dynamic_node_cols_1d]

        # ── Dynamic: extrapolated features appended to 2D only ────────────────
        if len(self._extrap_dynamic_node_cols_2d) > 0:
            extrap_from_2d = x[:, self._extrap_dynamic_node_cols_2d]  # [N_2d*B, F_extrap]
            x_aligned    = torch.cat([x_static, x_dyn_common, extrap_from_2d], dim=-1)
        else:
            x_aligned    = torch.cat([x_static, x_dyn_common], dim=-1)

        # 1D nodes: common features only — no extrapolation injected
        x_1d_aligned = torch.cat([x_1d_static, x_1d_dyn_common], dim=-1)

        # ── Edges: strict intersection ────────────────────────────────────────
        edge_attr_aligned = self._extract(
            edge_attr, self._common_static_edge_cols_2d, self._common_dynamic_edge_cols_2d
        )
        edge_attr_1d_aligned = self._extract(
            edge_attr_1d, self._common_static_edge_cols_1d, self._common_dynamic_edge_cols_1d
        )

        return x_aligned, x_1d_aligned, edge_attr_aligned, edge_attr_1d_aligned

    def align_with_extrapolation(
        self,
        x: Tensor,
        x_1d: Tensor,
        edge_attr: Tensor,
        edge_attr_1d: Tensor,
    ) -> Tuple[Tensor, Tensor, Tensor, Tensor]:
        """
        Common-feature alignment + spatial extrapolation for missing 1D features.

        Starts from the common-feature intersection (same as align_common_features),
        then appends any 2D dynamic node feature that is absent from 1D by
        looking up its value from the spatially nearest 2D node (precomputed
        via KD-tree at construction time).

        Use this when:
        - 1D nodes are physically embedded within the 2D domain (e.g. manholes).
        - Missing 1D features have a physically meaningful 2D proxy nearby
          (e.g. rainfall — a 2D surface quantity inherited by sub-surface nodes).
        - You want the richest possible shared feature set.

        Edge features are NOT extrapolated because 1D pipe geometry and 2D
        mesh geometry are structurally different — cross-domain borrowing
        would not be physically meaningful for edges.

        Args:
            x            : [N_2d, F_2d]   raw 2D node features
            x_1d         : [N_1d, F_1d]   raw 1D node features
            edge_attr    : [E_2d, Fe_2d]  raw 2D edge features
            edge_attr_1d : [E_1d, Fe_1d]  raw 1D edge features

        Returns:
            x_aligned            : [N_2d, F_common_node + F_extrap]
            x_1d_aligned         : [N_1d, F_common_node + F_extrap]
            edge_attr_aligned    : [E_2d, F_common_edge]   (same as common-only)
            edge_attr_1d_aligned : [E_1d, F_common_edge]   (same as common-only)

        Output node feature order:
            [common_static | common_dynamic | extrapolated_dynamic]
        """
        if self._nearest_2d_idx is None:
            # No extrapolatable features were found — fall back to common only
            return self.align_common_features(x, x_1d, edge_attr, edge_attr_1d)

        # ── Step 1: common features (same as align_common_features) ───────────
        x_common, x_1d_common, edge_attr_aligned, edge_attr_1d_aligned = (
            self.align_common_features(x, x_1d, edge_attr, edge_attr_1d)
        )

        # ── Step 2: extrapolated features ─────────────────────────────────────
        # Extract the missing features from the 2D tensor.
        extrap_from_2d = x[:, self._extrap_dynamic_node_cols_2d]    # [N_2d, F_extrap]

        # 2D nodes: the extrapolated features are already their own values.
        # 1D nodes: inherit from the nearest 2D neighbour.
        n_1d_single = self._nearest_2d_idx.size(0)                 # N_1d (single graph)
        n_2d_single = self._n_2d_single                             # N_2d (single graph)
        batch_size   = x_1d.size(0) // n_1d_single                 # B
        offsets      = torch.arange(batch_size, device=x.device) * n_2d_single  # [B]
        nearest_tiled = (
            self._nearest_2d_idx.unsqueeze(0)          # [1, N_1d]
            + offsets.unsqueeze(1)                      # [B, 1]
        ).reshape(-1)                                   # [N_1d * B]

        extrap_for_1d = extrap_from_2d[nearest_tiled]              # [N_1d * B, F_extrap]

        # ── Step 3: concatenate ───────────────────────────────────────────────
        x_aligned    = torch.cat([x_common,    extrap_from_2d], dim=-1)
        x_1d_aligned = torch.cat([x_1d_common, extrap_for_1d],  dim=-1)

        return x_aligned, x_1d_aligned, edge_attr_aligned, edge_attr_1d_aligned

    def align_1d_to_full_2d_schema(
        self,
        x: Tensor,
        x_1d: Tensor,
        edge_attr: Tensor,
        edge_attr_1d: Tensor,
    ) -> Tuple[Tensor, Tensor, Tensor, Tensor]:
        """
        Project 1D nodes onto the complete 2D node feature schema.

        The output 2D and 1D node tensors have identical shape and column
        ordering — exactly matching the full 2D feature schema (static + dynamic).
        For each column in the output:

          - If 1D already has this feature (by name or alias):
              → use the 1D node's own value.
          - If 1D is missing this feature:
              → borrow the value from the spatially nearest 2D node
                (precomputed KD-tree at construction time, zero runtime cost).

        For 2D nodes the output is simply the raw 2D tensor column-selected in
        the canonical order (static features first, then dynamic features),
        which is already what the 2D tensor contains — so the 2D path is a
        no-op reorder rather than any cross-domain borrowing.

        Edge features are NOT modified beyond the common-feature alignment
        (same output as align_common_features for edges) because 1D pipe
        geometry and 2D mesh geometry are structurally different; borrowing
        across domains for edges is not physically meaningful.

        Use this when:
        - You want 1D and 2D nodes to be fully feature-compatible for a
          shared encoder without a type embedding carrying structural load.
        - You accept that some 1D node columns are approximations from the
          nearest 2D cell (e.g. area, roughness, aspect) rather than true
          1D physical properties.
        - The richer feature set is worth the approximation cost.

        Args:
            x            : [N_2d, F_2d]   raw 2D node features
            x_1d         : [N_1d, F_1d]   raw 1D node features
            edge_attr    : [E_2d, Fe_2d]  raw 2D edge features
            edge_attr_1d : [E_1d, Fe_1d]  raw 1D edge features

        Returns:
            x_aligned            : [N_2d, F_2d_total]   2D nodes, full 2D schema
            x_1d_aligned         : [N_1d, F_2d_total]   1D nodes, full 2D schema
            edge_attr_aligned    : [E_2d, F_common_edge]
            edge_attr_1d_aligned : [E_1d, F_common_edge]

        Output node feature order:
            Exactly matches the 2D schema: [static_2d... | dynamic_2d...]
        """
        n_out = len(self._full_2d_node_names)

        # ── 2D nodes: select static columns + last dynamic slot per feature ─────
        # Static: cols 0 .. n_static-1  (contiguous, direct)
        # Dynamic layout: [dyn_0_slot_0 | ... | dyn_0_slot_W-1 | dyn_1_slot_0 | ...]
        # We want the last slot of each dynamic feature: n_static + i*W + (W-1)
        static_cols_2d = torch.arange(self._n_static_2d, dtype=torch.long, device=x.device)
        dynamic_last_cols_2d = torch.tensor(
            [self._n_static_2d + i * self._window_size + (self._window_size - 1)
             for i in range(len(self._dynamic_2d))],
            dtype=torch.long, device=x.device,
        )
        all_2d_cols = torch.cat([static_cols_2d, dynamic_last_cols_2d])
        x_aligned = x[:, all_2d_cols]                            # [N_2d, F_2d_total]

        # ── 1D nodes: build output column by column ────────────────────────────
        # Allocate output, then fill own-value positions and nearest-2D positions.
        x_1d_aligned = torch.empty(
            x_1d.size(0), n_out, dtype=x_1d.dtype, device=x_1d.device
        )

        # Positions where 1D uses its own values
        if len(self._full_2d_own_positions) > 0:
            x_1d_aligned[:, self._full_2d_own_positions] = (
                x_1d[:, self._full_2d_own_cols_1d]
            )

        # Positions where 1D borrows from its nearest 2D neighbour
        if len(self._full_2d_nearest_positions) > 0:
            batch_size    = x_1d.size(0) // self._n_1d_single
            offsets       = torch.arange(batch_size, device=x.device) * self._n_2d_single
            nearest_tiled = (
                self._nearest_2d_idx.unsqueeze(0) + offsets.unsqueeze(1)
            ).reshape(-1)
            nearest_2d_values = x[:, self._full_2d_nearest_cols_2d]
            x_1d_aligned[:, self._full_2d_nearest_positions] = nearest_2d_values[nearest_tiled]

        # ── Edges: common-feature alignment only (no extrapolation for edges) ──
        edge_attr_aligned, edge_attr_1d_aligned = (
            self.align_common_features(x, x_1d, edge_attr, edge_attr_1d)[2:]
        )

        return x_aligned, x_1d_aligned, edge_attr_aligned, edge_attr_1d_aligned

    def inject_nearest_rainfall_to_1d(
        self,
        x: Tensor,
        x_1d: Tensor,
        edge_attr: Tensor,
        edge_attr_1d: Tensor,
    ) -> Tuple[Tensor, Tensor, Tensor, Tensor]:
        """
        Append the nearest 2D node's extrapolatable dynamic features (e.g. rainfall)
        as extra columns to x_1d, including the FULL window history (W slots each).

        x_1d gains F_extrap * W new columns, matching the windowed layout of all
        other dynamic features — so the model can treat injected rainfall identically
        to native 1D dynamic features.

        Output x_1d column order:
            [original_1d_features... | extrap_feat_0_slot_0 | ... | extrap_feat_0_slot_W-1 | ...]
        """
        if (
            self._nearest_2d_idx is None
            or len(self._extrap_dynamic_node_cols_2d) == 0
        ):
            return x, x_1d, edge_attr, edge_attr_1d

        # Build batch-aware nearest-neighbour index
        batch_size    = x_1d.size(0) // self._n_1d_single
        offsets       = torch.arange(batch_size, device=x.device) * self._n_2d_single
        nearest_tiled = (
            self._nearest_2d_idx.unsqueeze(0)   # [1, N_1d]
            + offsets.unsqueeze(1)              # [B,  1  ]
        ).reshape(-1)                           # [N_1d * B]

        # For each extrapolatable feature, collect ALL W window slots from 2D
        # Layout in x: [static | dyn_0_slot_0 | ... | dyn_0_slot_W-1 | dyn_1_slot_0 | ...]
        window_cols_per_feature = []
        for i, f in enumerate(self._dynamic_2d):
            if f not in {self._canonical(g) for g in self._dynamic_1d}:
                # All W slots for this feature
                slots = [
                    self._n_static_2d + i * self._window_size + slot
                    for slot in range(self._window_size)
                ]
                window_cols_per_feature.append(
                    torch.tensor(slots, dtype=torch.long, device=x.device)
                )

        if not window_cols_per_feature:
            return x, x_1d, edge_attr, edge_attr_1d

        # Gather all window columns: [N_2d*B, F_extrap * W]
        all_window_cols = torch.cat(window_cols_per_feature)          # [F_extrap * W]
        extrap_windowed = x[:, all_window_cols]                       # [N_2d*B, F_extrap*W]

        # Look up nearest 2D values for each 1D node
        extrap_for_1d = extrap_windowed[nearest_tiled]                # [N_1d*B, F_extrap*W]

        x_1d_augmented = torch.cat([x_1d, extrap_for_1d], dim=-1)

        return x, x_1d_augmented, edge_attr, edge_attr_1d

    # ── Schema introspection ───────────────────────────────────────────────────

    @property
    def aligned_dynamic_node_feature_names(self) -> List[str]:
        """
        Dynamic node features after alignment — same for ALL three methods.
        Rainfall is always included for both 1D and 2D nodes because it is
        a required forcing input. 1D nodes inherit it from the nearest 2D neighbour.
        Order: [common_dynamic... | extrapolated_dynamic...]
        which resolves to: ['water_level', 'rainfall'] for your feature schema.
        """
        return list(self._common_dynamic_node_names) + list(self._extrap_node_names)

    @property
    def common_node_feature_names(self) -> List[str]:
        """Feature names output by align_common_features() — nodes."""
        return list(self._common_node_names)

    @property
    def common_edge_feature_names(self) -> List[str]:
        """Feature names output by align_common_features() — edges."""
        return list(self._common_edge_names)

    @property
    def extrapolated_node_feature_names(self) -> List[str]:
        """Feature names appended by align_with_extrapolation() — nodes only."""
        return list(self._extrap_node_names)

    @property
    def full_2d_node_feature_names(self) -> List[str]:
        """Full 2D node feature schema used by align_1d_to_full_2d_schema()."""
        return list(self._full_2d_node_names)

    @property
    def node_feature_size_common(self) -> int:
        """Node feature size after align_common_features() — includes extrapolated dynamic."""
        return (len(self._common_static_node_cols_2d) +
                len(self._common_dynamic_node_cols_2d) +
                len(self._extrap_dynamic_node_cols_2d))

    @property
    def node_feature_size_with_extrapolation(self) -> int:
        """Node feature size after align_with_extrapolation()."""
        return len(self._common_node_names) + len(self._extrap_node_names)

    @property
    def node_feature_size_full_2d(self) -> int:
        """Node feature size after align_1d_to_full_2d_schema() — equals full 2D schema."""
        return len(self._full_2d_node_names)

    @property
    def edge_feature_size(self) -> int:
        """Edge feature size (same for all three alignment methods)."""
        return len(self._common_edge_names)
    
    @property
    def injected_feature_names(self) -> List[str]:
        """Extra feature names appended to x_1d by inject_nearest_rainfall_to_1d()."""
        return list(self._extrap_node_names)
    
    @property
    def node_feature_size_common_no_rainfall_1d(self) -> int:
        """Node feature size for 1D after align_common_features_no_rainfall_1d()."""
        return (len(self._common_static_node_cols_1d) +
                len(self._common_dynamic_node_cols_1d))