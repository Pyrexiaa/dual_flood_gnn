"""
CSV-based loaders for the pre-extracted 1D/2D flood dataset.

The data has already been extracted from HEC-RAS into per-split CSV folders:

    <root_dir>/
        train/            test/
            2d_nodes_static.csv        (node_idx, position_x, position_y, area,
                                        roughness, min_elevation, elevation,
                                        aspect, curvature, flow_accumulation)
            2d_edges_static.csv        (edge_idx, relative_position_x,
                                        relative_position_y, face_length, length, slope)
            1d_nodes_static.csv        (node_idx, position_x, position_y, depth,
                                        invert_elevation, surface_elevation, base_area)
            1d_edges_static.csv        (edge_idx, relative_position_x,
                                        relative_position_y, length, diameter,
                                        shape, roughness, slope)
            2d_edge_index.csv          (edge_idx, from_node, to_node)
            1d_edge_index.csv          (edge_idx, from_node, to_node)
            1d2d_connections.csv       (connection_idx, node_1d, node_2d)
            dataset_summary.csv
            event_<id>/
                2d_nodes_dynamic_all.csv  (timestep, node_idx, rainfall, water_level, water_volume)
                2d_edges_dynamic_all.csv  (timestep, edge_idx, flow, velocity)
                1d_nodes_dynamic_all.csv  (timestep, node_idx, water_level, inlet_flow)
                1d_edges_dynamic_all.csv  (timestep, edge_idx, flow, velocity)
                timesteps.csv             (timestep_idx, timestamp)

These loaders build exactly the same in-memory ``Data1D2D`` objects that the
HEC-RAS pipeline produced, so the model / trainer / tester are unchanged. They
reuse every windowing / label helper from ``FloodEvent1D2DDataset`` (and its
autoregressive subclass) and only replace the raw-data ingestion.

Notes
-----
* Rainfall is spatially uniform, so 1D nodes are assigned the per-timestep basin
  rainfall (matching DYNAMIC_1D_NODE_FEATURES = ["rainfall", "water_level"]).
* The pre-extracted data contains no boundary/ghost nodes (indices are already
  0..N-1), so all boundary masks are empty and every node/link is predicted.
"""

import os
import gc
import numpy as np
import pandas as pd
import torch

from types import SimpleNamespace
from typing import Dict, List

from .flood_event_1d2d_dataset import FloodEvent1D2DDataset
from .autoregressive_flood_1d2d_dataset import AutoregressiveFlood1D2DDataset
from .flood_event_1d2d_dataset_wrapper import Data1D2D
from .dataset_normalizer import DatasetNormalizer


class _CsvFlood1D2DMixin:
    """Shared CSV ingestion. Bypasses the PyG ``process()`` machinery."""

    def _csv_init(self, **kw):
        root_dir = kw["root_dir"]
        split_dir = kw["split_dir"]
        event_ids = list(kw["event_ids"])
        mode = kw["mode"]

        # --- logging ---
        logger = kw.get("logger", None)
        self.log_func = print
        if logger is not None and hasattr(logger, "log"):
            self.log_func = logger.log

        # --- PyG plumbing (we do NOT call Dataset.__init__) ---
        self.root = root_dir
        self.transform = None
        self.pre_transform = None
        self.pre_filter = None
        self._indices = None
        self.log = False

        # --- config ---
        self.mode = mode
        self.previous_timesteps = int(kw.get("previous_timesteps", 1))
        self.is_normalized = bool(kw.get("normalize", True))
        self.timestep_interval = int(kw.get("timestep_interval", 300))
        self.features_stats_file = kw.get("features_stats_file", "features_stats.yaml")
        self.event_stats_file = kw.get("event_stats_file", "event_stats.yaml")
        self.model_name = kw.get("model_name", "Model1")
        self.with_global_mass_loss = bool(kw.get("with_global_mass_loss", False))
        self.with_local_mass_loss = bool(kw.get("with_local_mass_loss", False))
        self.num_label_timesteps = int(kw.get("num_label_timesteps", 1))

        # --- feature counts (class attrs come from FloodEvent1D2DDataset) ---
        self.num_static_node_features = len(self.STATIC_NODE_FEATURES)
        self.num_dynamic_node_features = len(self.DYNAMIC_NODE_FEATURES)
        self.num_static_edge_features = len(self.STATIC_EDGE_FEATURES)
        self.num_dynamic_edge_features = len(self.DYNAMIC_EDGE_FEATURES)
        self.num_static_1d_node_features = len(self.STATIC_1D_NODE_FEATURES)
        self.num_dynamic_1d_node_features = len(self.DYNAMIC_1D_NODE_FEATURES)
        self.num_static_1d_edge_features = len(self.STATIC_1D_EDGE_FEATURES)
        self.num_dynamic_1d_edge_features = len(self.DYNAMIC_1D_EDGE_FEATURES)

        # --- run ids ---
        self.hec_ras_run_ids = list(event_ids)

        # --- normalizer (train computes & saves stats; test loads them) ---
        self.normalizer = DatasetNormalizer(mode, root_dir, self.features_stats_file)

        # --- read data ---
        self._read_static(split_dir)
        self._read_events(split_dir, event_ids)
        self._normalize_all()
        self._make_boundary_condition()
        self._save_constants()
        self._compute_event_indices()

        # --- cache constants for lazy per-sample construction ---
        # NOTE: samples are built lazily in get() rather than materialized up front.
        # Materializing every autoregressive rollout window (nodes x features x
        # num_label_timesteps) for large graphs needs tens of GB; lazy construction
        # keeps only the per-event arrays in memory (~1-2 GB).
        self._prepare_runtime_cache()
        gc.collect()

    def _prepare_runtime_cache(self) -> None:
        self._static_nodes_f = self._static_nodes.astype(np.float32)
        self._static_edges_f = self._static_edges.astype(np.float32)
        self._static_nodes_1d_f = self._static_nodes_1d.astype(np.float32)
        self._static_edges_1d_f = self._static_edges_1d.astype(np.float32)
        for ev in self._events:
            ev.dynamic_nodes = ev.dynamic_nodes.astype(np.float32)
            ev.dynamic_edges = ev.dynamic_edges.astype(np.float32)
            ev.dynamic_nodes_1d = ev.dynamic_nodes_1d.astype(np.float32)
            ev.dynamic_edges_1d = ev.dynamic_edges_1d.astype(np.float32)
            ev.node_rainfall_per_ts = ev.node_rainfall_per_ts.astype(np.float32)
        self._t_edge_index = torch.from_numpy(self._edge_index.copy()).long()
        self._t_edge_index_1d = torch.from_numpy(self._edge_index_1d.copy()).long()
        self._t_edge_index_1d_2d = torch.from_numpy(self._edge_index_1d_2d.copy()).long()
        self._event_start_arr = np.asarray(self.event_start_idx, dtype=np.int64)

    # ---------------- CSV reading ----------------

    @staticmethod
    def _select(df: pd.DataFrame, cols: List[str]) -> np.ndarray:
        return df[cols].to_numpy(dtype=np.float64)

    def _read_static(self, split_dir: str) -> None:
        n2 = pd.read_csv(os.path.join(split_dir, "2d_nodes_static.csv"))
        e2 = pd.read_csv(os.path.join(split_dir, "2d_edges_static.csv"))
        n1 = pd.read_csv(os.path.join(split_dir, "1d_nodes_static.csv"))
        e1 = pd.read_csv(os.path.join(split_dir, "1d_edges_static.csv"))

        self._static_nodes = self._select(n2, self.STATIC_NODE_FEATURES)
        self._static_edges = self._select(e2, self.STATIC_EDGE_FEATURES)
        self._static_nodes_1d = self._select(n1, self.STATIC_1D_NODE_FEATURES)
        self._static_edges_1d = self._select(e1, self.STATIC_1D_EDGE_FEATURES)

        ei2 = pd.read_csv(os.path.join(split_dir, "2d_edge_index.csv"))
        ei1 = pd.read_csv(os.path.join(split_dir, "1d_edge_index.csv"))
        c12 = pd.read_csv(os.path.join(split_dir, "1d2d_connections.csv"))

        self._edge_index = ei2[["from_node", "to_node"]].to_numpy(dtype=np.int64).T
        self._edge_index_1d = ei1[["from_node", "to_node"]].to_numpy(dtype=np.int64).T
        self._edge_index_1d_2d = c12[["node_1d", "node_2d"]].to_numpy(dtype=np.int64).T

        self._num_nodes_2d = self._static_nodes.shape[0]
        self._num_edges_2d = self._static_edges.shape[0]
        self._num_nodes_1d = self._static_nodes_1d.shape[0]
        self._num_edges_1d = self._static_edges_1d.shape[0]

    @staticmethod
    def _pivot_dynamic(path: str, idx_col: str, feat_cols: List[str]) -> np.ndarray:
        """Read a long-format dynamic CSV -> array (T, N, F)."""
        df = pd.read_csv(path)
        df = df.sort_values(["timestep", idx_col], kind="stable")
        n = df[idx_col].nunique()
        t = df["timestep"].nunique()
        assert len(df) == t * n, (
            f"{path}: expected {t}*{n} rows, got {len(df)} (irregular grid)."
        )
        arr = df[feat_cols].to_numpy(dtype=np.float64).reshape(t, n, len(feat_cols))
        return arr

    def _read_events(self, split_dir: str, event_ids: List) -> None:
        self._events = []
        for ev in event_ids:
            ev_dir = os.path.join(split_dir, f"event_{ev}")

            dyn_nodes_2d = self._pivot_dynamic(
                os.path.join(ev_dir, "2d_nodes_dynamic_all.csv"),
                "node_idx", self.DYNAMIC_NODE_FEATURES,  # ["rainfall", "water_level"]
            )
            dyn_edges_2d = self._pivot_dynamic(
                os.path.join(ev_dir, "2d_edges_dynamic_all.csv"),
                "edge_idx", self.DYNAMIC_EDGE_FEATURES,  # ["flow"]
            )
            # 1D water level (rainfall added below from the uniform basin value)
            wl_1d = self._pivot_dynamic(
                os.path.join(ev_dir, "1d_nodes_dynamic_all.csv"),
                "node_idx", ["water_level"],
            )  # (T, N1, 1)
            dyn_edges_1d = self._pivot_dynamic(
                os.path.join(ev_dir, "1d_edges_dynamic_all.csv"),
                "edge_idx", self.DYNAMIC_1D_EDGE_FEATURES,  # ["flow"]
            )

            # Spatially-uniform rainfall per timestep -> broadcast to 1D nodes
            rainfall_idx = self.DYNAMIC_NODE_FEATURES.index("rainfall")
            uniform_rainfall = np.nanmean(
                dyn_nodes_2d[:, :, rainfall_idx], axis=1, keepdims=True
            )  # (T, 1)
            rainfall_1d = np.repeat(
                uniform_rainfall[:, :, None], self._num_nodes_1d, axis=1
            )  # (T, N1, 1)

            # Assemble 1D node dynamic features in DYNAMIC_1D_NODE_FEATURES order
            wl_col = self.DYNAMIC_1D_NODE_FEATURES.index("water_level")
            rf_col = self.DYNAMIC_1D_NODE_FEATURES.index("rainfall")
            dyn_nodes_1d = np.empty(
                (wl_1d.shape[0], self._num_nodes_1d, 2), dtype=np.float64
            )
            dyn_nodes_1d[:, :, rf_col] = rainfall_1d[:, :, 0]
            dyn_nodes_1d[:, :, wl_col] = wl_1d[:, :, 0]

            # Rainfall on 2D nodes (denormalized) for physics-loss info
            node_rainfall_per_ts = dyn_nodes_2d[:, :, rainfall_idx].copy()

            ts_df = pd.read_csv(os.path.join(ev_dir, "timesteps.csv"))
            event_timesteps = ts_df["timestamp"].to_numpy()

            self._events.append(
                SimpleNamespace(
                    run_id=ev,
                    dynamic_nodes=dyn_nodes_2d,
                    dynamic_edges=dyn_edges_2d,
                    dynamic_nodes_1d=dyn_nodes_1d,
                    dynamic_edges_1d=dyn_edges_1d,
                    node_rainfall_per_ts=node_rainfall_per_ts,
                    event_timesteps=event_timesteps,
                    num_timesteps=dyn_nodes_2d.shape[0],
                )
            )

    # ---------------- normalization ----------------

    def _normalize_all(self) -> None:
        if not self.is_normalized:
            return

        # Static (order mirrors the HEC-RAS pipeline: 2D block, then 1D block)
        self._static_nodes = self.normalizer.normalize_feature_vector(
            self.STATIC_NODE_FEATURES, self._static_nodes
        )
        self._static_edges = self.normalizer.normalize_feature_vector(
            self.STATIC_EDGE_FEATURES, self._static_edges
        )
        self._static_nodes_1d = self.normalizer.normalize_feature_vector(
            self.STATIC_1D_NODE_FEATURES, self._static_nodes_1d
        )
        self._static_edges_1d = self.normalizer.normalize_feature_vector(
            self.STATIC_1D_EDGE_FEATURES, self._static_edges_1d
        )

        # Dynamic: concatenate across events -> normalize once -> split back
        def norm_dynamic(attr: str, feature_list: List[str]) -> None:
            lengths = [getattr(ev, attr).shape[0] for ev in self._events]
            stacked = np.concatenate([getattr(ev, attr) for ev in self._events], axis=0)
            stacked = self.normalizer.normalize_feature_vector(feature_list, stacked)
            start = 0
            for ev, ln in zip(self._events, lengths):
                setattr(ev, attr, stacked[start : start + ln])
                start += ln

        norm_dynamic("dynamic_nodes", self.DYNAMIC_NODE_FEATURES)
        norm_dynamic("dynamic_edges", self.DYNAMIC_EDGE_FEATURES)
        norm_dynamic("dynamic_nodes_1d", self.DYNAMIC_1D_NODE_FEATURES)
        norm_dynamic("dynamic_edges_1d", self.DYNAMIC_1D_EDGE_FEATURES)

        if self.mode == "train":
            os.makedirs(os.path.join(self.root, "processed"), exist_ok=True)
            self.normalizer.save_feature_stats()
            self.log_func(
                f"Saved feature stats to {self.normalizer.feature_stats_path}"
            )

    # ---------------- boundary / constants / indices ----------------

    def _make_boundary_condition(self) -> None:
        # Pre-extracted data has no boundary/ghost nodes -> empty masks
        self.inflow_boundary_nodes = None
        self.outflow_boundary_nodes = None
        self.boundary_condition = SimpleNamespace(
            boundary_nodes_mask=np.zeros(self._num_nodes_2d, dtype=bool),
            boundary_edges_mask=np.zeros(self._num_edges_2d, dtype=bool),
            inflow_edges_mask=np.zeros(self._num_edges_2d, dtype=bool),
            outflow_edges_mask=np.zeros(self._num_edges_2d, dtype=bool),
        )

    def _save_constants(self) -> None:
        # Write constant_values.npz so BatchTensorAligner (processed_paths[3]) works
        os.makedirs(os.path.join(self.root, "processed"), exist_ok=True)
        np.savez(
            self.processed_paths[3],
            edge_index=self._edge_index,
            static_nodes=self._static_nodes,
            static_edges=self._static_edges,
            edge_index_1d=self._edge_index_1d,
            edge_index_1d_2d=self._edge_index_1d_2d,
            static_nodes_1d=self._static_nodes_1d,
            static_edges_1d=self._static_edges_1d,
        )

    def _compute_event_indices(self) -> None:
        trim_start = self.previous_timesteps
        trim_end = self.num_label_timesteps
        self.event_start_idx = []
        total = 0
        for ev in self._events:
            rollout = ev.num_timesteps - trim_start - trim_end
            assert rollout > 0, (
                f"Event {ev.run_id} has too few timesteps "
                f"({ev.num_timesteps}) for previous={trim_start}, labels={trim_end}."
            )
            self.event_start_idx.append(total)
            total += rollout
        self.total_rollout_timesteps = total

    # ---------------- lazy per-sample construction ----------------

    def _locate(self, idx: int):
        """Return (event_idx, within_event_idx) for a global rollout index."""
        event_idx = int(np.searchsorted(self._event_start_arr, idx, side="right") - 1)
        start_idx = self.event_start_idx[event_idx]
        within_event_idx = idx - start_idx + self.previous_timesteps
        return event_idx, within_event_idx

    def get(self, idx):
        if idx < 0:
            idx += self.total_rollout_timesteps
        if idx < 0 or idx >= self.total_rollout_timesteps:
            raise IndexError(
                f"Index {idx} out of bounds for dataset with "
                f"{self.total_rollout_timesteps} timesteps."
            )
        event_idx, within = self._locate(idx)
        ev = self._events[event_idx]

        node_2d = self._get_2d_node_timestep_data(
            self._static_nodes_f, ev.dynamic_nodes, within
        )
        edge_2d = self._get_2d_edge_timestep_data(
            self._static_edges_f, ev.dynamic_edges, within
        )
        node_1d = self._get_1d_node_timestep_data(
            self._static_nodes_1d_f, ev.dynamic_nodes_1d, within
        )
        edge_1d = self._get_1d_edge_timestep_data(
            self._static_edges_1d_f, ev.dynamic_edges_1d, within
        )
        y_2d, y_2d_edge, y_1d, y_1d_edge = self._get_timestep_labels(
            ev.dynamic_nodes,
            ev.dynamic_edges,
            ev.dynamic_nodes_1d,
            ev.dynamic_edges_1d,
            within,
        )

        global_mass_info = None
        if self.with_global_mass_loss:
            global_mass_info = self._get_global_mass_info_for_timestep(
                ev.node_rainfall_per_ts, within
            )
        local_mass_info = None
        if self.with_local_mass_loss:
            local_mass_info = self._get_local_mass_info_for_timestep(
                ev.node_rainfall_per_ts, within
            )

        data = Data1D2D(
            x=node_2d,
            edge_index=self._t_edge_index,
            edge_attr=edge_2d,
            y=y_2d,
            y_edge=y_2d_edge,
            x_1d=node_1d,
            edge_index_1d=self._t_edge_index_1d,
            edge_attr_1d=edge_1d,
            y_1d=y_1d,
            y_1d_edge=y_1d_edge,
            edge_index_1d_2d=self._t_edge_index_1d_2d,
            timestep=ev.event_timesteps[within],
            global_mass_info=global_mass_info,
            local_mass_info=local_mass_info,
        )
        data.num_nodes = node_2d.size(0)
        data.num_nodes_1d = node_1d.size(0)
        return data

    def len(self):
        return self.total_rollout_timesteps


# NOTE: the mixin is listed FIRST so its get()/len() take precedence over the
# parent's npz-based implementations, while _get_*_timestep_data / _get_timestep_labels
# still resolve to the (autoregressive or single-step) parent versions via the MRO.
class CsvInMemoryFlood1D2DDataset(_CsvFlood1D2DMixin, FloodEvent1D2DDataset):
    """Non-autoregressive (single-step) CSV dataset for validation / testing."""

    def __init__(self, **kwargs):
        kwargs.setdefault("num_label_timesteps", 1)
        self._csv_init(**kwargs)


class CsvInMemoryAutoregressiveFlood1D2DDataset(
    _CsvFlood1D2DMixin, AutoregressiveFlood1D2DDataset
):
    """Autoregressive (multi-step) CSV dataset for training."""

    def __init__(self, **kwargs):
        kwargs.setdefault("num_label_timesteps", 1)
        self._csv_init(**kwargs)


def list_event_ids(split_dir: str) -> List:
    """Return event ids (folder suffix after 'event_') sorted numerically when possible."""
    ids = []
    for name in os.listdir(split_dir):
        if os.path.isdir(os.path.join(split_dir, name)) and name.startswith("event_"):
            ids.append(name[len("event_"):])
    def _key(s):
        try:
            return (0, int(s))
        except ValueError:
            return (1, s)
    return sorted(ids, key=_key)
