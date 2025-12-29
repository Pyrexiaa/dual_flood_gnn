from collections import defaultdict
from typing import Counter
from constants import FEATURE_NAMES, DYNAMIC_FEATURES, STATIC_FEATURES
from utils import (
    attach_timestamps,
    load_static,
    load_dynamic,
)
import torch
from torch.utils.data import Dataset
import os
import numpy as np
from pathlib import Path
import joblib


class JointWaterLevelDataset(Dataset):
    """
    Optimized Joint 1D + 2D node dataset WITHOUT expensive edge aggregation.

    Assumes edge features are already joined with nodes in the data preprocessing stage.
    """

    def __init__(
        self,
        event_dirs,
        window,
        normalizer=None,
        fit_normalizer=False,
        return_sequence=False,
        debug=False,
        max_events=None,
        max_samples=None,
        verbose=True,
        separate_static_dynamic=False,
        node_type_filter=None,
        normalizer_save_path=None,
    ):
        self.window = window
        self.return_sequence = return_sequence
        self.debug = debug
        self.verbose = verbose
        self.separate_static_dynamic = separate_static_dynamic
        self.node_type_filter = node_type_filter

        self.X, self.y, self.y_raw, self.node_type = [], [], [], []
        self.X_static, self.X_dynamic = [], []
        self.samples_after_1d = 0
        self.samples_after_2d = 0
        self.debug_samples = [] if debug else None

        # Track feature consistency across samples
        self.feature_tracking = {
            "by_sample": [],
            "by_node_type": defaultdict(list),
            "by_event": defaultdict(list),
            "inconsistencies": [],
        }

        # Normalize event_dirs to Path list
        if isinstance(event_dirs, (str, Path)):
            events_root = Path(event_dirs)
            if events_root.is_dir():
                self.event_dirs = sorted(d for d in events_root.iterdir() if d.is_dir())
            else:
                raise ValueError(f"{events_root} is not a directory")
        elif isinstance(event_dirs, (list, tuple)):
            self.event_dirs = [Path(d) for d in event_dirs]
        else:
            raise TypeError("event_dirs must be str, Path, or list of them")

        if self.verbose:
            print(f"\n{'=' * 80}")
            print("INITIALIZING DATASET (OPTIMIZED - NO EDGE AGGREGATION)")
            print(f"{'=' * 80}")
            print(f"Events to process: {len(self.event_dirs)}")
            print(f"Window size: {window}")
            print(f"Max events: {max_events}")
            print(f"Max samples: {max_samples}")
            print(f"Separate static/dynamic: {separate_static_dynamic}")

        # Process events
        for event_idx, event_dir in enumerate(self.event_dirs):
            if max_events and event_idx >= max_events:
                break
            if self.verbose:
                print(f"\n{'-' * 80}")
                print(
                    f"Processing Event {event_idx + 1}/{len(self.event_dirs)}: {event_dir.name}"
                )
                print(f"{'-' * 80}")

            samples_before = len(self.X)
            self._process_event(event_dir, event_idx)
            samples_after = len(self.X)

            if self.verbose:
                print(f"\nEvent {event_dir.name} summary:")
                print(f"  Samples created: {samples_after - samples_before}")

            if max_samples and len(self.X) >= max_samples:
                self.X = self.X[:max_samples]
                self.y = self.y[:max_samples]
                self.y_raw = self.y_raw[:max_samples]
                self.node_type = self.node_type[:max_samples]
                if self.separate_static_dynamic:
                    self.X_static = self.X_static[:max_samples]
                    self.X_dynamic = self.X_dynamic[:max_samples]
                if self.debug:
                    self.debug_samples = self.debug_samples[:max_samples]
                break

        if len(self.X) == 0:
            raise ValueError(
                "No valid samples created. Check your data and window size."
            )

        if self.verbose:
            print(f"\n{'=' * 80}")
            print("DATASET CREATION COMPLETE")
            print(f"{'=' * 80}")
            print(f"Total samples created: {len(self.X)}")

        # Validate feature consistency
        self._validate_feature_consistency()

        # Convert to arrays
        try:
            self.X = np.array(self.X, dtype=np.float32)
        except ValueError as e:
            print("\n❌ FATAL ERROR: Cannot create uniform numpy array")
            print(f"Error: {e}")
            self._print_detailed_shape_analysis()
            raise

        self.y = np.array(self.y, dtype=np.float32).reshape(-1, 1)
        self.y_raw = np.array(self.y_raw, dtype=np.float32).reshape(-1, 1)
        self.node_type = np.array(self.node_type, dtype=np.int64)

        if self.separate_static_dynamic:
            try:
                self.X_static = np.array(self.X_static, dtype=np.float32)
                self.X_dynamic = np.array(self.X_dynamic, dtype=np.float32)
            except ValueError as e:
                print("\n❌ ERROR: Cannot create uniform arrays for static/dynamic")
                print(f"Error: {e}")
                raise

            if self.verbose:
                print(f"\nStatic features shape: {self.X_static.shape}")
                print(f"Dynamic features shape: {self.X_dynamic.shape}")

        # Normalization
        if normalizer is not None:
            if fit_normalizer:
                normalizer.fit(self.X, self.y)
                joblib.dump(normalizer, normalizer_save_path or "train_normalizer.pkl")
                if self.verbose:
                    print("\nNormalizer fitted and saved")
            self.X, self.y = normalizer.transform(self.X, self.y)
            if self.verbose:
                print("Data normalized")

        # Torch tensors
        self.X = torch.tensor(self.X, dtype=torch.float32)
        self.y = torch.tensor(self.y, dtype=torch.float32)
        self.y_raw = torch.tensor(self.y_raw, dtype=torch.float32)
        self.node_type = torch.tensor(self.node_type, dtype=torch.long)

        if self.separate_static_dynamic:
            self.X_static = torch.tensor(self.X_static, dtype=torch.float32)
            self.X_dynamic = torch.tensor(self.X_dynamic, dtype=torch.float32)

        if self.verbose:
            print(f"\n{'=' * 80}")
            print("FINAL TENSOR SHAPES")
            print(f"{'=' * 80}")
            print(f"  X: {self.X.shape}")
            print(f"  y: {self.y.shape}")
            print(f"  node_type: {self.node_type.shape}")
            if self.separate_static_dynamic:
                print(f"  X_static: {self.X_static.shape}")
                print(f"  X_dynamic: {self.X_dynamic.shape}")

    def _validate_feature_consistency(self):
        """Validate that all samples have consistent feature dimensions."""
        if not self.X:
            return

        print(f"\n{'=' * 80}")
        print("VALIDATING FEATURE CONSISTENCY")
        print(f"{'=' * 80}")

        if self.return_sequence:
            shapes = [(x.shape[0], x.shape[1]) for x in self.X]
        else:
            shapes = [x.shape[0] for x in self.X]

        unique_shapes = set(shapes)

        print(f"Total samples: {len(shapes)}")
        print(f"Unique shapes: {len(unique_shapes)}")

        if len(unique_shapes) == 1:
            print(f"✓ All samples have consistent shape: {list(unique_shapes)[0]}")
            print("\nFeature padding analysis:")
            print(f"  Total feature slots: {len(FEATURE_NAMES)}")
            return

        # INCONSISTENCY DETECTED
        print("\n❌ INCONSISTENT SHAPES DETECTED!")
        print("Expected: All samples should have the same shape")
        print(f"Found: {len(unique_shapes)} different shapes\n")

        shape_counts = Counter(shapes)
        print("Shape distribution:")
        for shape, count in shape_counts.most_common():
            pct = count / len(shapes) * 100
            print(f"  {shape}: {count:>7} samples ({pct:>5.1f}%)")

        raise ValueError(
            f"\nFeature padding failed! All samples should be padded to {len(FEATURE_NAMES)} features."
        )

    def _print_detailed_shape_analysis(self):
        """Print detailed analysis when array conversion fails."""
        print(f"\n{'=' * 80}")
        print("DETAILED SHAPE ANALYSIS")
        print(f"{'=' * 80}")

        print("\nFirst 20 sample shapes:")
        for i, x in enumerate(self.X[:20]):
            print(
                f"  Sample {i}: shape={x.shape}, dtype={x.dtype}, node_type={self.node_type[i]}"
            )

    def _process_event(self, event_dir, event_idx):
        """Process a single event directory - SIMPLIFIED without edge aggregation."""
        static = load_static(os.path.dirname(event_dir))
        dynamic = load_dynamic(event_dir)

        if self.verbose:
            print(f"\nLoaded data from: {event_dir}")
            print(f"  1D nodes (static): {len(static.get('1d_node', []))} nodes")
            print(f"  2D nodes (static): {len(static.get('2d_node', []))} nodes")
            print(f"  Timesteps: {len(dynamic.get('timesteps', []))}")

        # Attach timestamps
        for k in ["1d_node", "2d_node"]:
            if k in dynamic:
                dynamic[k] = attach_timestamps(dynamic[k], dynamic["timesteps"])

        samples_before = len(self.X)

        # Process 1D nodes
        if self.node_type_filter is None or 0 in self.node_type_filter:
            self._process_nodes(
                node_type=0,
                node_dyn=dynamic.get("1d_node"),
                node_static=static.get("1d_node"),
                event_idx=event_idx,
            )

            self.samples_after_1d = len(self.X)
            if self.verbose:
                print(
                    f"\n1D nodes processed: {self.samples_after_1d - samples_before} samples created"
                )

        # Process 2D nodes
        if self.node_type_filter is None or 1 in self.node_type_filter:
            self._process_nodes(
                node_type=1,
                node_dyn=dynamic.get("2d_node"),
                node_static=static.get("2d_node"),
                event_idx=event_idx,
            )

            self.samples_after_2d = len(self.X)
            if self.verbose:
                print(
                    f"2D nodes processed: {self.samples_after_2d - self.samples_after_1d} samples created"
                )
                print(
                    f"Total samples from this event: {self.samples_after_2d - samples_before}"
                )

    def _process_nodes(self, node_type, node_dyn, node_static, event_idx):
        """Process nodes WITHOUT edge aggregation - much faster!"""
        if node_static is None or node_dyn is None:
            return

        node_type_str = "1D" if node_type == 0 else "2D"
        if self.verbose:
            print(f"\nProcessing {node_type_str} nodes...")
            print(f"  Static columns: {list(node_static.columns)}")
            print(f"  Dynamic columns: {list(node_dyn.columns)}")

        nodes_processed = 0
        for node_id in node_static["node_idx"].values:
            node_data = node_dyn[node_dyn["node_idx"] == node_id].copy()

            if len(node_data) == 0:
                continue

            # Merge with static features - that's it! No edge aggregation
            df = (
                node_data.merge(node_static, on="node_idx", how="left")
                .sort_values("timestep_idx")
                .reset_index(drop=True)
            )

            if self.verbose and self.debug and nodes_processed == 0:
                print(f"\n  Example node {node_id} (first node):")
                print(f"    Timesteps: {len(df)}")
                print(f"    Columns: {list(df.columns)}")

            df = df.fillna(0.0)

            samples_before = len(self.X)
            self._make_samples_from_df(df, node_type, node_id, event_idx)
            samples_created = len(self.X) - samples_before

            if samples_created > 0:
                nodes_processed += 1

        if self.verbose:
            print(
                f"  {node_type_str} nodes with samples: {nodes_processed}/{len(node_static)}"
            )

    def _make_samples_from_df(self, df, node_type, node_id, event_idx):
        """Create sliding window samples with feature padding to uniform size."""
        if len(df) < self.window + 1:
            return

        if "water_level" not in df.columns:
            return

        for t in range(self.window, len(df) - 1):
            # Create padded array with ALL feature slots
            window_data = np.zeros((self.window, len(FEATURE_NAMES)), dtype=np.float32)

            # Fill in available features at correct positions
            for feat_idx, feat_name in enumerate(FEATURE_NAMES):
                if feat_name in df.columns:
                    window_data[:, feat_idx] = df.iloc[t - self.window : t][
                        feat_name
                    ].values

            X_seq = window_data
            X_out = X_seq if self.return_sequence else X_seq.flatten()
            y_out = df.iloc[t + 1]["water_level"]

            self.X.append(X_out)
            self.y.append(y_out)
            self.y_raw.append(y_out)
            self.node_type.append(node_type)

            # Track which features are actually present
            available_features = [f for f in FEATURE_NAMES if f in df.columns]
            self.feature_tracking["by_sample"].append(available_features)
            self.feature_tracking["by_node_type"][node_type].append(available_features)
            self.feature_tracking["by_event"][event_idx].append(available_features)

            # Two-head architecture features
            if self.separate_static_dynamic:
                static_data = np.zeros(
                    (self.window, len(STATIC_FEATURES)), dtype=np.float32
                )
                dynamic_data = np.zeros(
                    (self.window, len(DYNAMIC_FEATURES)), dtype=np.float32
                )

                for feat_idx, feat_name in enumerate(STATIC_FEATURES):
                    if feat_name in df.columns:
                        static_data[:, feat_idx] = df.iloc[t - self.window : t][
                            feat_name
                        ].values

                for feat_idx, feat_name in enumerate(DYNAMIC_FEATURES):
                    if feat_name in df.columns:
                        dynamic_data[:, feat_idx] = df.iloc[t - self.window : t][
                            feat_name
                        ].values

                self.X_static.append(
                    static_data if self.return_sequence else static_data.flatten()
                )
                self.X_dynamic.append(
                    dynamic_data if self.return_sequence else dynamic_data.flatten()
                )

            if self.debug:
                debug_info = {
                    "node_id": node_id,
                    "node_type": node_type,
                    "event_idx": event_idx,
                    "timestep": df.iloc[t]["timestep_idx"],
                    "X_seq": X_seq,
                    "y": y_out,
                    "available_features": available_features,
                    "num_features": len(available_features),
                    "num_padded_features": len(FEATURE_NAMES) - len(available_features),
                    "feature_values": {
                        feat: df.iloc[t][feat] for feat in available_features
                    },
                }

                if self.separate_static_dynamic:
                    debug_info["X_static_seq"] = static_data
                    debug_info["X_dynamic_seq"] = dynamic_data

                self.debug_samples.append(debug_info)

                if len(self.X) == 1 and self.verbose:
                    print("\n  FIRST SAMPLE DETAILS:")
                    print(
                        f"    Node ID: {node_id}, Node type: {node_type}, Event: {event_idx}"
                    )
                    print(f"    Window: timesteps {t - self.window} to {t}")
                    print(f"    Target: timestep {t + 1}, water_level = {y_out:.4f}")
                    print(f"    Total feature slots: {len(FEATURE_NAMES)}")
                    print(f"    Available features: {len(available_features)}")
                    print(f"    Input shape: {X_seq.shape}")
                    print(f"    Features present: {available_features}")

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        if self.separate_static_dynamic:
            return (
                (self.X_static[idx], self.X_dynamic[idx]),
                self.y[idx],
                self.node_type[idx],
            )
        return self.X[idx], self.y[idx], self.node_type[idx]

    def get_feature_names(self):
        """Return the list of features used in this dataset."""
        if self.separate_static_dynamic:
            return {"static": STATIC_FEATURES, "dynamic": DYNAMIC_FEATURES}
        return FEATURE_NAMES

    def get_stats(self):
        """Get detailed dataset statistics."""
        stats = {
            "total_samples": len(self),
            "1d_samples": (self.node_type == 0).sum().item(),
            "2d_samples": (self.node_type == 1).sum().item(),
            "window_size": self.window,
            "feature_dim": len(FEATURE_NAMES),
        }

        if self.separate_static_dynamic:
            stats["static_feature_dim"] = len(STATIC_FEATURES)
            stats["dynamic_feature_dim"] = len(DYNAMIC_FEATURES)
            stats["static_shape"] = self.X_static[0].shape if len(self) > 0 else None
            stats["dynamic_shape"] = self.X_dynamic[0].shape if len(self) > 0 else None
        else:
            stats["input_shape"] = self.X[0].shape if len(self) > 0 else None

        if self.debug and len(self.debug_samples) > 0:
            sample = self.debug_samples[0]
            stats["actual_features"] = sample["available_features"]
            stats["num_actual_features"] = len(sample["available_features"])

        return stats


class WaterLevelDataset1D(Dataset):
    """
    Dataset for 1D nodes only - simplified for node-level features.
    
    Since we're using node-level features only, 1D nodes will have:
    - 1d_position_x, 1d_position_y (static)
    - inlet_flow (dynamic)
    All other features will be padded with zeros.
    """

    def __init__(
        self, event_dirs, window, normalizer=None, fit_normalizer=False, **kwargs
    ):
        # Filter to only include 1D node types during processing
        kwargs["node_type_filter"] = [0]  # Only 1D nodes

        self.dataset = JointWaterLevelDataset(
            event_dirs, window, normalizer, fit_normalizer, **kwargs
        )

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        X, y, node_type = self.dataset[idx]
        return X, y  # Don't need node_type since all are 1D


class WaterLevelDataset2D(Dataset):
    """
    Dataset for 2D nodes only - simplified for node-level features.
    
    Since we're using node-level features only, 2D nodes will have:
    - 2d_position_x, 2d_position_y, area, roughness, elevation, 
      aspect, curvature, flow_accumulation, slope (static)
    - rainfall, water_volume (dynamic)
    1D features will be padded with zeros.
    """

    def __init__(
        self, event_dirs, window, normalizer=None, fit_normalizer=False, **kwargs
    ):
        # Filter to only include 2D node types during processing
        kwargs["node_type_filter"] = [1]  # Only 2D nodes

        self.dataset = JointWaterLevelDataset(
            event_dirs, window, normalizer, fit_normalizer, **kwargs
        )

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        X, y, node_type = self.dataset[idx]
        return X, y


class CombinedDataset(Dataset):
    """
    Combines 1D and 2D datasets with consistent feature dimensions.
    
    Since both datasets already use the same FEATURE_NAMES list with padding,
    they should already have the same dimensions. This class just concatenates them.
    """
    
    def __init__(self, dataset_1d, dataset_2d):
        self.dataset_1d = dataset_1d
        self.dataset_2d = dataset_2d

        # Verify both datasets have same feature dimension
        dim_1d = dataset_1d.dataset.X.shape[-1]
        dim_2d = dataset_2d.dataset.X.shape[-1]
        
        if dim_1d != dim_2d:
            raise ValueError(
                f"Feature dimension mismatch! 1D: {dim_1d}, 2D: {dim_2d}. "
                f"Both should be {len(FEATURE_NAMES)} after padding."
            )
        
        self.feature_dim = dim_1d
        print(f"✓ CombinedDataset: Both 1D and 2D have {self.feature_dim} features")

    def __len__(self):
        return len(self.dataset_1d) + len(self.dataset_2d)

    def __getitem__(self, idx):
        if idx < len(self.dataset_1d):
            # 1D sample
            X, y = self.dataset_1d[idx]
            return X, y, torch.tensor(0)  # node_type=0
        else:
            # 2D sample
            idx_2d = idx - len(self.dataset_1d)
            X, y = self.dataset_2d[idx_2d]
            return X, y, torch.tensor(1)  # node_type=1