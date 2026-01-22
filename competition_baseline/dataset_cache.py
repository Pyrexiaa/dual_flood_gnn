from dataset import JointWaterLevelDataset
import torch
import numpy as np
import h5py
from pathlib import Path
import hashlib
import json
from datetime import datetime
import argparse
import joblib
import shutil


class DatasetCache:
    """
    Cache system for preprocessed datasets.

    Saves preprocessed data to disk so you only process once.
    """

    def __init__(self, cache_dir="./preprocessed_cache"):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(exist_ok=True)

    def _get_cache_key(self, event_dirs, window, config):
        """Generate unique cache key based on configuration."""
        # Convert event_dirs to sorted list of paths
        if isinstance(event_dirs, (str, Path)):
            events = sorted(Path(event_dirs).iterdir())
        else:
            events = sorted([Path(d) for d in event_dirs])

        # Create hashable string from config
        config_str = f"{window}_{config.get('return_sequence', False)}_{config.get('separate_static_dynamic', False)}"
        events_str = "_".join([e.name for e in events])

        # Hash for unique filename
        key = hashlib.md5(f"{config_str}_{events_str}".encode()).hexdigest()
        return key

    def get_cache_path(self, event_dirs, window, config, prefix=""):
        """Get path to cache file."""
        key = self._get_cache_key(event_dirs, window, config)
        filename = f"{prefix}dataset_{key}.h5"
        return self.cache_dir / filename

    def exists(self, event_dirs, window, config, prefix=""):
        """Check if cached dataset exists."""
        cache_path = self.get_cache_path(event_dirs, window, config, prefix)
        return cache_path.exists()

    def save(self, dataset, event_dirs, window, config, prefix=""):
        """
        Save preprocessed dataset to cache.

        Args:
            dataset: JointWaterLevelDataset instance
            event_dirs: Event directories used
            window: Window size
            config: Dict with configuration
            prefix: Prefix for filename (e.g., 'train', 'val')
        """
        cache_path = self.get_cache_path(event_dirs, window, config, prefix)

        print(f"\n{'=' * 80}")
        print("SAVING PREPROCESSED DATASET TO CACHE")
        print(f"{'=' * 80}")
        print(f"Cache path: {cache_path}")

        with h5py.File(cache_path, "w") as f:
            # Save data arrays
            f.create_dataset("X", data=dataset.X.cpu().numpy(), compression="gzip")
            f.create_dataset("y", data=dataset.y.cpu().numpy(), compression="gzip")
            f.create_dataset(
                "y_raw", data=dataset.y_raw.cpu().numpy(), compression="gzip"
            )
            f.create_dataset(
                "node_type", data=dataset.node_type.cpu().numpy(), compression="gzip"
            )

            # Save static/dynamic if separate
            if hasattr(dataset, "X_static") and dataset.X_static is not None:
                f.create_dataset(
                    "X_static", data=dataset.X_static.cpu().numpy(), compression="gzip"
                )
                f.create_dataset(
                    "X_dynamic",
                    data=dataset.X_dynamic.cpu().numpy(),
                    compression="gzip",
                )

            # Save metadata
            f.attrs["window"] = window
            f.attrs["return_sequence"] = config.get("return_sequence", False)
            f.attrs["separate_static_dynamic"] = config.get(
                "separate_static_dynamic", False
            )
            f.attrs["num_samples"] = len(dataset)
            f.attrs["created_at"] = datetime.now().isoformat()
            f.attrs["feature_names"] = json.dumps(
                dataset.get_feature_names()
                if hasattr(dataset, "get_feature_names")
                else []
            )

            # Save debug samples if available (first 100 only to save space)
            if hasattr(dataset, "debug_samples") and dataset.debug_samples:
                debug_data = []
                for sample in dataset.debug_samples[:100]:
                    # Convert numpy arrays to lists for JSON
                    sample_copy = {}
                    for key, val in sample.items():
                        if isinstance(val, np.ndarray):
                            sample_copy[key] = val.tolist()
                        elif isinstance(
                            val, (list, dict, str, int, float, bool, type(None))
                        ):
                            sample_copy[key] = val
                        else:
                            sample_copy[key] = str(val)
                    debug_data.append(sample_copy)
                f.attrs["debug_samples"] = json.dumps(debug_data)

        file_size = cache_path.stat().st_size / (1024**3)  # GB
        print(f"✓ Saved {len(dataset)} samples ({file_size:.2f} GB)")
        print(f"{'=' * 80}\n")

    def load(self, event_dirs, window, config, prefix="", device="cpu"):
        """
        Load preprocessed dataset from cache.

        Returns:
            CachedDataset instance
        """
        cache_path = self.get_cache_path(event_dirs, window, config, prefix)

        if not cache_path.exists():
            raise FileNotFoundError(f"Cache file not found: {cache_path}")

        print(f"\n{'=' * 80}")
        print("LOADING PREPROCESSED DATASET FROM CACHE")
        print(f"{'=' * 80}")
        print(f"Cache path: {cache_path}")

        dataset = CachedDataset(cache_path, device)

        print(f"✓ Loaded {len(dataset)} samples")
        print(f"{'=' * 80}\n")

        return dataset

    def load_or_create(
        self,
        event_dirs,
        window,
        config,
        prefix="",
        device="cpu",
        normalizer=None,
        force_recreate=False,
    ):
        """
        Load from cache if exists, otherwise create and cache.

        Args:
            event_dirs: Path to event directories
            window: Window size
            config: Configuration dict
            prefix: Filename prefix
            device: Device to load tensors on
            normalizer: Normalizer instance (for creation only)
            force_recreate: Force recreation even if cache exists

        Returns:
            dataset, normalizer
        """
        cache_exists = self.exists(event_dirs, window, config, prefix)
        normalizer_path = self.cache_dir / f"{prefix}_normalizer.pkl"

        if cache_exists and not force_recreate:
            # Load from cache
            dataset = self.load(event_dirs, window, config, prefix, device)

            # Load normalizer if exists
            if normalizer_path.exists():
                normalizer = joblib.load(normalizer_path)
                print(f"✓ Loaded normalizer from {normalizer_path}")

            return dataset, normalizer

        else:
            # Create new dataset
            print(f"\n{'=' * 80}")
            print("CREATING NEW DATASET (not in cache)")
            print(f"{'=' * 80}\n")

            dataset = JointWaterLevelDataset(
                event_dirs=event_dirs,
                window=window,
                normalizer=normalizer,
                fit_normalizer=(prefix == "train"),
                return_sequence=config.get("return_sequence", False),
                debug=config.get("debug", False),
                max_events=config.get("max_events", None),
                max_samples=config.get("max_samples", None),
                verbose=config.get("verbose", True),
                separate_static_dynamic=config.get("separate_static_dynamic", False),
            )

            # Save to cache
            self.save(dataset, event_dirs, window, config, prefix)

            # Save normalizer if fitted
            if prefix == "train" and Path("train_normalizer.pkl").exists():
                shutil.copy("train_normalizer.pkl", normalizer_path)
                print(f"✓ Saved normalizer to {normalizer_path}")

            return dataset, normalizer


class CachedDataset(torch.utils.data.Dataset):
    """
    Lightweight dataset wrapper for cached data.

    Loads preprocessed data from HDF5 file.
    """

    def __init__(self, cache_path, device="cpu"):
        self.cache_path = Path(cache_path)
        self.device = device

        with h5py.File(cache_path, "r") as f:
            # Load all data into memory (faster than lazy loading)
            self.X = torch.from_numpy(f["X"][:]).to(device)
            self.y = torch.from_numpy(f["y"][:]).to(device)
            self.y_raw = torch.from_numpy(f["y_raw"][:]).to(device)
            self.node_type = torch.from_numpy(f["node_type"][:]).to(device)

            # Load metadata
            self.window = f.attrs["window"]
            self.return_sequence = f.attrs["return_sequence"]
            self.separate_static_dynamic = f.attrs.get("separate_static_dynamic", False)

            # Load static/dynamic if separate
            if self.separate_static_dynamic and "X_static" in f:
                self.X_static = torch.from_numpy(f["X_static"][:]).to(device)
                self.X_dynamic = torch.from_numpy(f["X_dynamic"][:]).to(device)
            else:
                self.X_static = None
                self.X_dynamic = None

            # Load feature names
            if "feature_names" in f.attrs:
                self.feature_names = json.loads(f.attrs["feature_names"])
            else:
                self.feature_names = None

            # Load debug samples
            if "debug_samples" in f.attrs:
                self.debug_samples = json.loads(f.attrs["debug_samples"])
            else:
                self.debug_samples = None

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

    def get_stats(self):
        """Get dataset statistics."""
        stats = {
            "total_samples": len(self),
            "1d_samples": (self.node_type == 0).sum().item(),
            "2d_samples": (self.node_type == 1).sum().item(),
            "window_size": self.window,
            "input_shape": self.X[0].shape if len(self) > 0 else None,
        }

        if self.separate_static_dynamic:
            stats["static_shape"] = self.X_static[0].shape if len(self) > 0 else None
            stats["dynamic_shape"] = self.X_dynamic[0].shape if len(self) > 0 else None

        return stats


###############################################################################
# USAGE EXAMPLES
###############################################################################


def preprocess_all_data(train_events, test_events, normalizer):
    """
    Preprocess all data once and save to cache.

    Run this script once to preprocess everything.
    Then use load_cached_data() for training.
    """
    cache = DatasetCache(cache_dir="./preprocessed_cache")

    config = {
        "return_sequence": True,
        "debug": False,
        "verbose": True,
        "separate_static_dynamic": False,
    }

    print("=" * 80)
    print("PREPROCESSING ALL DATA")
    print("=" * 80)

    # Preprocess training data
    print("\n1. Processing TRAINING data...")
    train_dataset, train_normalizer = cache.load_or_create(
        event_dirs=train_events,
        window=5,
        config=config,
        prefix="train",
        normalizer=normalizer,
        force_recreate=False,  # Set True to rebuild
    )

    # Preprocess test data
    print("\n3. Processing TEST data...")
    test_dataset, _ = cache.load_or_create(
        event_dirs=test_events,
        window=5,
        config=config,
        prefix="test",
        normalizer=train_normalizer,
        force_recreate=False,
    )

    print("\n" + "=" * 80)
    print("PREPROCESSING COMPLETE!")
    print("=" * 80)
    print("\nDataset statistics:")
    print(f"  Training:   {len(train_dataset):,} samples")
    print(f"  Test:       {len(test_dataset):,} samples")
    print(f"\nCached to: {cache.cache_dir}")


def load_cached_data(device="cuda"):
    """
    Load preprocessed data from cache (fast!).

    Use this in your training script.
    """
    cache = DatasetCache(cache_dir="./preprocessed_cache")

    config = {
        "return_sequence": True,
        "debug": False,
        "verbose": True,
        "separate_static_dynamic": False,
    }

    # Load training data
    train_dataset, train_normalizer = cache.load_or_create(
        event_dirs="data/train",
        window=5,
        config=config,
        prefix="train",
        device=device,
    )

    # Load validation data
    val_dataset, _ = cache.load_or_create(
        event_dirs="data/val",
        window=5,
        config=config,
        prefix="val",
        device=device,
        normalizer=train_normalizer,
    )

    return train_dataset, val_dataset, train_normalizer


# def train_with_cached_data():
#     """Complete training example with cached data."""
#     from torch.utils.data import DataLoader

#     # Load cached data (instant!)
#     train_dataset, val_dataset, normalizer = load_cached_data(device='cuda')

#     print(f"Loaded datasets:")
#     print(f"  Train: {len(train_dataset)} samples")
#     print(f"  Val:   {len(val_dataset)} samples")

#     # Create dataloaders
#     train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
#     val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)

#     # Initialize model
#     model = YourModel(input_dim=train_dataset.X.shape[-1])

#     # Train
#     for epoch in range(50):
#         # Training loop
#         for X, y, node_type in train_loader:
#             # X, y, node_type are already on correct device!
#             # ... training code ...
#             pass


def clear_cache(cache_dir="./preprocessed_cache"):
    """Clear all cached datasets."""
    cache_path = Path(cache_dir)
    if cache_path.exists():
        shutil.rmtree(cache_path)
        print(f"✓ Cleared cache: {cache_dir}")
    else:
        print(f"Cache directory does not exist: {cache_dir}")


def list_cache(cache_dir="./preprocessed_cache"):
    """List all cached datasets."""
    cache_path = Path(cache_dir)
    if not cache_path.exists():
        print("No cache directory found")
        return

    print(f"\n{'=' * 80}")
    print(f"CACHED DATASETS in {cache_dir}")
    print(f"{'=' * 80}\n")

    for file in sorted(cache_path.glob("*.h5")):
        size = file.stat().st_size / (1024**3)  # GB

        with h5py.File(file, "r") as f:
            num_samples = f.attrs.get("num_samples", "unknown")
            created = f.attrs.get("created_at", "unknown")
            window = f.attrs.get("window", "unknown")

        print(f"{file.name}")
        print(f"  Samples: {num_samples:,}")
        print(f"  Size: {size:.2f} GB")
        print(f"  Window: {window}")
        print(f"  Created: {created}")
        print()


###############################################################################
# COMMAND LINE INTERFACE
###############################################################################

if __name__ == "__main__":
    train_events = Path("data/Model1/processed/features_csv/train/")
    test_events = Path("data/Model1/processed/features_csv/test/")

    parser = argparse.ArgumentParser(description="Dataset caching utilities")
    parser.add_argument(
        "command",
        choices=["preprocess", "list", "clear", "train"],
        help="Command to execute",
    )
    parser.add_argument(
        "--cache-dir", default="./preprocessed_cache", help="Cache directory"
    )
    parser.add_argument("--device", default="cuda", help="Device for training")

    args = parser.parse_args()

    if args.command == "preprocess":
        preprocess_all_data()

    elif args.command == "list":
        list_cache(args.cache_dir)

    elif args.command == "clear":
        clear_cache(args.cache_dir)


###############################################################################
# RECOMMENDED WORKFLOW
###############################################################################

"""
RECOMMENDED WORKFLOW:

1. PREPROCESS ONCE (slow, ~30-60 min for large datasets):
   
   python dataset_cache.py preprocess
   
   This processes all events and saves to ./preprocessed_cache/

2. TRAIN MANY TIMES (fast, instant loading):
   
   python train.py
   
   Uses load_cached_data() which loads preprocessed data instantly

3. MANAGE CACHE:
   
   # List cached datasets
   python dataset_cache.py list
   
   # Clear cache (to force rebuild)
   python dataset_cache.py clear

BENEFITS:
- Process once, train many times
- 100x faster loading (seconds vs minutes)
- Experiment with different models without reprocessing
- Cached data is compressed (saves disk space)
- Easy to share preprocessed data with team

FILE SIZES (approximate):
- 1M samples: ~2-5 GB compressed
- 10M samples: ~20-50 GB compressed
"""
