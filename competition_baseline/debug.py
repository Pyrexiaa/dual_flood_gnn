import numpy as np
import pandas as pd
from torch.utils.data import DataLoader
from pathlib import Path
import json


def debug_combined_dataset(
    train_combined,
    test_combined,
    normalizer_1d,
    normalizer_2d,
    num_batches=3,
    batch_size=16,
    output_dir="./debug_two_head",
):
    """
    Debug the CombinedDataset for two-head architecture with SEPARATE normalizers.

    Args:
        train_combined: CombinedDataset for training
        test_combined: CombinedDataset for testing
        normalizer_1d: Fitted normalizer for 1D data
        normalizer_2d: Fitted normalizer for 2D data
        num_batches: Number of batches to save and analyze
        batch_size: Batch size for DataLoader
        output_dir: Directory to save debug outputs
    """
    print("\n" + "=" * 80)
    print("DEBUGGING TWO-HEAD COMBINED DATASET (NODE-LEVEL FEATURES)")
    print("=" * 80)

    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True, parents=True)

    # =====================
    # DATASET STATISTICS
    # =====================
    print("\n" + "-" * 80)
    print("DATASET STATISTICS")
    print("-" * 80)

    print("\nTraining Set:")
    print(f"  Total samples: {len(train_combined)}")
    print(f"  1D samples: {len(train_combined.dataset_1d)}")
    print(f"  2D samples: {len(train_combined.dataset_2d)}")

    print("\nTest Set:")
    print(f"  Total samples: {len(test_combined)}")
    print(f"  1D samples: {len(test_combined.dataset_1d)}")
    print(f"  2D samples: {len(test_combined.dataset_2d)}")

    # Get feature dimensions
    X_1d_sample, y_1d_sample = train_combined.dataset_1d[0]
    X_2d_sample, y_2d_sample = train_combined.dataset_2d[0]

    print("\nFeature Dimensions:")
    print(f"  1D input shape: {X_1d_sample.shape}")
    print(f"  2D input shape: {X_2d_sample.shape}")
    print("  Both should be: (window_size, 15) with node-level features only")

    # =====================
    # SAMPLE INSPECTION
    # =====================
    print("\n" + "-" * 80)
    print("SAMPLE INSPECTION")
    print("-" * 80)

    # Check first sample from each type
    X_1d, y_1d, node_type_1d = train_combined[0]  # First sample (1D)

    # Find first 2D sample
    idx_2d = len(train_combined.dataset_1d)
    X_2d, y_2d, node_type_2d = train_combined[idx_2d]

    print("\n1D Sample (first 1D sample):")
    print(f"  Input shape: {X_1d.shape}")
    print(f"  Target: {y_1d.item():.6f}")
    print(f"  Node type: {node_type_1d.item()}")
    print(
        f"  Input stats - Min: {X_1d.min():.4f}, Max: {X_1d.max():.4f}, Mean: {X_1d.mean():.4f}"
    )

    print("\n2D Sample (first 2D sample):")
    print(f"  Input shape: {X_2d.shape}")
    print(f"  Target: {y_2d.item():.6f}")
    print(f"  Node type: {node_type_2d.item()}")
    print(
        f"  Input stats - Min: {X_2d.min():.4f}, Max: {X_2d.max():.4f}, Mean: {X_2d.mean():.4f}"
    )

    # =====================
    # NORMALIZATION CHECK
    # =====================
    print("\n" + "-" * 80)
    print("NORMALIZATION CHECK (SEPARATE NORMALIZERS)")
    print("-" * 80)

    # Get raw targets if available
    if hasattr(train_combined.dataset_1d.dataset, "y_raw"):
        y_1d_raw = train_combined.dataset_1d.dataset.y_raw[0].item()
        print("\n1D Target (using 1D normalizer):")
        print(f"  Raw: {y_1d_raw:.6f}")
        print(f"  Normalized: {y_1d.item():.6f}")

        if hasattr(normalizer_1d, "inverse_transform_y"):
            y_1d_recovered = normalizer_1d.inverse_transform_y(
                np.array([[y_1d.item()]])
            )[0][0]
            print(f"  Recovered: {y_1d_recovered:.6f}")
            print(f"  Match: {np.isclose(y_1d_raw, y_1d_recovered)}")

    if hasattr(train_combined.dataset_2d.dataset, "y_raw"):
        y_2d_raw = train_combined.dataset_2d.dataset.y_raw[0].item()
        print("\n2D Target (using 2D normalizer):")
        print(f"  Raw: {y_2d_raw:.6f}")
        print(f"  Normalized: {y_2d.item():.6f}")

        if hasattr(normalizer_2d, "inverse_transform_y"):
            y_2d_recovered = normalizer_2d.inverse_transform_y(
                np.array([[y_2d.item()]])
            )[0][0]
            print(f"  Recovered: {y_2d_recovered:.6f}")
            print(f"  Match: {np.isclose(y_2d_raw, y_2d_recovered)}")

    # =====================
    # BATCH PROCESSING
    # =====================
    print("\n" + "=" * 80)
    print(f"PROCESSING {num_batches} BATCHES")
    print("=" * 80)

    train_loader = DataLoader(train_combined, batch_size=batch_size, shuffle=True)

    batch_stats = []

    for batch_idx, (X_batch, y_batch, node_type_batch) in enumerate(train_loader):
        if batch_idx >= num_batches:
            break

        print(f"\n{'─' * 80}")
        print(f"BATCH {batch_idx + 1}/{num_batches}")
        print(f"{'─' * 80}")

        # Batch statistics
        mask_1d = node_type_batch == 0
        mask_2d = node_type_batch == 1
        n_1d = mask_1d.sum().item()
        n_2d = mask_2d.sum().item()

        print("\nBatch Composition:")
        print(f"  Total samples: {len(X_batch)}")
        print(f"  1D samples: {n_1d}")
        print(f"  2D samples: {n_2d}")
        print(f"  Input shape: {X_batch.shape}")
        print(f"  Target shape: {y_batch.shape}")

        # Input statistics
        print("\nInput Statistics:")
        print(f"  Min: {X_batch.min():.4f}")
        print(f"  Max: {X_batch.max():.4f}")
        print(f"  Mean: {X_batch.mean():.4f}")
        print(f"  Std: {X_batch.std():.4f}")

        if n_1d > 0:
            X_1d_batch = X_batch[mask_1d]
            y_1d_batch = y_batch[mask_1d]
            print("\n1D Samples in Batch:")
            print(
                f"  Input - Min: {X_1d_batch.min():.4f}, Max: {X_1d_batch.max():.4f}, Mean: {X_1d_batch.mean():.4f}"
            )
            print(
                f"  Target - Min: {y_1d_batch.min():.4f}, Max: {y_1d_batch.max():.4f}, Mean: {y_1d_batch.mean():.4f}"
            )

        if n_2d > 0:
            X_2d_batch = X_batch[mask_2d]
            y_2d_batch = y_batch[mask_2d]
            print("\n2D Samples in Batch:")
            print(
                f"  Input - Min: {X_2d_batch.min():.4f}, Max: {X_2d_batch.max():.4f}, Mean: {X_2d_batch.mean():.4f}"
            )
            print(
                f"  Target - Min: {y_2d_batch.min():.4f}, Max: {y_2d_batch.max():.4f}, Mean: {y_2d_batch.mean():.4f}"
            )

        # Save batch to CSV
        batch_df = save_batch_to_csv(
            X_batch,
            y_batch,
            node_type_batch,
            batch_idx,
            output_path,
            normalizer_1d,
            normalizer_2d,
        )

        # Collect stats
        stats = {
            "batch_idx": batch_idx,
            "total_samples": len(X_batch),
            "n_1d": n_1d,
            "n_2d": n_2d,
            "input_min": X_batch.min().item(),
            "input_max": X_batch.max().item(),
            "input_mean": X_batch.mean().item(),
            "input_std": X_batch.std().item(),
            "target_min": y_batch.min().item(),
            "target_max": y_batch.max().item(),
            "target_mean": y_batch.mean().item(),
            "target_std": y_batch.std().item(),
        }
        batch_stats.append(stats)

    # =====================
    # SAVE SUMMARY
    # =====================
    print("\n" + "=" * 80)
    print("SAVING SUMMARY")
    print("=" * 80)

    summary = {
        "train_samples": len(train_combined),
        "train_1d": len(train_combined.dataset_1d),
        "train_2d": len(train_combined.dataset_2d),
        "test_samples": len(test_combined),
        "test_1d": len(test_combined.dataset_1d),
        "test_2d": len(test_combined.dataset_2d),
        "input_dim": X_1d_sample.shape[-1],  # Should be same for both
        "window_size": X_1d_sample.shape[0],
        "batch_size": batch_size,
        "num_batches_analyzed": len(batch_stats),
        "batch_statistics": batch_stats,
        "note": "Using separate normalizers for 1D and 2D data",
    }

    # Save JSON summary
    summary_path = output_path / "debug_summary.json"
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\n✓ Saved summary to: {summary_path}")

    # Save batch stats as CSV
    stats_df = pd.DataFrame(batch_stats)
    stats_path = output_path / "batch_statistics.csv"
    stats_df.to_csv(stats_path, index=False)
    print(f"✓ Saved batch statistics to: {stats_path}")

    print("\n" + "=" * 80)
    print("DEBUG COMPLETE")
    print("=" * 80 + "\n")

    return summary, stats_df


def save_batch_to_csv(
    X_batch,
    y_batch,
    node_type_batch,
    batch_idx,
    output_path,
    normalizer_1d,
    normalizer_2d,
):
    """
    Save a single batch to CSV with SEPARATE normalizers for 1D/2D.

    Args:
        X_batch: Input tensor (batch_size, window, features)
        y_batch: Target tensor (batch_size, 1)
        node_type_batch: Node type tensor (batch_size,)
        batch_idx: Batch index
        output_path: Output directory path
        normalizer_1d: Normalizer for 1D data
        normalizer_2d: Normalizer for 2D data

    Returns:
        DataFrame with batch data
    """
    X_np = X_batch.numpy()
    y_np = y_batch.numpy()
    node_type_np = node_type_batch.numpy()

    B, W, F = X_np.shape

    rows = []
    for i in range(B):
        node_type_val = int(node_type_np[i])

        # Select correct normalizer
        normalizer = normalizer_1d if node_type_val == 0 else normalizer_2d

        # Get original scale targets if possible
        if hasattr(normalizer, "inverse_transform_y"):
            y_original = normalizer.inverse_transform_y(y_np[i : i + 1])[0, 0]
        elif hasattr(normalizer, "inverse_y"):
            y_original = normalizer.inverse_y(y_np[i : i + 1])[0, 0]
        else:
            y_original = y_np[i, 0]

        row = {
            "sample_idx": i,
            "node_type": node_type_val,
            "target_normalized": float(y_np[i, 0]),
            "target_original": float(y_original),
        }

        # Add features for each timestep
        for t in range(W):
            for f in range(F):
                row[f"timestep_{t}_feat_{f}"] = float(X_np[i, t, f])

        # Add aggregated features
        for f in range(F):
            row[f"feat_{f}_mean"] = float(X_np[i, :, f].mean())
            row[f"feat_{f}_std"] = float(X_np[i, :, f].std())
            row[f"feat_{f}_min"] = float(X_np[i, :, f].min())
            row[f"feat_{f}_max"] = float(X_np[i, :, f].max())
            row[f"feat_{f}_last"] = float(X_np[i, -1, f])

        rows.append(row)

    df = pd.DataFrame(rows)

    # Save combined batch
    combined_path = output_path / f"batch_{batch_idx:03d}_combined.csv"
    df.to_csv(combined_path, index=False)
    print(f"  ✓ Saved batch {batch_idx} to: {combined_path}")

    # Save separate 1D and 2D files
    df_1d = df[df["node_type"] == 0]
    df_2d = df[df["node_type"] == 1]

    if len(df_1d) > 0:
        path_1d = output_path / f"batch_{batch_idx:03d}_1d.csv"
        df_1d.to_csv(path_1d, index=False)
        print(f"    → 1D samples: {len(df_1d)} saved to {path_1d.name}")

    if len(df_2d) > 0:
        path_2d = output_path / f"batch_{batch_idx:03d}_2d.csv"
        df_2d.to_csv(path_2d, index=False)
        print(f"    → 2D samples: {len(df_2d)} saved to {path_2d.name}")

    return df


def evaluate_batches(output_dir="./debug_two_head", num_batches=3):
    """
    Evaluate the saved batches and generate analysis.

    Args:
        output_dir: Directory containing saved batches
        num_batches: Number of batches to evaluate

    Returns:
        df_1d_all: All 1D samples
        df_2d_all: All 2D samples
    """
    print("\n" + "=" * 80)
    print("EVALUATING SAVED BATCHES")
    print("=" * 80)

    output_path = Path(output_dir)

    all_1d_samples = []
    all_2d_samples = []

    for batch_idx in range(num_batches):
        combined_path = output_path / f"batch_{batch_idx:03d}_combined.csv"

        if not combined_path.exists():
            print(f"Warning: Batch {batch_idx} not found at {combined_path}")
            continue

        df = pd.read_csv(combined_path)

        print(f"\nBatch {batch_idx}:")
        print(f"  Total samples: {len(df)}")
        print(f"  1D samples: {(df['node_type'] == 0).sum()}")
        print(f"  2D samples: {(df['node_type'] == 1).sum()}")

        # Collect samples by type
        all_1d_samples.append(df[df["node_type"] == 0])
        all_2d_samples.append(df[df["node_type"] == 1])

    # Combine all batches
    df_1d_all = (
        pd.concat(all_1d_samples, ignore_index=True)
        if all_1d_samples
        else pd.DataFrame()
    )
    df_2d_all = (
        pd.concat(all_2d_samples, ignore_index=True)
        if all_2d_samples
        else pd.DataFrame()
    )

    print("\n" + "-" * 80)
    print("AGGREGATE STATISTICS")
    print("-" * 80)

    print(f"\nTotal 1D samples across {num_batches} batches: {len(df_1d_all)}")
    if len(df_1d_all) > 0:
        print(
            f"  Target (normalized) - Mean: {df_1d_all['target_normalized'].mean():.4f}, "
            f"Std: {df_1d_all['target_normalized'].std():.4f}"
        )
        print(
            f"  Target (original) - Mean: {df_1d_all['target_original'].mean():.4f}, "
            f"Std: {df_1d_all['target_original'].std():.4f}, "
            f"Min: {df_1d_all['target_original'].min():.4f}, "
            f"Max: {df_1d_all['target_original'].max():.4f}"
        )

    print(f"\nTotal 2D samples across {num_batches} batches: {len(df_2d_all)}")
    if len(df_2d_all) > 0:
        print(
            f"  Target (normalized) - Mean: {df_2d_all['target_normalized'].mean():.4f}, "
            f"Std: {df_2d_all['target_normalized'].std():.4f}"
        )
        print(
            f"  Target (original) - Mean: {df_2d_all['target_original'].mean():.4f}, "
            f"Std: {df_2d_all['target_original'].std():.4f}, "
            f"Min: {df_2d_all['target_original'].min():.4f}, "
            f"Max: {df_2d_all['target_original'].max():.4f}"
        )

    # Check for anomalies
    print("\n" + "-" * 80)
    print("ANOMALY DETECTION")
    print("-" * 80)

    anomalies = []

    if len(df_1d_all) > 0:
        # Check for NaN or Inf
        if df_1d_all.isnull().any().any():
            anomalies.append("✗ Found NaN values in 1D samples")
        if np.isinf(df_1d_all.select_dtypes(include=[np.number]).values).any():
            anomalies.append("✗ Found Inf values in 1D samples")

    if len(df_2d_all) > 0:
        # Check for NaN or Inf
        if df_2d_all.isnull().any().any():
            anomalies.append("✗ Found NaN values in 2D samples")
        if np.isinf(df_2d_all.select_dtypes(include=[np.number]).values).any():
            anomalies.append("✗ Found Inf values in 2D samples")

    if anomalies:
        print("\nAnomalies detected:")
        for anomaly in anomalies:
            print(f"  {anomaly}")
    else:
        print("\n✓ No anomalies detected (no NaN or Inf values)")

    print("\n" + "=" * 80)
    print("EVALUATION COMPLETE")
    print("=" * 80 + "\n")

    return df_1d_all, df_2d_all


def full_debug_workflow(train_combined, test_combined, normalizer_1d, normalizer_2d):
    """
    Complete workflow for debugging the two-head dataset with SEPARATE normalizers.

    Args:
        train_combined: CombinedDataset for training
        test_combined: CombinedDataset for testing
        normalizer_1d: Fitted normalizer for 1D data
        normalizer_2d: Fitted normalizer for 2D data

    Returns:
        summary: Debug summary dict
        stats_df: Batch statistics DataFrame
        df_1d_all: All 1D samples from batches
        df_2d_all: All 2D samples from batches
    """
    print("\n" + "=" * 80)
    print("FULL DEBUG WORKFLOW - NODE-LEVEL FEATURES ONLY")
    print("=" * 80)

    # 1. Debug the combined datasets
    print("\nStep 1: Debugging combined datasets...")
    summary, stats_df = debug_combined_dataset(
        train_combined=train_combined,
        test_combined=test_combined,
        normalizer_1d=normalizer_1d,
        normalizer_2d=normalizer_2d,
        num_batches=3,
        batch_size=16,
        output_dir="./debug_two_head",
    )

    # 2. Evaluate the saved batches
    print("\nStep 2: Evaluating saved batches...")
    df_1d_all, df_2d_all = evaluate_batches(
        output_dir="./debug_two_head",
        num_batches=3,
    )

    print("\n" + "=" * 80)
    print("✓ FULL DEBUG WORKFLOW COMPLETE!")
    print("=" * 80)
    print("\nOutput files saved to: ./debug_two_head/")
    print("  - debug_summary.json")
    print("  - batch_statistics.csv")
    print("  - batch_XXX_combined.csv (all samples)")
    print("  - batch_XXX_1d.csv (1D samples only)")
    print("  - batch_XXX_2d.csv (2D samples only)")
    print("\nKey insights:")
    print(f"  - Total 1D samples analyzed: {len(df_1d_all)}")
    print(f"  - Total 2D samples analyzed: {len(df_2d_all)}")
    print(f"  - Feature dimension: {summary['input_dim']} (node-level only)")
    print(f"  - Window size: {summary['window_size']}")
    print("=" * 80 + "\n")

    return summary, stats_df, df_1d_all, df_2d_all
