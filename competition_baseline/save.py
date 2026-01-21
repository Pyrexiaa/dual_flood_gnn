import pandas as pd
import matplotlib.pyplot as plt
from constants import FEATURE_NAMES, STATIC_FEATURES, DYNAMIC_FEATURES
from torch.utils.data import DataLoader
import torch
from pathlib import Path
import numpy as np


def save_batch_timeseries(
    X_batch, y_batch, node_type, dataset, separate_static_dynamic=False
):
    """
    Save batch timeseries data to dataframe with proper feature handling.

    Args:
        X_batch: Tensor of shape (B, W, F) or tuple of (X_static, X_dynamic)
        y_batch: Tensor of shape (B, 1)
        node_type: Tensor of shape (B,)
        dataset: The JointWaterLevelDataset instance
        separate_static_dynamic: Whether using two-head architecture

    Returns:
        DataFrame with timeseries data
    """
    rows = []

    # Handle two-head architecture
    if separate_static_dynamic:
        X_static, X_dynamic = X_batch
        B, W_s, F_s = X_static.shape
        _, W_d, F_d = X_dynamic.shape

        # Get actual feature names
        if dataset.debug and len(dataset.debug_samples) > 0:
            static_features = dataset.debug_samples[0].get("available_static", [])
            dynamic_features = dataset.debug_samples[0].get("available_dynamic", [])
        else:
            static_features = STATIC_FEATURES[:F_s]
            dynamic_features = DYNAMIC_FEATURES[:F_d]

        print(f"Batch shape - Static: {X_static.shape}, Dynamic: {X_dynamic.shape}")
        print(f"Static features ({len(static_features)}): {static_features}")
        print(f"Dynamic features ({len(dynamic_features)}): {dynamic_features}")

        for sample_idx in range(B):
            for t in range(W_d):  # Use dynamic window (usually same as static)
                row = {
                    "sample_id": sample_idx,
                    "timestep": t,
                    "node_type": int(node_type[sample_idx].item()),
                    "target": y_batch[sample_idx].item(),
                }

                # Add static features (same across all timesteps)
                for f_idx, fname in enumerate(static_features):
                    if t < W_s:  # Ensure we're within static window
                        row[fname] = X_static[sample_idx, t, f_idx].item()

                # Add dynamic features
                for f_idx, fname in enumerate(dynamic_features):
                    row[fname] = X_dynamic[sample_idx, t, f_idx].item()

                rows.append(row)

    else:
        # Single feature tensor
        B, W, F = X_batch.shape

        # Get actual feature names from dataset
        if dataset.debug and len(dataset.debug_samples) > 0:
            actual_features = dataset.debug_samples[0]["available_features"]
        else:
            actual_features = FEATURE_NAMES[:F]

        print(f"Batch shape: {X_batch.shape}")
        print(f"Using {len(actual_features)} features: {actual_features}")

        for sample_idx in range(B):
            for t in range(W):
                row = {
                    "sample_id": sample_idx,
                    "timestep": t,
                    "node_type": int(node_type[sample_idx].item()),
                    "target": y_batch[sample_idx].item(),
                }

                for f_idx, fname in enumerate(actual_features):
                    row[fname] = X_batch[sample_idx, t, f_idx].item()

                rows.append(row)

    df_batch = pd.DataFrame(rows)
    return df_batch


def save_batch_by_node_type(df_batch, output_dir="./debug_output", prefix="batch"):
    """
    Save batch data split by node type.

    Args:
        df_batch: DataFrame with all batch data
        output_dir: Directory to save files
        prefix: Prefix for output files

    Returns:
        dict: Paths to saved files
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True)

    saved_files = {}

    # Split by node type
    df_1d = df_batch[df_batch["node_type"] == 0]
    df_2d = df_batch[df_batch["node_type"] == 1]

    # Save combined
    combined_path = output_dir / f"{prefix}_combined.csv"
    df_batch.to_csv(combined_path, index=False)
    saved_files["combined"] = combined_path
    print(f"✓ Saved combined data: {combined_path} ({len(df_batch)} rows)")

    # Save 1D if exists
    if len(df_1d) > 0:
        path_1d = output_dir / f"{prefix}_1d.csv"
        df_1d.to_csv(path_1d, index=False)
        saved_files["1d"] = path_1d

        num_samples_1d = (
            len(df_1d) // df_batch["timestep"].nunique()
            if "timestep" in df_batch.columns
            else len(df_1d)
        )
        print(
            f"✓ Saved 1D nodes: {path_1d} ({num_samples_1d} samples, {len(df_1d)} rows)"
        )
    else:
        print("⚠ No 1D node data in batch")

    # Save 2D if exists
    if len(df_2d) > 0:
        path_2d = output_dir / f"{prefix}_2d.csv"
        df_2d.to_csv(path_2d, index=False)
        saved_files["2d"] = path_2d

        num_samples_2d = (
            len(df_2d) // df_batch["timestep"].nunique()
            if "timestep" in df_batch.columns
            else len(df_2d)
        )
        print(
            f"✓ Saved 2D nodes: {path_2d} ({num_samples_2d} samples, {len(df_2d)} rows)"
        )
    else:
        print("⚠ No 2D node data in batch")

    return saved_files


def save_multiple_batches(
    loader,
    dataset,
    num_batches=5,
    output_dir="./debug_output",
    separate_static_dynamic=False,
):
    """
    Save multiple batches for comprehensive analysis.

    Args:
        loader: DataLoader instance
        dataset: Dataset instance
        num_batches: Number of batches to save
        output_dir: Output directory
        separate_static_dynamic: Whether using two-head architecture

    Returns:
        list: Paths to all saved files
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True)

    all_saved_files = []

    print(f"\n{'=' * 80}")
    print(f"SAVING {num_batches} BATCHES")
    print(f"{'=' * 80}\n")

    for batch_idx, batch_data in enumerate(loader):
        if batch_idx >= num_batches:
            break

        print(f"\n--- Batch {batch_idx + 1}/{num_batches} ---")

        if separate_static_dynamic:
            (X_static, X_dynamic), y_batch, node_type = batch_data
            X_batch = (X_static, X_dynamic)
        else:
            X_batch, y_batch, node_type = batch_data

        # Convert to DataFrame
        df_batch = save_batch_timeseries(
            X_batch, y_batch, node_type, dataset, separate_static_dynamic
        )

        # Save split by node type
        saved_files = save_batch_by_node_type(
            df_batch, output_dir=output_dir, prefix=f"batch_{batch_idx}"
        )
        all_saved_files.append(saved_files)

    print(f"\n{'=' * 80}")
    print(f"All batches saved to: {output_dir}")
    print(f"{'=' * 80}\n")

    return all_saved_files


def plot_timeseries(sample_id, df_batch):
    sample_df = df_batch[df_batch["sample_id"] == sample_id]

    plt.figure(figsize=(6, 4))
    plt.plot(
        sample_df["timestep"],
        sample_df["water_level"],
        marker="o",
        label="Input water level",
    )

    plt.axhline(
        y=sample_df["target"].iloc[0],
        color="r",
        linestyle="--",
        label="Target (t+1)",
    )

    plt.xlabel("Timestep in window")
    plt.ylabel("Water level")
    plt.legend()
    plt.tight_layout()
    plt.savefig("debug_timeseries_plot.png")
    plt.close()


def plot_1d2d_comparison(df_batch):
    plt.figure(figsize=(6, 4))

    for node_type, label in [(0, "1D"), (1, "2D")]:
        subset = df_batch[df_batch["node_type"] == node_type]
        grouped = subset.groupby("timestep")["water_level"].mean()

        plt.plot(grouped.index, grouped.values, marker="o", label=label)

    plt.xlabel("Timestep in window")
    plt.ylabel("Mean water level")
    plt.legend()
    plt.tight_layout()
    plt.savefig("debug_1d2d_comparison.png")
    plt.close()


def plot_rainfall(df_2d, sample_id):
    df_2d_sample = df_2d[df_2d["sample_id"] == sample_id]

    plt.figure(figsize=(6, 4))
    plt.plot(df_2d_sample["timestep"], df_2d_sample["rainfall"], label="Rainfall")
    plt.plot(
        df_2d_sample["timestep"],
        df_2d_sample["water_level"],
        label="Water level",
    )

    plt.xlabel("Timestep in window")
    plt.legend()
    plt.tight_layout()
    plt.savefig("debug_rainfall_plot.png")
    plt.close()


def save_predictions(
    model,
    dataset,
    normalizer_1d,
    normalizer_2d,
    save_path,
    batch_size=32,
    device="cuda" if torch.cuda.is_available() else "cpu",
):
    """
    Save model predictions with node_id and timestep information.
    """
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    rows = []

    model.eval()

    print("\nGenerating predictions...")
    print(f"  Device: {device}")
    print(f"  Batch size: {batch_size}")

    with torch.no_grad():
        for batch_idx, batch_data in enumerate(loader):
            # Unpack batch - adjust based on your dataset's actual structure
            if len(batch_data) == 3:
                X, y, node_type = batch_data
                node_ids = None
                timesteps = None
            elif len(batch_data) == 6:
                X, y, node_type, node_ids, timesteps, event_ids = batch_data
            else:
                # Try to extract from dataset metadata
                X, y, node_type = batch_data[:3]
                node_ids = None
                timesteps = None

            X = X.to(device)
            y = y.to(device)
            node_type = node_type.to(device)

            preds = model(X, node_type)

            # Move back to CPU
            X_np = X.cpu().numpy()
            y_np = y.cpu().numpy()
            preds_np = preds.cpu().numpy()
            node_type_np = node_type.cpu().numpy()
            node_ids_np = node_ids.cpu().numpy()
            timesteps_np = timesteps.cpu().numpy()
            event_ids_np = event_ids.cpu().numpy()

            B, W, F = X_np.shape

            # Process each sample in batch
            for i in range(B):
                sample_idx = batch_idx * batch_size + i
                node_type_val = int(node_type_np[i])
                node_id = int(node_ids_np[i])
                timestep = int(timesteps_np[i])
                event_id = int(event_ids_np[i])

                # Select correct normalizer
                normalizer = normalizer_1d if node_type_val == 0 else normalizer_2d

                # Inverse transform
                if hasattr(normalizer, "inverse_transform_y"):
                    y_original = normalizer.inverse_transform_y(y_np[i : i + 1])[0, 0]
                    pred_original = normalizer.inverse_transform_y(preds_np[i : i + 1])[
                        0, 0
                    ]
                elif hasattr(normalizer, "inverse_y"):
                    y_original = normalizer.inverse_y(y_np[i : i + 1])[0, 0]
                    pred_original = normalizer.inverse_y(preds_np[i : i + 1])[0, 0]
                else:
                    y_original = y_np[i, 0]
                    pred_original = preds_np[i, 0]

                row = {
                    "sample_idx": sample_idx,
                    "event_id": event_id,
                    "node_id": node_id,
                    "timestep": timestep,
                    "node_type": node_type_val,
                    "target_water_level": float(y_original),
                    "predicted_water_level": float(pred_original),
                    "target_water_level_normalized": float(y_np[i, 0]),
                    "predicted_water_level_normalized": float(preds_np[i, 0]),
                }

                # # Add feature statistics across window
                # for f in range(F):
                #     feat_mean = X_np[i, :, f].mean()
                #     feat_std = X_np[i, :, f].std()
                #     feat_last = X_np[i, -1, f]

                #     row[f"feat_{f}_mean"] = float(feat_mean)
                #     row[f"feat_{f}_std"] = float(feat_std)
                #     row[f"feat_{f}_last"] = float(feat_last)

                rows.append(row)

            if (batch_idx + 1) % 100000 == 0:
                print(f"  Processed {(batch_idx + 1) * batch_size} samples...")

    df = pd.DataFrame(rows)
    df.to_csv(save_path, index=False)

    print(f"\n✓ Saved predictions to {save_path}")
    print(f"  Total samples: {len(df)}")
    print(f"  Unique nodes: {df['node_id'].nunique() if 'node_id' in df else 'N/A'}")
    print(f"  1D samples: {(df['node_type'] == 0).sum()}")
    print(f"  2D samples: {(df['node_type'] == 1).sum()}")

    return df


def save_predictions_autoregressive(
    model,
    dataset,
    normalizer_1d,
    normalizer_2d,
    save_path,
    water_level_idx=0,
    batch_size=32,
    device="cuda" if torch.cuda.is_available() else "cpu",
    initial_ground_truth_steps=0,  # Number of initial steps to use ground truth before AR
    max_ar_steps=None,  # Optional: limit max predictions per sequence (None = predict all)
    use_sequence_dataset=False,  # Whether input is SequenceDataset
):
    """
    Save autoregressive model predictions for ALL available timesteps.

    Works with both regular datasets and SequenceDataset:
    - Regular dataset: Groups samples by (event_id, node_id) internally
    - SequenceDataset: Uses pre-organized sequences directly (faster!)

    Prediction strategy:
    1. Get sequences (either from SequenceDataset or by grouping)
    2. For each sequence, predict ALL available timesteps
    3. Use ground truth warmup for first N steps if specified
    4. Then switch to pure autoregressive prediction

    Args:
        model: Trained TwoHeadGRU model
        dataset: Dataset (regular or SequenceDataset)
        normalizer_1d: Normalizer for 1D nodes
        normalizer_2d: Normalizer for 2D nodes
        save_path: Path to save CSV file
        water_level_idx: Index of water level feature in input
        batch_size: Batch size for inference (only for data loading)
        device: Device to run on
        initial_ground_truth_steps: How many initial steps use GT before pure AR
        max_ar_steps: Optional limit on predictions per sequence (None = no limit)
        use_sequence_dataset: Set True if dataset is SequenceDataset
    """
    model.eval()

    print("\nGenerating AUTOREGRESSIVE predictions for all timesteps...")
    print(f"  Device: {device}")
    print(f"  Water level index: {water_level_idx}")
    print(f"  Initial ground truth steps: {initial_ground_truth_steps}")
    print(
        f"  Max AR steps per sequence: {max_ar_steps if max_ar_steps else 'unlimited (predict all)'}"
    )
    print(f"  Using SequenceDataset: {use_sequence_dataset}")

    rows = []

    if use_sequence_dataset:
        # ================================================================
        # SEQUENCE DATASET MODE (Pre-organized sequences)
        # ================================================================
        print("\n  Using pre-organized SequenceDataset...")
        print(f"  Total sequences: {len(dataset)}")

        sample_idx = 0
        total_predictions = 0

        for seq_idx in range(len(dataset)):
            # Get full sequence
            X_seq, y_seq, node_type_val, node_id, timesteps, event_id, seq_len = (
                dataset[seq_idx]
            )

            # Convert to device
            X_seq = X_seq.to(device)  # (seq_len, window, features)
            y_seq = y_seq.to(device)  # (seq_len,)
            node_type_val = int(node_type_val.item())
            node_id = int(node_id.item())
            event_id = int(event_id.item())
            timesteps = timesteps.numpy()

            # Determine how many steps to predict
            num_steps = seq_len
            if max_ar_steps is not None:
                num_steps = min(num_steps, max_ar_steps)

            if num_steps == 0:
                continue

            # Get normalizer
            normalizer = normalizer_1d if node_type_val == 0 else normalizer_2d

            # Initialize with first window
            X_current = X_seq[0].unsqueeze(0)  # (1, window, features)
            node_type_tensor = torch.tensor([node_type_val], dtype=torch.long).to(
                device
            )

            # Get ground truth targets
            ground_truth_targets = y_seq[:num_steps].cpu().numpy()

            # Predict through sequence
            for step in range(num_steps):
                # Make prediction
                pred = model(X_current, node_type_tensor)  # (1, 1)
                pred_value = pred.squeeze().cpu().item()

                # Inverse transform prediction
                if hasattr(normalizer, "inverse_transform_y"):
                    pred_original = normalizer.inverse_transform_y(
                        np.array([[pred_value]])
                    )[0, 0]
                elif hasattr(normalizer, "inverse_y"):
                    pred_original = normalizer.inverse_y(np.array([[pred_value]]))[0, 0]
                else:
                    pred_original = pred_value

                # Get ground truth
                target_normalized = ground_truth_targets[step]
                if hasattr(normalizer, "inverse_transform_y"):
                    target_original = normalizer.inverse_transform_y(
                        np.array([[target_normalized]])
                    )[0, 0]
                elif hasattr(normalizer, "inverse_y"):
                    target_original = normalizer.inverse_y(
                        np.array([[target_normalized]])
                    )[0, 0]
                else:
                    target_original = target_normalized

                # Save prediction
                row = {
                    "sample_idx": sample_idx,
                    "event_id": event_id,
                    "node_id": node_id,
                    "base_timestep": int(timesteps[0]),
                    "ar_step": step + 1,
                    "predicted_timestep": int(timesteps[step]),
                    "node_type": node_type_val,
                    "target_water_level": float(target_original),
                    "predicted_water_level": float(pred_original),
                    "target_water_level_normalized": float(target_normalized),
                    "predicted_water_level_normalized": float(pred_value),
                    "used_ground_truth": step < initial_ground_truth_steps,
                    "is_pure_ar": step >= initial_ground_truth_steps,
                    "error": float(pred_original - target_original),
                    "abs_error": float(abs(pred_original - target_original)),
                    "squared_error": float((pred_original - target_original) ** 2),
                }
                rows.append(row)
                total_predictions += 1

                # Update window for next prediction
                if step < num_steps - 1:
                    next_timestep = X_current[:, -1:, :].clone()

                    # Decide whether to use ground truth or prediction
                    if step < initial_ground_truth_steps - 1 and step + 1 < seq_len:
                        # Use ground truth from next sample in sequence
                        gt_water_level = X_seq[step + 1, -1, water_level_idx]
                        next_timestep[0, 0, water_level_idx] = gt_water_level
                    else:
                        # Use prediction (autoregressive mode)
                        next_timestep[0, 0, water_level_idx] = pred.squeeze()

                    # Slide window
                    X_current = torch.cat([X_current[:, 1:, :], next_timestep], dim=1)

            sample_idx += 1

            if sample_idx % 100 == 0:
                print(
                    f"  Processed {sample_idx}/{len(dataset)} sequences ({total_predictions} predictions)..."
                )

        print(f"  Completed processing all {len(dataset)} sequences")

    else:
        # ================================================================
        # REGULAR DATASET MODE (Group samples internally)
        # ================================================================
        loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

        print("\n  Organizing data into sequences...")

        all_samples = []
        for batch_data in loader:
            X, y, node_type, node_ids, timesteps, event_ids = batch_data

            X_np = X.numpy()
            y_np = y.numpy()
            node_type_np = node_type.numpy()
            node_ids_np = node_ids.numpy()
            timesteps_np = timesteps.numpy()
            event_ids_np = event_ids.numpy()

            for i in range(len(X)):
                all_samples.append(
                    {
                        "X": X_np[i],
                        "y": y_np[i, 0],
                        "node_type": node_type_np[i],
                        "node_id": node_ids_np[i],
                        "timestep": timesteps_np[i],
                        "event_id": event_ids_np[i],
                    }
                )

        print(f"  Collected {len(all_samples)} samples")

        # Group by (event_id, node_id) to get sequences
        from collections import defaultdict

        sequences = defaultdict(list)
        for sample in all_samples:
            key = (sample["event_id"], sample["node_id"])
            sequences[key].append(sample)

        # Sort each sequence by timestep
        for key in sequences:
            sequences[key] = sorted(sequences[key], key=lambda x: x["timestep"])

        print(f"  Organized into {len(sequences)} sequences")

        # Analyze sequence lengths
        seq_lengths = [len(seq) for seq in sequences.values()]
        print("  Sequence length stats:")
        print(
            f"    Min: {min(seq_lengths)}, Max: {max(seq_lengths)}, Mean: {np.mean(seq_lengths):.1f}"
        )

        # Process each sequence
        sample_idx = 0
        total_predictions = 0

        for (event_id, node_id), seq_samples in sequences.items():
            # Determine how many steps to predict
            num_steps = len(seq_samples)
            if max_ar_steps is not None:
                num_steps = min(num_steps, max_ar_steps)

            if num_steps == 0:
                continue

            # Get sequence info
            first_sample = seq_samples[0]
            node_type_val = int(first_sample["node_type"])
            normalizer = normalizer_1d if node_type_val == 0 else normalizer_2d

            # Initialize with first window
            X_current = (
                torch.tensor(first_sample["X"], dtype=torch.float32)
                .unsqueeze(0)
                .to(device)
            )
            node_type_tensor = torch.tensor([node_type_val], dtype=torch.long).to(
                device
            )

            # Collect ground truth targets for comparison
            ground_truth_targets = [seq_samples[i]["y"] for i in range(num_steps)]

            # Perform predictions for all timesteps
            for step in range(num_steps):
                # Make prediction
                pred = model(X_current, node_type_tensor)  # (1, 1)
                pred_value = pred.squeeze().cpu().item()

                # Inverse transform prediction
                if hasattr(normalizer, "inverse_transform_y"):
                    pred_original = normalizer.inverse_transform_y(
                        np.array([[pred_value]])
                    )[0, 0]
                elif hasattr(normalizer, "inverse_y"):
                    pred_original = normalizer.inverse_y(np.array([[pred_value]]))[0, 0]
                else:
                    pred_original = pred_value

                # Get ground truth
                target_normalized = ground_truth_targets[step]
                if hasattr(normalizer, "inverse_transform_y"):
                    target_original = normalizer.inverse_transform_y(
                        np.array([[target_normalized]])
                    )[0, 0]
                elif hasattr(normalizer, "inverse_y"):
                    target_original = normalizer.inverse_y(
                        np.array([[target_normalized]])
                    )[0, 0]
                else:
                    target_original = target_normalized

                # Save prediction
                row = {
                    "sample_idx": sample_idx,
                    "event_id": int(event_id),
                    "node_id": int(node_id),
                    "base_timestep": int(first_sample["timestep"]),
                    "ar_step": step + 1,
                    "predicted_timestep": int(first_sample["timestep"]) + step + 1,
                    "node_type": node_type_val,
                    "target_water_level": float(target_original),
                    "predicted_water_level": float(pred_original),
                    "target_water_level_normalized": float(target_normalized),
                    "predicted_water_level_normalized": float(pred_value),
                    "used_ground_truth": step < initial_ground_truth_steps,
                    "is_pure_ar": step >= initial_ground_truth_steps,
                    "error": float(pred_original - target_original),
                    "abs_error": float(abs(pred_original - target_original)),
                    "squared_error": float((pred_original - target_original) ** 2),
                }
                rows.append(row)
                total_predictions += 1

                # Update window for next prediction
                next_timestep = X_current[:, -1:, :].clone()

                # Decide whether to use ground truth or prediction
                if step < initial_ground_truth_steps - 1 and step + 1 < len(
                    seq_samples
                ):
                    # Use ground truth for warmup period
                    next_gt_sample = seq_samples[step + 1]
                    gt_water_level = next_gt_sample["X"][-1, water_level_idx]
                    next_timestep[0, 0, water_level_idx] = torch.tensor(
                        gt_water_level, dtype=torch.float32
                    ).to(device)
                else:
                    # Use prediction (autoregressive mode)
                    next_timestep[0, 0, water_level_idx] = pred.squeeze()

                # Slide window
                X_current = torch.cat([X_current[:, 1:, :], next_timestep], dim=1)

            sample_idx += 1

            if sample_idx % 100 == 0:
                print(
                    f"  Processed {sample_idx}/{len(sequences)} sequences ({total_predictions} predictions)..."
                )

        print(f"  Completed processing all {len(sequences)} sequences")

    # ================================================================
    # SAVE AND REPORT RESULTS (Common for both modes)
    # ================================================================
    df = pd.DataFrame(rows)
    df.to_csv(save_path, index=False)

    print(f"\n✓ Saved autoregressive predictions to {save_path}")
    print(f"  Total predictions: {len(df)}")
    print(f"  Unique sequences: {df['sample_idx'].nunique()}")
    print(f"  Unique nodes: {df['node_id'].nunique()}")
    print(f"  Unique events: {df['event_id'].nunique()}")
    print(f"  1D predictions: {(df['node_type'] == 0).sum()}")
    print(f"  2D predictions: {(df['node_type'] == 1).sum()}")

    print("\n  Predictions breakdown:")
    print(f"    Ground truth warmup steps: {(df['used_ground_truth'] == True).sum()}")
    print(f"    Pure AR steps: {(df['is_pure_ar'] == True).sum()}")

    # Predictions per sequence statistics
    preds_per_seq = df.groupby("sample_idx").size()
    print("\n  Predictions per sequence:")
    print(
        f"    Min: {preds_per_seq.min()}, Max: {preds_per_seq.max()}, Mean: {preds_per_seq.mean():.1f}"
    )

    # Overall performance
    print("\n  Overall Performance:")
    print(f"    RMSE: {np.sqrt(df['squared_error'].mean()):.4f}")
    print(f"    MAE: {df['abs_error'].mean():.4f}")
    print(f"    Mean Error: {df['error'].mean():.4f}")

    # Performance by warmup vs pure AR
    df_warmup = df[df["used_ground_truth"] == True]
    df_pure_ar = df[df["is_pure_ar"] == True]

    if len(df_warmup) > 0:
        print("\n  Performance (warmup steps with GT in window):")
        print(f"    RMSE: {np.sqrt(df_warmup['squared_error'].mean()):.4f}")
        print(f"    MAE: {df_warmup['abs_error'].mean():.4f}")
        print(f"    Count: {len(df_warmup)}")

    if len(df_pure_ar) > 0:
        print("\n  Performance (pure AR steps):")
        print(f"    RMSE: {np.sqrt(df_pure_ar['squared_error'].mean()):.4f}")
        print(f"    MAE: {df_pure_ar['abs_error'].mean():.4f}")
        print(f"    Count: {len(df_pure_ar)}")

    # Per node type
    print("\n  Performance by node type:")
    for nt in [0, 1]:
        df_nt = df[df["node_type"] == nt]
        if len(df_nt) > 0:
            node_type_name = "1D" if nt == 0 else "2D"
            print(
                f"    {node_type_name} RMSE: {np.sqrt(df_nt['squared_error'].mean()):.4f} ({len(df_nt)} predictions)"
            )

    # Performance by prediction horizon (first 10 steps)
    print("\n  Performance by prediction step (first 10 steps):")
    for step in range(1, min(11, df["ar_step"].max() + 1)):
        df_step = df[df["ar_step"] == step]
        if len(df_step) > 0:
            warmup_flag = (
                " (warmup)" if step <= initial_ground_truth_steps else " (pure AR)"
            )
            print(
                f"    Step {step:2d}{warmup_flag}: RMSE={np.sqrt(df_step['squared_error'].mean()):.4f}, MAE={df_step['abs_error'].mean():.4f} (n={len(df_step)})"
            )

    return df


def load_checkpoint(checkpoint_path, model, optimizer=None, device="cpu"):
    """
    Load model from checkpoint.

    Args:
        checkpoint_path: Path to checkpoint file
        model: Model instance to load weights into
        optimizer: Optional optimizer to load state
        device: Device to load model on

    Returns:
        model, optimizer (if provided), checkpoint_dict
    """
    checkpoint = torch.load(checkpoint_path, map_location=device)

    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)

    if optimizer is not None and "optimizer_state_dict" in checkpoint:
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

    print(f"Loaded checkpoint from epoch {checkpoint.get('epoch', 'unknown')}")
    print(f"  Train RMSE: {checkpoint.get('train_rmse', 'N/A')}")
    print(f"  Val RMSE: {checkpoint.get('val_rmse', 'N/A')}")

    if "train_rmse_1d" in checkpoint:
        print(f"  Train RMSE (1D): {checkpoint.get('train_rmse_1d', 'N/A'):.4f}")
        print(f"  Train RMSE (2D): {checkpoint.get('train_rmse_2d', 'N/A'):.4f}")
        print(f"  Val RMSE (1D): {checkpoint.get('val_rmse_1d', 'N/A'):.4f}")
        print(f"  Val RMSE (2D): {checkpoint.get('val_rmse_2d', 'N/A'):.4f}")

    return model, optimizer, checkpoint
