import pandas as pd
import matplotlib.pyplot as plt
from constants import FEATURE_NAMES, STATIC_FEATURES, DYNAMIC_FEATURES
from torch.utils.data import DataLoader
import torch
from pathlib import Path


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
