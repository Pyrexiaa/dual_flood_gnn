from constants import FEATURE_NAMES
from debug import full_debug_workflow
from save import save_predictions
from utils import evaluate_predictions, NSE
from scaler import SequenceNormalizer
from model import TwoHeadGRU
from dataset import CombinedDataset, WaterLevelDataset1D, WaterLevelDataset2D
from torch.utils.data import DataLoader
from pathlib import Path
import torch
import torch.nn as nn
import joblib


def create_filtered_datasets(
    train_events,
    test_events,
    window,
    normalizer_1d,
    normalizer_2d,
    debug=False,
    max_events=None,
):
    """
    Create separate 1D and 2D datasets using node_type_filter.

    IMPORTANT: This creates separate normalizers for 1D and 2D data because:
    - 1D nodes have different feature distributions than 2D nodes
    - 1D features (positions, inlet_flow) are active, 2D features are padded zeros
    - 2D features (terrain, rainfall) are active, 1D features are padded zeros
    - Normalizing them together would be incorrect

    Args:
        train_events: Path to training event directories
        test_events: Path to test event directories
        window: Window size
        normalizer_1d: Normalizer instance for 1D data (e.g., SequenceNormalizer())
        normalizer_2d: Normalizer instance for 2D data (e.g., SequenceNormalizer())
        debug: Whether to enable debug mode
        max_events: Maximum number of events to process (for testing)

    Returns:
        train_1d, train_2d, test_1d, test_2d, trained_normalizer_1d, trained_normalizer_2d
    """

    print("=" * 80)
    print("CREATING NODE-LEVEL DATASETS (NO EDGE FEATURES)")
    print("=" * 80)

    # Create 1D training dataset
    print("\n1. Creating 1D TRAINING dataset...")
    print("   Features: 1d_position_x, 1d_position_y")
    train_1d = WaterLevelDataset1D(
        event_dirs=train_events,
        window=window,
        normalizer=normalizer_1d,
        fit_normalizer=True,
        return_sequence=True,
        debug=debug,
        max_events=max_events,
        verbose=True,
        normalizer_save_path="debug_train_normalizer_1d.pkl",
    )

    # Create 2D training dataset
    print("\n2. Creating 2D TRAINING dataset...")
    print("   Features: 2d_position_x/y, area, roughness, elevation, aspect,")
    print("             curvature, flow_accumulation, slope, rainfall")
    train_2d = WaterLevelDataset2D(
        event_dirs=train_events,
        window=window,
        normalizer=normalizer_2d,
        fit_normalizer=True,
        return_sequence=True,
        debug=debug,
        max_events=max_events,
        verbose=True,
        normalizer_save_path="debug_train_normalizer_2d.pkl",
    )

    # Load fitted normalizers
    trained_normalizer_1d = joblib.load("debug_train_normalizer_1d.pkl")
    trained_normalizer_2d = joblib.load("debug_train_normalizer_2d.pkl")

    # Create 1D test dataset
    print("\n3. Creating 1D TEST dataset...")
    test_1d = WaterLevelDataset1D(
        event_dirs=test_events,
        window=window,
        normalizer=trained_normalizer_1d,
        fit_normalizer=False,
        return_sequence=True,
        debug=debug,
        max_events=max_events,
        verbose=True,
    )

    # Create 2D test dataset
    print("\n4. Creating 2D TEST dataset...")
    test_2d = WaterLevelDataset2D(
        event_dirs=test_events,
        window=window,
        normalizer=trained_normalizer_2d,
        fit_normalizer=False,
        return_sequence=True,
        debug=debug,
        max_events=max_events,
        verbose=True,
    )

    print("\n" + "=" * 80)
    print("DATASET CREATION COMPLETE")
    print("=" * 80)
    print("\nDataset sizes:")
    print(
        f"  Train 1D: {len(train_1d):>8,} samples  (shape: {train_1d.dataset.X.shape})"
    )
    print(
        f"  Train 2D: {len(train_2d):>8,} samples  (shape: {train_2d.dataset.X.shape})"
    )
    print(f"  Test 1D:  {len(test_1d):>8,} samples  (shape: {test_1d.dataset.X.shape})")
    print(f"  Test 2D:  {len(test_2d):>8,} samples  (shape: {test_2d.dataset.X.shape})")

    print("\nFeature dimensions:")
    print(
        f"  1D feature dim: {train_1d.dataset.X.shape[-1]} (from {len(FEATURE_NAMES)} total)"
    )
    print(
        f"  2D feature dim: {train_2d.dataset.X.shape[-1]} (from {len(FEATURE_NAMES)} total)"
    )
    print(f"  ✓ Both should be {len(FEATURE_NAMES)} after padding")

    return (
        train_1d,
        train_2d,
        test_1d,
        test_2d,
        trained_normalizer_1d,
        trained_normalizer_2d,
    )


def train(
    train_events,
    test_events,
    window,
    max_events,
    normalizer_1d,
    normalizer_2d,
    epochs=50,
    lr=1e-3,
    batch_size=32,
    hidden_dim=128,
    device="cuda" if torch.cuda.is_available() else "cpu",
    save_checkpoints=True,
    checkpoint_dir="./two_head_checkpoints",
):
    """
    Train the two-head GRU model with NODE-LEVEL FEATURES ONLY.

    This baseline model uses only node features (no edge aggregation) for faster training.

    Args:
        train_events: Path to training event directories
        test_events: Path to test event directories
        window: Window size for sequences
        max_events: Maximum number of events to process (for testing)
        normalizer_1d: Normalizer for 1D nodes
        normalizer_2d: Normalizer for 2D nodes
        epochs: Number of training epochs
        lr: Learning rate
        batch_size: Batch size
        hidden_dim: Hidden dimension for GRU
        device: 'cuda' or 'cpu'
        save_checkpoints: Whether to save model checkpoints
        checkpoint_dir: Directory to save checkpoints

    Returns:
        model: Trained TwoHeadGRU model
        train_combined: Combined training dataset
        test_combined: Combined test dataset
        trained_normalizer_1d: Fitted 1D normalizer
        trained_normalizer_2d: Fitted 2D normalizer
    """
    print(f"Training on device: {device}")
    print("Architecture: Two-head model with shared GRU backbone")
    print("Features: NODE-LEVEL ONLY (no edge features)")

    # =====================
    # DATASETS
    # =====================
    print("\n" + "=" * 80)
    print("CREATING DATASETS")
    print("=" * 80)

    # Create filtered datasets for 1D and 2D separately
    (
        train_1d,
        train_2d,
        test_1d,
        test_2d,
        trained_normalizer_1d,
        trained_normalizer_2d,
    ) = create_filtered_datasets(
        train_events,
        test_events,
        window=window,
        normalizer_1d=normalizer_1d,
        normalizer_2d=normalizer_2d,
        debug=False,
        max_events=max_events,
    )

    # Save the fitted normalizers
    joblib.dump(trained_normalizer_1d, "train_normalizer_two_head_1d.pkl")
    joblib.dump(trained_normalizer_2d, "train_normalizer_two_head_2d.pkl")
    print(
        "  → Saved fitted normalizers to 'train_normalizer_two_head_1d.pkl' and 'train_normalizer_two_head_2d.pkl'"
    )

    # Create combined datasets with node_type labels
    train_combined = CombinedDataset(train_1d, train_2d)
    test_combined = CombinedDataset(test_1d, test_2d)

    # Get input dimensions (should be same for both after padding)
    input_dim = train_1d.dataset.X.shape[-1]

    print("\nDataset Statistics:")
    print(f"  Training samples (1D): {len(train_1d)}")
    print(f"  Training samples (2D): {len(train_2d)}")
    print(f"  Training samples (Total): {len(train_combined)}")
    print(f"  Test samples (1D): {len(test_1d)}")
    print(f"  Test samples (2D): {len(test_2d)}")
    print(f"  Test samples (Total): {len(test_combined)}")
    print(f"  Input dimension: {input_dim} (both 1D and 2D)")
    print(f"  Window size: {window}")
    print(f"  Hidden dimension: {hidden_dim}")

    # =====================
    # DATA LOADERS
    # =====================
    # NOTE: Change num_workers to 2 when using compute clusters, else 0
    train_loader = DataLoader(
        train_combined, batch_size=batch_size, shuffle=True, num_workers=2
    )
    test_loader = DataLoader(
        test_combined, batch_size=batch_size, shuffle=False, num_workers=2
    )

    print("\nData Loaders:")
    print(f"  Train batches: {len(train_loader)}")
    print(f"  Test batches: {len(test_loader)}")
    print(f"  Batch size: {batch_size}")

    # =====================
    # MODEL
    # =====================
    print("\n" + "=" * 80)
    print("INITIALIZING MODEL")
    print("=" * 80)

    model = TwoHeadGRU(input_dim=input_dim, hidden_dim=hidden_dim).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.MSELoss()

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    print("Model Architecture:")
    print(f"  Total parameters: {total_params:,}")
    print(f"  Trainable parameters: {trainable_params:,}")
    print(f"  Optimizer: Adam (lr={lr})")
    print("  Loss function: MSE")

    # Create checkpoint directory
    if save_checkpoints:
        checkpoint_path = Path(checkpoint_dir)
        checkpoint_path.mkdir(exist_ok=True, parents=True)
        print(f"\nCheckpoint directory: {checkpoint_path.absolute()}")

    best_val_rmse = float("inf")
    best_val_rmse_1d = float("inf")
    best_val_rmse_2d = float("inf")
    best_val_nse = float("-inf")
    best_val_nse_modified = float("-inf")
    best_val_nse_1d = float("-inf")
    best_val_nse_2d = float("-inf")

    # =====================
    # TRAINING LOOP
    # =====================
    print("\n" + "=" * 80)
    print("TRAINING")
    print("=" * 80 + "\n")

    for epoch in range(epochs):
        # =====================
        # TRAINING PHASE
        # =====================
        model.train()

        train_loss_sum = 0.0
        train_loss_1d = 0.0
        train_loss_2d = 0.0
        n_samples_total = 0
        n_samples_1d = 0
        n_samples_2d = 0

        # For NSE calculation (accumulate predictions and targets)
        train_preds_all = []
        train_targets_all = []
        train_preds_1d = []
        train_targets_1d = []
        train_preds_2d = []
        train_targets_2d = []

        for batch_idx, (X, y, node_type) in enumerate(train_loader):
            X = X.to(device)  # (batch, window, features)
            y = y.to(device)  # (batch, 1)
            node_type = node_type.to(device)  # (batch,)

            # Forward pass
            pred = model(X, node_type)  # (batch, 1)
            loss = criterion(pred, y)

            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # Track overall loss
            batch_size_actual = len(X)
            train_loss_sum += loss.item() * batch_size_actual
            n_samples_total += batch_size_actual

            # Accumulate for NSE
            train_preds_all.append(pred.detach())
            train_targets_all.append(y.detach())

            # Track per-type losses
            mask_1d = node_type == 0
            mask_2d = node_type == 1

            if mask_1d.any():
                loss_1d_batch = criterion(pred[mask_1d], y[mask_1d]).item()
                count_1d = mask_1d.sum().item()
                train_loss_1d += loss_1d_batch * count_1d
                n_samples_1d += count_1d

                train_preds_1d.append(pred[mask_1d].detach())
                train_targets_1d.append(y[mask_1d].detach())

            if mask_2d.any():
                loss_2d_batch = criterion(pred[mask_2d], y[mask_2d]).item()
                count_2d = mask_2d.sum().item()
                train_loss_2d += loss_2d_batch * count_2d
                n_samples_2d += count_2d

                train_preds_2d.append(pred[mask_2d].detach())
                train_targets_2d.append(y[mask_2d].detach())

        # Compute training metrics
        train_rmse = (train_loss_sum / n_samples_total) ** 0.5
        train_rmse_1d = (
            (train_loss_1d / n_samples_1d) ** 0.5 if n_samples_1d > 0 else 0.0
        )
        train_rmse_2d = (
            (train_loss_2d / n_samples_2d) ** 0.5 if n_samples_2d > 0 else 0.0
        )

        # Compute training NSE
        train_preds_all = torch.cat(train_preds_all, dim=0)
        train_targets_all = torch.cat(train_targets_all, dim=0)
        train_nse = NSE(train_preds_all, train_targets_all).item()

        train_nse_1d = 0.0
        if len(train_preds_1d) > 0:
            train_preds_1d = torch.cat(train_preds_1d, dim=0)
            train_targets_1d = torch.cat(train_targets_1d, dim=0)
            train_nse_1d = NSE(train_preds_1d, train_targets_1d).item()

        train_nse_2d = 0.0
        if len(train_preds_2d) > 0:
            train_preds_2d = torch.cat(train_preds_2d, dim=0)
            train_targets_2d = torch.cat(train_targets_2d, dim=0)
            train_nse_2d = NSE(train_preds_2d, train_targets_2d).item()

        train_nse_modified = (
            (train_nse_1d + train_nse_2d) / 2
            if (n_samples_1d > 0 and n_samples_2d > 0)
            else train_nse
        )

        # =====================
        # VALIDATION PHASE
        # =====================
        print("\n" + "=" * 80)
        print(f"VALIDATING EPOCH {epoch + 1}/{epochs}")
        print("=" * 80 + "\n")
        model.eval()

        val_loss_sum = 0.0
        val_loss_1d = 0.0
        val_loss_2d = 0.0
        v_samples_total = 0
        v_samples_1d = 0
        v_samples_2d = 0

        # For NSE calculation
        val_preds_all = []
        val_targets_all = []
        val_preds_1d = []
        val_targets_1d = []
        val_preds_2d = []
        val_targets_2d = []

        with torch.no_grad():
            for X, y, node_type in test_loader:
                X = X.to(device)
                y = y.to(device)
                node_type = node_type.to(device)

                pred = model(X, node_type)
                loss = criterion(pred, y)

                batch_size_actual = len(X)
                val_loss_sum += loss.item() * batch_size_actual
                v_samples_total += batch_size_actual

                # Accumulate for NSE
                val_preds_all.append(pred)
                val_targets_all.append(y)

                # Per-type validation losses
                mask_1d = node_type == 0
                mask_2d = node_type == 1

                if mask_1d.any():
                    loss_1d_batch = criterion(pred[mask_1d], y[mask_1d]).item()
                    count_1d = mask_1d.sum().item()
                    val_loss_1d += loss_1d_batch * count_1d
                    v_samples_1d += count_1d

                    val_preds_1d.append(pred[mask_1d])
                    val_targets_1d.append(y[mask_1d])

                if mask_2d.any():
                    loss_2d_batch = criterion(pred[mask_2d], y[mask_2d]).item()
                    count_2d = mask_2d.sum().item()
                    val_loss_2d += loss_2d_batch * count_2d
                    v_samples_2d += count_2d

                    val_preds_2d.append(pred[mask_2d])
                    val_targets_2d.append(y[mask_2d])

        # Compute validation RMSE
        val_rmse = (val_loss_sum / v_samples_total) ** 0.5
        val_rmse_1d = (val_loss_1d / v_samples_1d) ** 0.5 if v_samples_1d > 0 else 0.0
        val_rmse_2d = (val_loss_2d / v_samples_2d) ** 0.5 if v_samples_2d > 0 else 0.0

        # Compute validation NSE
        val_preds_all = torch.cat(val_preds_all, dim=0)
        val_targets_all = torch.cat(val_targets_all, dim=0)
        val_nse = NSE(val_preds_all, val_targets_all).item()

        val_nse_1d = 0.0
        if len(val_preds_1d) > 0:
            val_preds_1d = torch.cat(val_preds_1d, dim=0)
            val_targets_1d = torch.cat(val_targets_1d, dim=0)
            val_nse_1d = NSE(val_preds_1d, val_targets_1d).item()

        val_nse_2d = 0.0
        if len(val_preds_2d) > 0:
            val_preds_2d = torch.cat(val_preds_2d, dim=0)
            val_targets_2d = torch.cat(val_targets_2d, dim=0)
            val_nse_2d = NSE(val_preds_2d, val_targets_2d).item()

        val_nse_modified = (
            (val_nse_1d + val_nse_2d) / 2
            if (v_samples_1d > 0 and v_samples_2d > 0)
            else val_nse
        )

        # =====================
        # LOGGING
        # =====================
        print(
            f"Epoch {epoch + 1:03d}/{epochs}\n"
            f"  Train RMSE: {train_rmse:.4f} (1D: {train_rmse_1d:.4f}, 2D: {train_rmse_2d:.4f})\n"
            f"  Train NSE:  {train_nse:.4f} (1D: {train_nse_1d:.4f}, 2D: {train_nse_2d:.4f})\n"
            f"  Val RMSE:   {val_rmse:.4f} (1D: {val_rmse_1d:.4f}, 2D: {val_rmse_2d:.4f})\n"
            f"  Val NSE:    {val_nse:.4f} (1D: {val_nse_1d:.4f}, 2D: {val_nse_2d:.4f})"
        )

        # =====================
        # CHECKPOINT SAVING
        # =====================
        if save_checkpoints:
            checkpoint = {
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "modified_train_nse": train_nse_modified,
                "modified_val_nse": val_nse_modified,
                "train_rmse": train_rmse,
                "train_rmse_1d": train_rmse_1d,
                "train_rmse_2d": train_rmse_2d,
                "train_nse": train_nse,
                "train_nse_1d": train_nse_1d,
                "train_nse_2d": train_nse_2d,
                "val_rmse": val_rmse,
                "val_rmse_1d": val_rmse_1d,
                "val_rmse_2d": val_rmse_2d,
                "val_nse": val_nse,
                "val_nse_1d": val_nse_1d,
                "val_nse_2d": val_nse_2d,
                "input_dim_1d": input_dim,
                "input_dim_2d": input_dim,
                "hidden_dim": hidden_dim,
                "window": window,
                "lr": lr,
            }

            # Save latest checkpoint
            torch.save(
                checkpoint, checkpoint_path / f"checkpoint_epoch_{epoch + 1:03d}.pt"
            )

            # Save best overall model (by RMSE)
            if val_rmse < best_val_rmse:
                best_val_rmse = val_rmse
                torch.save(checkpoint, checkpoint_path / "best_model_overall_rmse.pt")
                print(
                    f"  → Saved best overall model by RMSE (val_rmse: {val_rmse:.4f})"
                )

            # Save best overall model (by NSE)
            if val_nse > best_val_nse:
                best_val_nse = val_nse
                torch.save(checkpoint, checkpoint_path / "best_model_overall_nse.pt")
                print(f"  → Saved best overall model by NSE (val_nse: {val_nse:.4f})")

            # Save best overall model (by Modified NSE)
            if val_nse_modified > best_val_nse_modified:
                best_val_nse_modified = val_nse_modified
                torch.save(
                    checkpoint, checkpoint_path / "best_model_overall_nse_modified.pt"
                )
                print(
                    f"  → Saved best overall model by Modified NSE (val_nse_modified: {val_nse_modified:.4f})"
                )

            # Save best 1D model (by RMSE)
            if val_rmse_1d < best_val_rmse_1d and v_samples_1d > 0:
                best_val_rmse_1d = val_rmse_1d
                torch.save(checkpoint, checkpoint_path / "best_model_1d_rmse.pt")
                print(
                    f"  → Saved best 1D model by RMSE (val_rmse_1d: {val_rmse_1d:.4f})"
                )

            # Save best 1D model (by NSE)
            if val_nse_1d > best_val_nse_1d and v_samples_1d > 0:
                best_val_nse_1d = val_nse_1d
                torch.save(checkpoint, checkpoint_path / "best_model_1d_nse.pt")
                print(f"  → Saved best 1D model by NSE (val_nse_1d: {val_nse_1d:.4f})")

            # Save best 2D model (by RMSE)
            if val_rmse_2d < best_val_rmse_2d and v_samples_2d > 0:
                best_val_rmse_2d = val_rmse_2d
                torch.save(checkpoint, checkpoint_path / "best_model_2d_rmse.pt")
                print(
                    f"  → Saved best 2D model by RMSE (val_rmse_2d: {val_rmse_2d:.4f})"
                )

            # Save best 2D model (by NSE)
            if val_nse_2d > best_val_nse_2d and v_samples_2d > 0:
                best_val_nse_2d = val_nse_2d
                torch.save(checkpoint, checkpoint_path / "best_model_2d_nse.pt")
                print(f"  → Saved best 2D model by NSE (val_nse_2d: {val_nse_2d:.4f})")

        print()  # Empty line for readability

    # =====================
    # TRAINING SUMMARY
    # =====================
    print("=" * 80)
    print("TRAINING COMPLETE")
    print("=" * 80)
    print(f"Best Validation RMSE (Overall): {best_val_rmse:.4f}")
    print(f"Best Validation RMSE (1D): {best_val_rmse_1d:.4f}")
    print(f"Best Validation RMSE (2D): {best_val_rmse_2d:.4f}")
    print(f"Best Validation NSE (Modified Overall): {best_val_nse_modified:.4f}")
    print(f"Best Validation NSE (Overall): {best_val_nse:.4f}")
    print(f"Best Validation NSE (1D): {best_val_nse_1d:.4f}")
    print(f"Best Validation NSE (2D): {best_val_nse_2d:.4f}")

    if save_checkpoints:
        print(f"\nCheckpoints saved to: {checkpoint_path.absolute()}")
        print("  - best_model_overall_rmse.pt")
        print("  - best_model_overall_nse.pt")
        print("  - best_model_overall_nse_modified.pt")
        print("  - best_model_1d_rmse.pt")
        print("  - best_model_1d_nse.pt")
        print("  - best_model_2d_rmse.pt")
        print("  - best_model_2d_nse.pt")
        print("  - checkpoint_epoch_*.pt (all epochs)")

    print("=" * 80 + "\n")

    return (
        model,
        train_combined,
        test_combined,
        trained_normalizer_1d,
        trained_normalizer_2d,
    )


def load_model_and_predict(
    max_events=None,
    checkpoint_path="./two_head_checkpoints/best_model_overall.pt",
    test_events=Path("data/Model1/processed/features_csv/test/"),
    normalizer_1d_path="train_normalizer_two_head_1d.pkl",
    normalizer_2d_path="train_normalizer_two_head_2d.pkl",
    device="cuda" if torch.cuda.is_available() else "cpu",
    save_path="gru_test_predictions.csv",
):
    """
    Load trained model and generate predictions on test set.

    Args:
        checkpoint_path: Path to saved model checkpoint
        test_events: Path to test event directories
        normalizer_1d_path: Path to fitted 1D normalizer
        normalizer_2d_path: Path to fitted 2D normalizer
        device: 'cuda' or 'cpu'
        save_path: Where to save predictions CSV

    Returns:
        pred_df: DataFrame with predictions
        metrics: Evaluation metrics
    """

    print("=" * 80)
    print("LOADING MODEL AND GENERATING PREDICTIONS")
    print("=" * 80)

    # =====================
    # 1. LOAD CHECKPOINT
    # =====================
    print(f"\n1. Loading checkpoint from: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location=device)

    # Extract model hyperparameters from checkpoint
    input_dim = checkpoint["input_dim_1d"]  # Should be same as input_dim_2d
    hidden_dim = checkpoint["hidden_dim"]
    window = checkpoint["window"]

    print("   Model config from checkpoint:")
    print(f"   - Input dim: {input_dim}")
    print(f"   - Hidden dim: {hidden_dim}")
    print(f"   - Window: {window}")
    print(f"   - Best val RMSE: {checkpoint['val_rmse']:.4f}")
    print(f"   - Epoch: {checkpoint['epoch'] + 1}")

    # =====================
    # 2. INITIALIZE MODEL
    # =====================
    print("\n2. Initializing model...")
    model = TwoHeadGRU(input_dim=input_dim, hidden_dim=hidden_dim).to(device)

    # Load trained weights
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    print("   ✓ Model weights loaded successfully")
    print(f"   Total parameters: {sum(p.numel() for p in model.parameters()):,}")

    # =====================
    # 3. LOAD NORMALIZERS
    # =====================
    print("\n3. Loading fitted normalizers...")
    trained_normalizer_1d = joblib.load(normalizer_1d_path)
    trained_normalizer_2d = joblib.load(normalizer_2d_path)
    print(f"   ✓ 1D normalizer loaded from: {normalizer_1d_path}")
    print(f"   ✓ 2D normalizer loaded from: {normalizer_2d_path}")

    # =====================
    # 4. RECREATE TEST DATASETS
    # =====================
    print("\n4. Recreating test datasets...")

    # Create 1D test dataset
    print("   Creating 1D test dataset...")
    test_1d = WaterLevelDataset1D(
        event_dirs=test_events,
        window=window,
        normalizer=trained_normalizer_1d,
        fit_normalizer=False,  # Use pre-fitted normalizer
        return_sequence=True,
        debug=False,
        max_events=max_events,
        verbose=True,
    )

    # Create 2D test dataset
    print("   Creating 2D test dataset...")
    test_2d = WaterLevelDataset2D(
        event_dirs=test_events,
        window=window,
        normalizer=trained_normalizer_2d,
        fit_normalizer=False,  # Use pre-fitted normalizer
        return_sequence=True,
        debug=False,
        max_events=max_events,
        verbose=True,
    )

    # In load_model_and_predict(), after creating datasets:
    print("\n   Dataset feature dimensions:")
    print(f"   - 1D features: {test_1d.dataset.X.shape[-1]}")
    print(f"   - 2D features: {test_2d.dataset.X.shape[-1]}")
    print(f"   - Model expects: {input_dim}")

    # Add assertion to catch mismatch
    assert test_1d.dataset.X.shape[-1] == input_dim, \
        f"1D feature mismatch! Dataset has {test_1d.dataset.X.shape[-1]}, model expects {input_dim}"
    assert test_2d.dataset.X.shape[-1] == input_dim, \
        f"2D feature mismatch! Dataset has {test_2d.dataset.X.shape[-1]}, model expects {input_dim}"

    # Combine datasets
    test_combined = CombinedDataset(test_1d, test_2d)

    print("\n   Test dataset statistics:")
    print(f"   - 1D samples: {len(test_1d):,}")
    print(f"   - 2D samples: {len(test_2d):,}")
    print(f"   - Total samples: {len(test_combined):,}")

    # =====================
    # 5. GENERATE PREDICTIONS
    # =====================
    print("\n5. Generating predictions...")
    pred_df = save_predictions(
        model=model,
        dataset=test_combined,
        normalizer_1d=trained_normalizer_1d,
        normalizer_2d=trained_normalizer_2d,
        save_path=save_path,
    )
    print(f"   ✓ Predictions saved to: {save_path}")

    # =====================
    # 6. EVALUATE
    # =====================
    print("\n6. Evaluating predictions...")
    metrics = evaluate_predictions(save_path)

    print("\n" + "=" * 80)
    print("COMPLETE")
    print("=" * 80)

    return pred_df, metrics


if __name__ == "__main__":
    model_name = "Model1"
    train_events = Path(f"data/{model_name}/processed/features_csv/train/")
    test_events = Path(f"data/{model_name}/processed/features_csv/test/")

    debug_dataset = False
    if debug_dataset:
        normalizer_1d = SequenceNormalizer()
        normalizer_2d = SequenceNormalizer()
        (
            train_1d,
            train_2d,
            test_1d,
            test_2d,
            debug_trained_normalizer_1d,
            debug_trained_normalizer_2d,
        ) = create_filtered_datasets(
            train_events,
            test_events,
            window=5,
            normalizer_1d=normalizer_1d,
            normalizer_2d=normalizer_2d,
            debug=True,
            max_events=2,
        )
        train_combined = CombinedDataset(train_1d, train_2d)
        test_combined = CombinedDataset(test_1d, test_2d)
        full_debug_workflow(
            train_combined,
            test_combined,
            debug_trained_normalizer_1d,
            debug_trained_normalizer_2d,
        )

    train_bool = True
    if train_bool:
        new_normalizer_1d = SequenceNormalizer()
        new_normalizer_2d = SequenceNormalizer()
        model, train_ds, test_ds, trained_normalizer_1d, trained_normalizer_2d = train(
            train_events,
            test_events,
            window=5,
            max_events=None,  # Set None to use full dataset
            normalizer_1d=new_normalizer_1d,
            normalizer_2d=new_normalizer_2d,
            epochs=1,
            lr=1e-3,
            batch_size=32,
            hidden_dim=128,
            device="cuda" if torch.cuda.is_available() else "cpu",
            save_checkpoints=True,
            checkpoint_dir="./two_head_checkpoints",
        )

        test_save_path = "gru_test_predictions.csv"

        # Save predictions
        pred_df = save_predictions(
            model=model,
            dataset=test_ds,
            normalizer_1d=trained_normalizer_1d,
            normalizer_2d=trained_normalizer_2d,
            save_path=test_save_path,
        )

        # Evaluate
        metrics = evaluate_predictions(test_save_path)

    test_only = False
    if test_only:
        load_model_and_predict(
            max_events=None,
            checkpoint_path="./two_head_checkpoints/best_model_overall_nse.pt",
            test_events=test_events,
            normalizer_1d_path="train_normalizer_two_head_1d.pkl",
            normalizer_2d_path="train_normalizer_two_head_2d.pkl",
            device="cuda" if torch.cuda.is_available() else "cpu",
            save_path="gru_test_predictions.csv",
        )
