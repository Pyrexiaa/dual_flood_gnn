from constants import FEATURE_NAMES
from debug import full_debug_workflow
from save import save_predictions, save_predictions_autoregressive
from utils import evaluate_predictions_hierarchical
from scaler import SequenceNormalizer
from model import TwoHeadGRU
from dataset import (
    CombinedDataset,
    WaterLevelDataset1D,
    WaterLevelDataset2D,
    SequenceDataset,
    collate_sequences,
)
from torch.utils.data import DataLoader
from pathlib import Path
import torch
import joblib
import pickle


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


def train_autoregressive(
    train_events,
    test_events,
    window,
    max_events,
    normalizer_1d,
    normalizer_2d,
    epochs=50,
    lr=1e-3,
    batch_size=16,  # Smaller batch size for sequences
    hidden_dim=128,
    device="cuda" if torch.cuda.is_available() else "cpu",
    save_checkpoints=True,
    checkpoint_dir="./two_head_checkpoints_seq_ar",
    teacher_forcing_ratio=1.0,
    teacher_forcing_decay=0.95,
    water_level_idx=0,
    min_sequence_length=2,
    max_sequence_length=10000,
):
    """
    Train with TRUE SEQUENTIAL AUTOREGRESSIVE prediction on full sequences.

    This version uses the actual consecutive timesteps from your data to train
    the model autoregressively across entire sequences.

    Key difference from previous version:
    - Uses SequenceDataset to group consecutive timesteps
    - Each training sample is a full sequence of consecutive timesteps
    - Model learns to predict across the entire sequence autoregressively

    Args:
        min_sequence_length: Minimum consecutive timesteps required
        max_sequence_length: Maximum sequence length per training sample
        Other args same as before
    """
    print(f"Training on device: {device}")
    print("Architecture: Sequential AR training on FULL SEQUENCES")
    print(f"Water level index: {water_level_idx}")
    print(f"Window size: {window}")
    print(f"Sequence length: {min_sequence_length}-{max_sequence_length} timesteps")

    if Path("train_seq_dataset.pkl").exists() and Path("test_seq_dataset.pkl").exists():
        print("Loading cached datasets...")

        with open("train_seq_dataset.pkl", "rb") as f:
            train_seq = pickle.load(f)
        with open("test_seq_dataset.pkl", "rb") as f:
            test_seq = pickle.load(f)

        # Load normalizers too!
        trained_normalizer_1d = joblib.load("train_normalizer_seq_ar_1d.pkl")
        trained_normalizer_2d = joblib.load("train_normalizer_seq_ar_2d.pkl")

        print(
            f"Loaded {len(train_seq)} training sequences and {len(test_seq)} test sequences"
        )

    else:
        print("Preparing datasets from scratch...")

        # Create base datasets
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

        # Save normalizers
        joblib.dump(trained_normalizer_1d, "train_normalizer_seq_ar_1d.pkl")
        joblib.dump(trained_normalizer_2d, "train_normalizer_seq_ar_2d.pkl")

        # Create combined base datasets
        train_combined_base = CombinedDataset(train_1d, train_2d)
        test_combined_base = CombinedDataset(test_1d, test_2d)

        # Create sequence datasets
        train_seq = SequenceDataset(
            train_combined_base,
            min_sequence_length=min_sequence_length,
            max_sequence_length=max_sequence_length,
        )
        test_seq = SequenceDataset(
            test_combined_base,
            min_sequence_length=min_sequence_length,
            max_sequence_length=max_sequence_length,
        )

        # Cache the datasets
        print("Caching datasets...")
        with open("train_seq_dataset.pkl", "wb") as f:
            pickle.dump(train_seq, f)
        with open("test_seq_dataset.pkl", "wb") as f:
            pickle.dump(test_seq, f)

        print("Datasets cached successfully!")

    input_dim = train_seq.base_dataset.dataset_1d.dataset.X.shape[-1]

    print("\nSequence Statistics:")
    print(f"  Training sequences: {len(train_seq)}")
    print(f"  Test sequences: {len(test_seq)}")
    print(f"  Input dimension: {input_dim}")

    # Data loaders with custom collate
    train_loader = DataLoader(
        train_seq,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=collate_sequences,
        num_workers=0,  # Set to 0 for debugging, increase for production
    )
    test_loader = DataLoader(
        test_seq,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=collate_sequences,
        num_workers=0,
    )

    # Model
    model = TwoHeadGRU(input_dim=input_dim, hidden_dim=hidden_dim).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = torch.nn.MSELoss()

    print(f"  Model parameters: {sum(p.numel() for p in model.parameters()):,}")

    if save_checkpoints:
        checkpoint_path = Path(checkpoint_dir)
        checkpoint_path.mkdir(exist_ok=True, parents=True)

    best_val_rmse = float("inf")
    current_tf_ratio = teacher_forcing_ratio

    # Training loop
    print("\n" + "=" * 80)
    print("TRAINING ON SEQUENCES")
    print("=" * 80 + "\n")

    for epoch in range(epochs):
        model.train()
        train_loss_sum = 0.0
        train_samples = 0

        for batch_idx, (
            X_seq,
            y_seq,
            node_type,
            node_id,
            timesteps,
            event_id,
            seq_lengths,
        ) in enumerate(train_loader):
            X_seq = X_seq.to(device)  # (batch, max_seq_len, window, features)
            y_seq = y_seq.to(device)  # (batch, max_seq_len)
            node_type = node_type.to(device)
            seq_lengths = seq_lengths.to(device)

            batch_size_actual = X_seq.shape[0]

            # Process each sequence in batch
            batch_loss = 0.0

            for b in range(batch_size_actual):
                seq_len = seq_lengths[b].item()
                if seq_len == 0:
                    continue

                # Get this sequence
                X_current = X_seq[b, 0, :, :].unsqueeze(
                    0
                )  # (1, window, features) - first window
                node_type_b = node_type[b].unsqueeze(0)

                # Predict through the sequence
                for step in range(seq_len):
                    # Make prediction
                    pred = model(X_current, node_type_b)
                    target = y_seq[b, step].unsqueeze(0).unsqueeze(1)

                    # Compute loss
                    loss = criterion(pred, target)
                    batch_loss += loss

                    # Prepare next window
                    if step < seq_len - 1:
                        # Teacher forcing decision
                        use_gt = torch.rand(1).item() < current_tf_ratio

                        # Get next window from data or construct it
                        if step + 1 < X_seq.shape[1]:
                            next_window = X_seq[b, step + 1, :, :].unsqueeze(0)
                        else:
                            # Use current window and slide
                            next_timestep = X_current[:, -1:, :].clone()
                            if use_gt:
                                next_timestep[0, 0, water_level_idx] = target.squeeze()
                            else:
                                next_timestep[0, 0, water_level_idx] = (
                                    pred.squeeze().detach()
                                )
                            next_window = torch.cat(
                                [X_current[:, 1:, :], next_timestep], dim=1
                            )

                        X_current = next_window

                train_samples += seq_len

            # Optimize
            if batch_loss > 0:
                optimizer.zero_grad()
                batch_loss.backward()
                optimizer.step()
                train_loss_sum += batch_loss.item()

        train_rmse = (
            (train_loss_sum / train_samples) ** 0.5 if train_samples > 0 else 0.0
        )

        # Validation
        model.eval()
        val_loss_sum = 0.0
        val_samples = 0

        with torch.no_grad():
            for (
                X_seq,
                y_seq,
                node_type,
                node_id,
                timesteps,
                event_id,
                seq_lengths,
            ) in test_loader:
                X_seq = X_seq.to(device)
                y_seq = y_seq.to(device)
                node_type = node_type.to(device)
                seq_lengths = seq_lengths.to(device)

                batch_size_actual = X_seq.shape[0]

                for b in range(batch_size_actual):
                    seq_len = seq_lengths[b].item()
                    if seq_len == 0:
                        continue

                    X_current = X_seq[b, 0, :, :].unsqueeze(0)
                    node_type_b = node_type[b].unsqueeze(0)

                    for step in range(seq_len):
                        pred = model(X_current, node_type_b)
                        target = y_seq[b, step].unsqueeze(0).unsqueeze(1)
                        loss = criterion(pred, target)
                        val_loss_sum += loss.item()

                        if step < seq_len - 1:
                            next_timestep = X_current[:, -1:, :].clone()
                            next_timestep[0, 0, water_level_idx] = pred.squeeze()
                            X_current = torch.cat(
                                [X_current[:, 1:, :], next_timestep], dim=1
                            )

                    val_samples += seq_len

        val_rmse = (val_loss_sum / val_samples) ** 0.5 if val_samples > 0 else 0.0

        print(
            f"Epoch {epoch + 1:03d}/{epochs} [TF: {current_tf_ratio:.3f}] Train RMSE: {train_rmse:.4f}, Val RMSE: {val_rmse:.4f}"
        )

        # Save best model
        if save_checkpoints and val_rmse < best_val_rmse:
            best_val_rmse = val_rmse
            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "val_rmse": val_rmse,
                },
                checkpoint_path / "best_model_seq_ar.pt",
            )
            print(f"  → Saved best model (RMSE: {val_rmse:.4f})")

        current_tf_ratio *= teacher_forcing_decay

    print("\n" + "=" * 80)
    print(f"TRAINING COMPLETE - Best Val RMSE: {best_val_rmse:.4f}")
    print("=" * 80)

    return model, train_seq, test_seq, trained_normalizer_1d, trained_normalizer_2d


def predict_autoregressive_sequence(
    model, X_init, node_type, num_steps, water_level_idx=0, device="cuda"
):
    """
    Generate autoregressive predictions for multiple timesteps.
    MAINTAINS CONSTANT WINDOW SIZE throughout prediction.

    This performs TRUE sequential autoregressive prediction:
    - Start with initial window of real data
    - Predict step 1 → update window with prediction
    - Predict step 2 using window containing step 1 prediction → update window
    - Continue for num_steps

    Each prediction depends on ALL previous predictions!

    Args:
        model: Trained TwoHeadGRU model
        X_init: Initial window of data (batch, window, features)
        node_type: Node type indicators (batch,)
        num_steps: Number of future steps to predict
        water_level_idx: Index of water level in features
        device: Device to run on

    Returns:
        predictions: Tensor of shape (batch, num_steps) with all predictions
        final_window: Final window state after all predictions
    """
    model.eval()
    batch_size = X_init.shape[0]
    window_size = X_init.shape[1]

    X_current = X_init.to(device).clone()
    node_type = node_type.to(device)
    predictions = []

    with torch.no_grad():
        for step in range(num_steps):
            # Predict next water level using current window
            pred = model(X_current, node_type)  # (batch, 1)
            predictions.append(pred)

            # Create next timestep: copy last timestep and update water level
            next_timestep = X_current[:, -1:, :].clone()
            next_timestep[:, 0, water_level_idx] = pred.squeeze(-1)

            # Slide window: remove oldest, add newest
            X_current = torch.cat([X_current[:, 1:, :], next_timestep], dim=1)

            assert X_current.shape[1] == window_size

    predictions = torch.cat(predictions, dim=1)  # (batch, num_steps)
    return predictions, X_current


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
    assert test_1d.dataset.X.shape[-1] == input_dim, (
        f"1D feature mismatch! Dataset has {test_1d.dataset.X.shape[-1]}, model expects {input_dim}"
    )
    assert test_2d.dataset.X.shape[-1] == input_dim, (
        f"2D feature mismatch! Dataset has {test_2d.dataset.X.shape[-1]}, model expects {input_dim}"
    )

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
    metrics = evaluate_predictions_hierarchical(save_path)

    print("\n" + "=" * 80)
    print("COMPLETE")
    print("=" * 80)

    return pred_df, metrics


if __name__ == "__main__":
    model_name = "Model1"
    train_events = Path(f"data/{model_name}/processed/features_csv/train/")
    test_events = Path(f"data/{model_name}/processed/features_csv/test_edited2/")

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
        water_level_idx = FEATURE_NAMES.index("water_level")
        new_normalizer_1d = SequenceNormalizer()
        new_normalizer_2d = SequenceNormalizer()
        model, train_ds, test_ds, trained_normalizer_1d, trained_normalizer_2d = (
            train_autoregressive(
                train_events,
                test_events,
                window=5,
                max_events=None,
                normalizer_1d=new_normalizer_1d,
                normalizer_2d=new_normalizer_2d,
                epochs=1,
                lr=1e-3,
                batch_size=32,
                hidden_dim=128,
                device="cuda" if torch.cuda.is_available() else "cpu",
                save_checkpoints=True,
                checkpoint_dir="./two_head_checkpoints",
                water_level_idx=water_level_idx,
            )
        )

        test_save_path = "gru_test_predictions.csv"

        pred_df = save_predictions_autoregressive(
            model=model,
            dataset=test_ds,
            normalizer_1d=trained_normalizer_1d,
            normalizer_2d=trained_normalizer_2d,
            save_path=test_save_path,
            water_level_idx=water_level_idx,
            initial_ground_truth_steps=5,
            max_ar_steps=None,
            use_sequence_dataset=True,
        )

        # Evaluate
        metrics = evaluate_predictions_hierarchical(test_save_path)

    test_only = False
    if test_only:
        load_model_and_predict(
            max_events=1,
            checkpoint_path="./two_head_checkpoints/best_model_overall_nse.pt",
            test_events=test_events,
            normalizer_1d_path="train_normalizer_two_head_1d.pkl",
            normalizer_2d_path="train_normalizer_two_head_2d.pkl",
            device="cuda" if torch.cuda.is_available() else "cpu",
            save_path="gru_test_predictions_test.csv",
        )
