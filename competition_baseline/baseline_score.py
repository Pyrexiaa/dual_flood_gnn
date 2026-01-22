from typing import List, Union
import pandas as pd
import numpy as np
from pathlib import Path
from collections import defaultdict


def nse(y_true, y_pred):
    """Calculate Nash-Sutcliffe Efficiency"""
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)

    denominator = np.sum((y_true - np.mean(y_true)) ** 2)

    # Safety check (important for dry or constant series)
    if denominator == 0:
        return np.nan

    return 1 - np.sum((y_true - y_pred) ** 2) / denominator


def calculate_event_nse(event_df, y_pred_col="water_level_pred"):
    """
    Calculate hierarchical NSE for a single event

    Args:
        event_df: DataFrame for one event with columns: node_type, node_id, water_level (truth), water_level_pred
        y_pred_col: Name of prediction column

    Returns:
        float: Event NSE score
    """
    node_types = event_df["node_type"].values
    node_ids = event_df["node_id"].values
    y_true = event_df["water_level"].values
    y_pred = event_df[y_pred_col].values

    # Find unique 1D and 2D nodes
    mask_1d = node_types == 1
    mask_2d = node_types == 2

    unique_1d_nodes = np.unique(node_ids[mask_1d]) if mask_1d.any() else np.array([])
    unique_2d_nodes = np.unique(node_ids[mask_2d]) if mask_2d.any() else np.array([])

    # Calculate NSE for 1D nodes (predict 0D nodes)
    nse_1d_list = []
    for node_id in unique_1d_nodes:
        mask = (node_ids == node_id) & (node_types == 1)
        if np.sum(mask) > 1:
            node_nse = nse(y_true[mask], y_pred[mask])
            if not np.isnan(node_nse):
                nse_1d_list.append(node_nse)

    # Calculate NSE for 2D nodes (predict 1D nodes)
    nse_2d_list = []
    for node_id in unique_2d_nodes:
        mask = (node_ids == node_id) & (node_types == 2)
        if np.sum(mask) > 1:
            node_nse = nse(y_true[mask], y_pred[mask])
            if not np.isnan(node_nse):
                nse_2d_list.append(node_nse)

    # Average 1D and 2D NSEs
    nse_1d_avg = np.mean(nse_1d_list) if nse_1d_list else np.nan
    nse_2d_avg = np.mean(nse_2d_list) if nse_2d_list else np.nan
    valid_nse = [x for x in [nse_1d_avg, nse_2d_avg] if not np.isnan(x)]

    return np.mean(valid_nse) if valid_nse else np.nan


def calculate_nse_score(solution_df, prediction_df):
    """
    Calculate hierarchical NSE score for entire submission

    Args:
        solution_df: Solution DataFrame with water_level (ground truth)
        prediction_df: Prediction DataFrame with water_level (predictions)

    Returns:
        float: Final NSE score (average across all models)
    """
    # Merge solution and prediction
    key_cols = ["model_id", "event_id", "node_type", "node_id"]

    # Sort both dataframes
    solution_sorted = solution_df.sort_values(key_cols).reset_index(drop=True)
    prediction_sorted = prediction_df.sort_values(key_cols).reset_index(drop=True)

    # Validate they match
    for col in key_cols:
        if not solution_sorted[col].equals(prediction_sorted[col]):
            raise ValueError(f"Mismatch in column: {col}")

    # Now merge
    merged = solution_sorted.copy()
    merged["water_level_pred"] = prediction_sorted["water_level"].values

    # Calculate NSE per event, then average per model
    model_event_nses = defaultdict(list)

    for (model_id, event_id), event_df in merged.groupby(["model_id", "event_id"]):
        event_nse = calculate_event_nse(event_df)

        if not np.isnan(event_nse):
            model_event_nses[model_id].append(event_nse)

    # Calculate model scores
    model_scores = []
    for model_id in sorted(model_event_nses.keys()):
        event_nses = model_event_nses[model_id]
        if event_nses:
            model_score = np.mean(event_nses)
            model_scores.append(model_score)

    # Final score is average across models
    if model_scores:
        return np.mean(model_scores)
    else:
        return np.nan


def calculate_training_std(model1_train_path: str, model2_train_path: str):
    """
    Calculate standard deviation of water levels from training data for each model and node type

    Args:
        model1_train_path: Path to model 1 training data directory (contains event_* folders)
        model2_train_path: Path to model 2 training data directory (contains event_* folders)

    Returns:
        dict: Dictionary with standard deviations for each model and node type
            Keys: 'model_1_node_1', 'model_1_node_2', 'model_2_node_1', 'model_2_node_2'
    """

    def get_std_for_model_path(model_path, node_type):
        """Get std for a specific model path and node type"""
        model_path = Path(model_path)
        all_water_levels = []

        # Find all event_* directories
        event_dirs = sorted([d for d in model_path.glob("event_*") if d.is_dir()])

        if not event_dirs:
            raise ValueError(f"No event_* directories found in {model_path}")

        # Determine filename based on node type
        if node_type == 1:
            filename = "1d_nodes_dynamic_all.csv"
        elif node_type == 2:
            filename = "2d_nodes_dynamic_all.csv"
        else:
            raise ValueError(f"Invalid node_type: {node_type}. Must be 1 or 2.")

        print(f"  Processing {len(event_dirs)} events for node type {node_type}...")

        # Collect water levels from all events
        for event_dir in event_dirs:
            csv_path = event_dir / filename

            if csv_path.exists():
                df = pd.read_csv(csv_path)

                # Check if water_level column exists
                if "water_level" in df.columns:
                    water_levels = df["water_level"].values
                    # Filter out NaN values
                    water_levels = water_levels[~np.isnan(water_levels)]
                    all_water_levels.extend(water_levels)
                else:
                    print(f"    Warning: 'water_level' column not found in {csv_path}")
            else:
                print(f"    Warning: {filename} not found in {event_dir}")

        if not all_water_levels:
            raise ValueError(
                f"No water level data found for node type {node_type} in {model_path}"
            )

        # Calculate standard deviation
        std_value = np.std(all_water_levels)
        print(
            f"    Found {len(all_water_levels):,} water level values, std = {std_value:.6f}"
        )

        return std_value

    print("=" * 80)
    print("CALCULATING TRAINING DATA STANDARD DEVIATIONS")
    print("=" * 80)

    results = {}

    # Model 1
    print(f"\nModel 1: {model1_train_path}")
    print("  Node Type 1 (1D - Surface):")
    results["model_1_node_1"] = get_std_for_model_path(model1_train_path, 1)

    print("  Node Type 2 (2D - Subsurface):")
    results["model_1_node_2"] = get_std_for_model_path(model1_train_path, 2)

    # Model 2
    print(f"\nModel 2: {model2_train_path}")
    print("  Node Type 1 (1D - Surface):")
    results["model_2_node_1"] = get_std_for_model_path(model2_train_path, 1)

    print("  Node Type 2 (2D - Subsurface):")
    results["model_2_node_2"] = get_std_for_model_path(model2_train_path, 2)

    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)
    for key, value in results.items():
        print(f"{key}: {value:.6f}")
    print("=" * 80 + "\n")

    return results


def calculate_rmse(y_true, y_pred):
    """Calculate Root Mean Squared Error."""
    return np.sqrt(np.mean((y_true - y_pred) ** 2))


def standardized_rmse(y_true, y_pred, std_dev):
    """Calculate RMSE standardized by provided standard deviation"""
    if std_dev == 0 or np.isnan(std_dev):
        return np.nan

    rmse_val = calculate_rmse(y_true, y_pred)
    return rmse_val / std_dev


def calculate_event_std_rmse(event_df_gt, event_df_pred, model_id, training_std):
    """
    Calculate standardized RMSE for a single event

    Args:
        event_df_gt: Ground truth DataFrame for one event
        event_df_pred: Prediction DataFrame for one event
        model_id: Model ID for this event
        training_std: Dictionary with training standard deviations

    Returns:
        Event-level standardized RMSE (average across node types)
    """
    node_type_rmse_scores = []

    for node_type in sorted(event_df_gt["node_type"].unique()):
        gt_node = event_df_gt[event_df_gt["node_type"] == node_type]
        pred_node = event_df_pred[event_df_pred["node_type"] == node_type]

        if len(gt_node) > 1:  # Need at least 2 points to calculate meaningful RMSE
            # Sort to ensure alignment
            key_cols = ["node_id"]
            gt_node_sorted = gt_node.sort_values(key_cols).reset_index(drop=True)
            pred_node_sorted = pred_node.sort_values(key_cols).reset_index(drop=True)

            y_true = gt_node_sorted["water_level"].values
            y_pred = pred_node_sorted["water_level"].values

            # Get std_dev for standardization
            std_key = f"model_{model_id}_node_{node_type}"
            std_value = training_std.get(std_key)

            if std_value is not None and std_value > 0:
                node_std_rmse = standardized_rmse(y_true, y_pred, std_value)
                if not np.isnan(node_std_rmse):
                    node_type_rmse_scores.append(node_std_rmse)

    # Average across node types for this event
    return np.mean(node_type_rmse_scores) if node_type_rmse_scores else np.nan


def calculate_hierarchical_rmse(gt_filtered, sol_filtered, training_std):
    """
    Calculate hierarchical standardized RMSE following the same structure as NSE

    1. For each event, calculate standardized RMSE per node, then average
    2. Average event scores for each model
    3. Average model scores for final score
    """
    model_event_rmses = {}

    for model_id in sorted(gt_filtered["model_id"].unique()):
        gt_model = gt_filtered[gt_filtered["model_id"] == model_id]
        sol_model = sol_filtered[sol_filtered["model_id"] == model_id]

        event_rmses = []

        for event_id in sorted(gt_model["event_id"].unique()):
            gt_event = gt_model[gt_model["event_id"] == event_id]
            sol_event = sol_model[sol_model["event_id"] == event_id]

            event_rmse = calculate_event_std_rmse(
                gt_event, sol_event, model_id, training_std
            )

            if not np.isnan(event_rmse):
                event_rmses.append(event_rmse)

        # Average across events for this model
        if event_rmses:
            model_event_rmses[model_id] = np.mean(event_rmses)
        else:
            model_event_rmses[model_id] = np.nan

    # Average across models for final score
    model_scores = [
        score for score in model_event_rmses.values() if not np.isnan(score)
    ]
    final_rmse = np.mean(model_scores) if model_scores else np.nan

    return final_rmse, model_event_rmses


def save_baseline_parquet(baseline_df, baseline_name, output_dir):
    """
    Save baseline predictions to parquet file

    Args:
        baseline_df: DataFrame with baseline predictions
        baseline_name: Name of the baseline method
        output_dir: Directory to save the file

    Returns:
        Path: Path to saved file
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True)

    output_path = output_dir / f"{baseline_name}.parquet"
    baseline_df.to_parquet(output_path, index=False)

    return output_path


def generate_comprehensive_baselines(
    solution_path, output_dir="baselines", random_seed=42, save_parquet=True
):
    """
    Generate comprehensive baseline submissions and calculate their NSE scores.

    Baselines include:
    - Naive: all zeros, per-event random, per-event mean
    - Location-aware: per-node random, per-node mean, per-node-type random
    - Temporal: persistence t10, linear trend
    - Hierarchical: per-model mean

    Args:
        solution_path: Path to solution file (CSV or Parquet)
        output_dir: Directory to save baseline results
        random_seed: Random seed for reproducibility
        save_parquet: If True, save each baseline as parquet file

    Returns:
        dict: Dictionary with baseline names and their NSE scores
    """
    np.random.seed(random_seed)

    # Load solution
    solution_path = Path(solution_path)
    print(f"Loading solution from: {solution_path}")

    if solution_path.suffix.lower() == ".parquet":
        solution = pd.read_parquet(solution_path)
    else:
        solution = pd.read_csv(solution_path)

    print(f"Loaded {len(solution):,} rows")
    print(f"Columns: {list(solution.columns)}")

    # Create output directory
    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True)

    # Required columns for submission
    required_cols = ["model_id", "event_id", "node_type", "node_id", "water_level"]

    # Check if solution has row_id
    has_row_id = "row_id" in solution.columns
    if has_row_id:
        submission_cols = ["row_id"] + required_cols
    else:
        submission_cols = required_cols

    results = {}

    print("\n" + "=" * 80)
    print("GENERATING COMPREHENSIVE BASELINES")
    print("=" * 80)

    # ============================================================================
    # NAIVE BASELINES
    # ============================================================================
    print("\n" + "=" * 80)
    print("NAIVE BASELINES")
    print("=" * 80)

    # 1. All Zeros
    print("\n[1/12] All Zeros")
    baseline = solution[submission_cols].copy()
    baseline["water_level"] = 0.0
    try:
        score = calculate_nse_score(solution, baseline)
        results["naive_all_zeros"] = score
        print(f"  NSE Score: {score:.6f}")
        if save_parquet:
            path = save_baseline_parquet(baseline, "naive_all_zeros", output_dir)
            print(f"  Saved to: {path}")
    except Exception as e:
        print(f"  Error: {e}")
        results["naive_all_zeros"] = None

    # 2. Random Per-Event
    print("\n[2/12] Random Per-Event (uniform within event min-max)")
    baseline = solution[submission_cols].copy()

    def random_per_event(group):
        min_val = group["water_level"].min()
        max_val = group["water_level"].max()
        if min_val == max_val:
            max_val = min_val + 0.01
        group["water_level"] = np.random.uniform(min_val, max_val, size=len(group))
        return group

    baseline = baseline.groupby(["model_id", "event_id"], group_keys=False).apply(
        random_per_event
    )
    try:
        score = calculate_nse_score(solution, baseline)
        results["naive_random_per_event"] = score
        print(f"  NSE Score: {score:.6f}")
        if save_parquet:
            path = save_baseline_parquet(baseline, "naive_random_per_event", output_dir)
            print(f"  Saved to: {path}")
    except Exception as e:
        print(f"  Error: {e}")
        results["naive_random_per_event"] = None

    # 3. Mean Per-Event
    print("\n[3/12] Mean Per-Event")
    baseline = solution[submission_cols].copy()

    def mean_per_event(group):
        mean_val = group["water_level"].mean()
        group["water_level"] = mean_val
        return group

    baseline = baseline.groupby(["model_id", "event_id"], group_keys=False).apply(
        mean_per_event
    )
    try:
        score = calculate_nse_score(solution, baseline)
        results["naive_mean_per_event"] = score
        print(f"  NSE Score: {score:.6f}")
        if save_parquet:
            path = save_baseline_parquet(baseline, "naive_mean_per_event", output_dir)
            print(f"  Saved to: {path}")
    except Exception as e:
        print(f"  Error: {e}")
        results["naive_mean_per_event"] = None

    # ============================================================================
    # LOCATION-AWARE BASELINES
    # ============================================================================
    print("\n" + "=" * 80)
    print("LOCATION-AWARE BASELINES")
    print("=" * 80)

    # 4. Random Per-Node
    print("\n[4/12] Random Per-Node (uniform within node min-max)")
    baseline = solution[submission_cols].copy()

    def random_per_node(group):
        min_val = group["water_level"].min()
        max_val = group["water_level"].max()
        if min_val == max_val:
            max_val = min_val + 0.01
        group["water_level"] = np.random.uniform(min_val, max_val, size=len(group))
        return group

    baseline = baseline.groupby(
        ["model_id", "event_id", "node_id"], group_keys=False
    ).apply(random_per_node)
    try:
        score = calculate_nse_score(solution, baseline)
        results["location_random_per_node"] = score
        print(f"  NSE Score: {score:.6f}")
        if save_parquet:
            path = save_baseline_parquet(
                baseline, "location_random_per_node", output_dir
            )
            print(f"  Saved to: {path}")
    except Exception as e:
        print(f"  Error: {e}")
        results["location_random_per_node"] = None

    # 5. Mean Per-Node
    print("\n[5/12] Mean Per-Node")
    baseline = solution[submission_cols].copy()

    def mean_per_node(group):
        mean_val = group["water_level"].mean()
        group["water_level"] = mean_val
        return group

    baseline = baseline.groupby(
        ["model_id", "event_id", "node_id"], group_keys=False
    ).apply(mean_per_node)
    try:
        score = calculate_nse_score(solution, baseline)
        results["location_mean_per_node"] = score
        print(f"  NSE Score: {score:.6f}")
        if save_parquet:
            path = save_baseline_parquet(baseline, "location_mean_per_node", output_dir)
            print(f"  Saved to: {path}")
    except Exception as e:
        print(f"  Error: {e}")
        results["location_mean_per_node"] = None

    # 6. Random Per-Node-Type
    print("\n[6/12] Random Per-Node-Type (separate ranges for 1D vs 2D)")
    baseline = solution[submission_cols].copy()

    def random_per_node_type(group):
        min_val = group["water_level"].min()
        max_val = group["water_level"].max()
        if min_val == max_val:
            max_val = min_val + 0.01
        group["water_level"] = np.random.uniform(min_val, max_val, size=len(group))
        return group

    baseline = baseline.groupby(
        ["model_id", "event_id", "node_type"], group_keys=False
    ).apply(random_per_node_type)
    try:
        score = calculate_nse_score(solution, baseline)
        results["location_random_per_node_type"] = score
        print(f"  NSE Score: {score:.6f}")
        if save_parquet:
            path = save_baseline_parquet(
                baseline, "location_random_per_node_type", output_dir
            )
            print(f"  Saved to: {path}")
    except Exception as e:
        print(f"  Error: {e}")
        results["location_random_per_node_type"] = None

    # 7. Mean Per-Node-Type
    print("\n[7/12] Mean Per-Node-Type")
    baseline = solution[submission_cols].copy()

    def mean_per_node_type(group):
        mean_val = group["water_level"].mean()
        group["water_level"] = mean_val
        return group

    baseline = baseline.groupby(
        ["model_id", "event_id", "node_type"], group_keys=False
    ).apply(mean_per_node_type)
    try:
        score = calculate_nse_score(solution, baseline)
        results["location_mean_per_node_type"] = score
        print(f"  NSE Score: {score:.6f}")
        if save_parquet:
            path = save_baseline_parquet(
                baseline, "location_mean_per_node_type", output_dir
            )
            print(f"  Saved to: {path}")
    except Exception as e:
        print(f"  Error: {e}")
        results["location_mean_per_node_type"] = None

    # ============================================================================
    # TEMPORAL BASELINES
    # ============================================================================
    print("\n" + "=" * 80)
    print("TEMPORAL BASELINES")
    print("=" * 80)

    # 8. Persistence (Timestep 10)
    print("\n[8/12] Persistence (repeat timestep 10 value)")
    baseline = solution[submission_cols].copy()

    def persistence_t10(group):
        if len(group) <= 10:
            return group
        timestep_10_value = group.iloc[10]["water_level"]
        group.loc[group.index[11:], "water_level"] = timestep_10_value
        return group

    baseline = baseline.groupby(
        ["model_id", "event_id", "node_id"], group_keys=False
    ).apply(persistence_t10)
    try:
        score = calculate_nse_score(solution, baseline)
        results["temporal_persistence_t10"] = score
        print(f"  NSE Score: {score:.6f}")
        if save_parquet:
            path = save_baseline_parquet(
                baseline, "temporal_persistence_t10", output_dir
            )
            print(f"  Saved to: {path}")
    except Exception as e:
        print(f"  Error: {e}")
        results["temporal_persistence_t10"] = None

    # 9. Last Known Value (Sequential)
    print("\n[9/12] Last Known Value (each timestep = previous timestep)")
    baseline = solution[submission_cols].copy()

    def last_value(group):
        if len(group) > 1:
            # Shift values forward by 1 (each timestep uses previous value)
            group["water_level"] = group["water_level"].shift(1)
            # First timestep keeps its original value
            group.iloc[0, group.columns.get_loc("water_level")] = group.iloc[0][
                "water_level"
            ]
        return group

    baseline = baseline.groupby(
        ["model_id", "event_id", "node_id"], group_keys=False
    ).apply(last_value)
    try:
        score = calculate_nse_score(solution, baseline)
        results["temporal_last_value"] = score
        print(f"  NSE Score: {score:.6f}")
        if save_parquet:
            path = save_baseline_parquet(baseline, "temporal_last_value", output_dir)
            print(f"  Saved to: {path}")
    except Exception as e:
        print(f"  Error: {e}")
        results["temporal_last_value"] = None

    # 10. Linear Trend (extrapolate from timesteps 0-10)
    print("\n[10/12] Linear Trend (fit line on timesteps 0-10, extrapolate)")
    baseline = solution[submission_cols].copy()

    def linear_trend(group):
        if len(group) <= 10:
            return group
        # Fit linear trend on first 11 timesteps (0-10)
        x = np.arange(11)
        y = group.iloc[:11]["water_level"].values
        # Simple linear regression
        coeffs = np.polyfit(x, y, 1)
        slope, intercept = coeffs[0], coeffs[1]
        # Extrapolate for all timesteps
        x_all = np.arange(len(group))
        group["water_level"] = slope * x_all + intercept
        return group

    baseline = baseline.groupby(
        ["model_id", "event_id", "node_id"], group_keys=False
    ).apply(linear_trend)
    try:
        score = calculate_nse_score(solution, baseline)
        results["temporal_linear_trend"] = score
        print(f"  NSE Score: {score:.6f}")
        if save_parquet:
            path = save_baseline_parquet(baseline, "temporal_linear_trend", output_dir)
            print(f"  Saved to: {path}")
    except Exception as e:
        print(f"  Error: {e}")
        results["temporal_linear_trend"] = None

    # ============================================================================
    # HIERARCHICAL BASELINES
    # ============================================================================
    print("\n" + "=" * 80)
    print("HIERARCHICAL BASELINES")
    print("=" * 80)

    # 11. Mean Per-Model
    print("\n[11/12] Mean Per-Model")
    baseline = solution[submission_cols].copy()

    def mean_per_model(group):
        mean_val = group["water_level"].mean()
        group["water_level"] = mean_val
        return group

    baseline = baseline.groupby(["model_id"], group_keys=False).apply(mean_per_model)
    try:
        score = calculate_nse_score(solution, baseline)
        results["hierarchical_mean_per_model"] = score
        print(f"  NSE Score: {score:.6f}")
        if save_parquet:
            path = save_baseline_parquet(
                baseline, "hierarchical_mean_per_model", output_dir
            )
            print(f"  Saved to: {path}")
    except Exception as e:
        print(f"  Error: {e}")
        results["hierarchical_mean_per_model"] = None

    # 12. Global Mean
    print("\n[12/12] Global Mean (single mean for all predictions)")
    baseline = solution[submission_cols].copy()
    global_mean = solution["water_level"].mean()
    baseline["water_level"] = global_mean
    try:
        score = calculate_nse_score(solution, baseline)
        results["hierarchical_global_mean"] = score
        print(f"  NSE Score: {score:.6f}")
        if save_parquet:
            path = save_baseline_parquet(
                baseline, "hierarchical_global_mean", output_dir
            )
            print(f"  Saved to: {path}")
    except Exception as e:
        print(f"  Error: {e}")
        results["hierarchical_global_mean"] = None

    # ============================================================================
    # SUMMARY & SAVE RESULTS
    # ============================================================================
    print("\n" + "=" * 80)
    print("BASELINE RESULTS SUMMARY")
    print("=" * 80)

    # Organize results by category
    categories = {
        "Naive": ["naive_all_zeros", "naive_random_per_event", "naive_mean_per_event"],
        "Location-Aware": [
            "location_random_per_node",
            "location_mean_per_node",
            "location_random_per_node_type",
            "location_mean_per_node_type",
        ],
        "Temporal": [
            "temporal_persistence_t10",
            "temporal_last_value",
            "temporal_linear_trend",
        ],
        "Hierarchical": ["hierarchical_mean_per_model", "hierarchical_global_mean"],
    }

    for category, baseline_names in categories.items():
        print(f"\n{category}:")
        for name in baseline_names:
            if name in results:
                score = results[name]
                if score is not None:
                    print(f"  {name:35s}: {score:8.6f}")
                else:
                    print(f"  {name:35s}: ERROR")

    # Save results to CSV
    results_df = pd.DataFrame(
        [
            {
                "baseline_method": name,
                "nse_score": score,
                "category": next(
                    (cat for cat, names in categories.items() if name in names), "Other"
                ),
            }
            for name, score in results.items()
        ]
    )
    results_df = results_df.sort_values("nse_score", ascending=False)

    results_path = output_dir / "comprehensive_baseline_results_rounded.csv"
    results_df.to_csv(results_path, index=False)

    print(f"\n{'=' * 80}")
    print(f"Results saved to: {results_path.absolute()}")

    if save_parquet:
        print(f"Parquet files saved to: {output_dir.absolute()}")

    # Find best baseline
    valid_results = [
        (name, score) for name, score in results.items() if score is not None
    ]
    if valid_results:
        best_name, best_score = max(valid_results, key=lambda x: x[1])
        print(f"\nBest Baseline: {best_name}")
        print(f"Best NSE Score: {best_score:.6f}")

    print("=" * 80 + "\n")

    return results


def parquet_to_csv(parquet_path):
    parquet_path = Path(parquet_path)
    csv_path = parquet_path.with_suffix(".csv")

    df = pd.read_parquet(parquet_path)
    df.to_csv(csv_path, index=False)

    return csv_path


def evaluate_single_solution(
    ground_truth,
    solution,
    usage_filter=None,
    training_std=None,
    metrics=["nse", "rmse"],
):
    """
    Evaluate a single solution against ground truth

    Args:
        ground_truth: Ground truth DataFrame
        solution: Solution DataFrame
        usage_filter: 'public', 'private', or None for all data
        training_std: Dictionary with training standard deviations for standardized RMSE
                     Keys: 'model_1_node_1', 'model_1_node_2', 'model_2_node_1', 'model_2_node_2'
                     If None, uses regular RMSE (simple average across all predictions)
        metrics: List of metrics to calculate. Options: 'nse', 'rmse', or both ['nse', 'rmse']
                 Default: ['nse', 'rmse']

    Returns:
        dict: Dictionary with requested metric scores (overall and broken down by model/node_type)
    """
    # Normalize metrics parameter
    if isinstance(metrics, str):
        metrics = [metrics]
    metrics = [m.lower() for m in metrics]

    # Validate metrics
    valid_metrics = {"nse", "rmse"}
    if not all(m in valid_metrics for m in metrics):
        raise ValueError(
            f"Invalid metrics. Must be 'nse', 'rmse', or both. Got: {metrics}"
        )

    calculate_nse_bool = "nse" in metrics
    calculate_rmse_bool = "rmse" in metrics

    # Filter by usage if specified
    if usage_filter:
        if "Usage" not in ground_truth.columns:
            raise ValueError("Ground truth does not have 'Usage' column")

        usage_filter = usage_filter.lower()
        gt_filtered = ground_truth[
            ground_truth["Usage"].str.lower() == usage_filter
        ].copy()

        # Filter solution to match
        key_cols = ["model_id", "event_id", "node_type", "node_id"]

        # Create a merge key
        gt_filtered["_merge_key"] = (
            gt_filtered["model_id"].astype(str)
            + "_"
            + gt_filtered["event_id"].astype(str)
            + "_"
            + gt_filtered["node_type"].astype(str)
            + "_"
            + gt_filtered["node_id"].astype(str)
        )
        solution["_merge_key"] = (
            solution["model_id"].astype(str)
            + "_"
            + solution["event_id"].astype(str)
            + "_"
            + solution["node_type"].astype(str)
            + "_"
            + solution["node_id"].astype(str)
        )

        sol_filtered = solution[
            solution["_merge_key"].isin(gt_filtered["_merge_key"])
        ].copy()

        # Clean up merge keys
        gt_filtered = gt_filtered.drop(columns=["_merge_key"])
        sol_filtered = sol_filtered.drop(columns=["_merge_key"])

    else:
        gt_filtered = ground_truth.copy()
        sol_filtered = solution.copy()

    # Initialize results dictionary
    results = {
        "num_predictions": len(sol_filtered),
    }

    # Calculate hierarchical NSE score (overall) if requested
    if calculate_nse_bool:
        nse_score = calculate_nse_score(gt_filtered, sol_filtered)
        results["nse_score"] = nse_score

    # Calculate hierarchical RMSE score (overall) if requested
    if calculate_rmse_bool:
        if training_std is None:
            # Regular RMSE - simple average across all predictions
            key_cols = ["model_id", "event_id", "node_type", "node_id"]
            gt_sorted = gt_filtered.sort_values(key_cols).reset_index(drop=True)
            sol_sorted = sol_filtered.sort_values(key_cols).reset_index(drop=True)

            y_true = gt_sorted["water_level"].values
            y_pred = sol_sorted["water_level"].values
            rmse_score = calculate_rmse(y_true, y_pred)
        else:
            # Hierarchical standardized RMSE (event-based like NSE)
            rmse_score, model_event_rmses = calculate_hierarchical_rmse(
                gt_filtered, sol_filtered, training_std
            )

        results["rmse_score"] = rmse_score

    # Calculate detailed metrics by model and node type
    key_cols = ["model_id", "event_id", "node_type", "node_id"]

    for model_id in sorted(gt_filtered["model_id"].unique()):
        # Filter for this model
        gt_model = gt_filtered[gt_filtered["model_id"] == model_id]
        sol_model = sol_filtered[sol_filtered["model_id"] == model_id]

        # Overall metrics for this model
        if len(gt_model) > 0:
            # NSE for model
            if calculate_nse_bool:
                model_nse = calculate_nse_score(gt_model, sol_model)
                results[f"nse_model_{model_id}_overall"] = model_nse

            # RMSE for model
            if calculate_rmse_bool:
                if training_std is None:
                    # Regular RMSE for model
                    gt_model_sorted = gt_model.sort_values(key_cols).reset_index(
                        drop=True
                    )
                    sol_model_sorted = sol_model.sort_values(key_cols).reset_index(
                        drop=True
                    )

                    model_rmse = calculate_rmse(
                        gt_model_sorted["water_level"].values,
                        sol_model_sorted["water_level"].values,
                    )
                else:
                    # Hierarchical standardized RMSE for model
                    model_rmse, _ = calculate_hierarchical_rmse(
                        gt_model, sol_model, training_std
                    )

                results[f"rmse_model_{model_id}_overall"] = model_rmse

        # Metrics by node type for this model
        for node_type in sorted(gt_model["node_type"].unique()):
            gt_node = gt_model[gt_model["node_type"] == node_type]
            sol_node = sol_model[sol_model["node_type"] == node_type]

            if len(gt_node) > 0:
                # NSE for node type
                if calculate_nse_bool:
                    node_nse = calculate_nse_score(gt_node, sol_node)
                    results[f"nse_model_{model_id}_node_{node_type}"] = node_nse

                # RMSE for node type
                if calculate_rmse_bool:
                    gt_node_sorted = gt_node.sort_values(key_cols).reset_index(
                        drop=True
                    )
                    sol_node_sorted = sol_node.sort_values(key_cols).reset_index(
                        drop=True
                    )

                    if training_std is None:
                        # Regular RMSE
                        y_true_node = gt_node_sorted["water_level"].values
                        y_pred_node = sol_node_sorted["water_level"].values
                        node_rmse = calculate_rmse(y_true_node, y_pred_node)
                    else:
                        # Hierarchical standardized RMSE for this node type
                        # Calculate per-event RMSE for this specific node type
                        event_rmses = []

                        for event_id in sorted(gt_node["event_id"].unique()):
                            gt_event = gt_node[gt_node["event_id"] == event_id]
                            sol_event = sol_node[sol_node["event_id"] == event_id]

                            if len(gt_event) > 1:
                                gt_event_sorted = gt_event.sort_values(
                                    ["node_id"]
                                ).reset_index(drop=True)
                                sol_event_sorted = sol_event.sort_values(
                                    ["node_id"]
                                ).reset_index(drop=True)

                                y_true = gt_event_sorted["water_level"].values
                                y_pred = sol_event_sorted["water_level"].values

                                std_key = f"model_{model_id}_node_{node_type}"
                                std_value = training_std.get(std_key)

                                if std_value is not None and std_value > 0:
                                    event_rmse = standardized_rmse(
                                        y_true, y_pred, std_value
                                    )
                                    if not np.isnan(event_rmse):
                                        event_rmses.append(event_rmse)

                        node_rmse = np.mean(event_rmses) if event_rmses else np.nan

                    results[f"rmse_model_{model_id}_node_{node_type}"] = node_rmse

    return results


def evaluate_and_rank_solutions(
    solution_paths: List[Union[str, Path]],
    ground_truth_path: Union[str, Path],
    output_csv: str = "baselines/solution_rankings.csv",
    solution_names: List[str] = None,
    split_by_usage: bool = True,
    training_std: dict = None,
    metrics: List[str] = ["nse", "rmse"],
    model1_train_path: str = None,
    model2_train_path: str = None,
):
    """
    Evaluate multiple solution files against ground truth, calculate NSE and/or RMSE,
    rank them, and save results to CSV. Optionally split by public/private usage.

    Args:
        solution_paths: List of paths to solution files (CSV or Parquet)
        ground_truth_path: Path to ground truth file (CSV or Parquet)
        output_csv: Path to save the ranked results CSV
        solution_names: Optional list of names for each solution. If None, uses filenames.
        split_by_usage: If True, evaluate separately for public and private data
        training_std: Dictionary with training standard deviations for standardized RMSE.
                     If None and model1_train_path/model2_train_path provided, will calculate.
        metrics: List of metrics to calculate. Options: 'nse', 'rmse', or both ['nse', 'rmse']
                 Default: ['nse', 'rmse']
        model1_train_path: Path to model 1 training data (for calculating training_std)
        model2_train_path: Path to model 2 training data (for calculating training_std)

    Returns:
        If split_by_usage=False: pd.DataFrame with overall results
        If split_by_usage=True: tuple of (overall_df, public_df, private_df)
    """

    # Normalize metrics parameter
    if isinstance(metrics, str):
        metrics = [metrics]
    metrics = [m.lower() for m in metrics]

    # Validate metrics
    valid_metrics = {"nse", "rmse"}
    if not all(m in valid_metrics for m in metrics):
        raise ValueError(
            f"Invalid metrics. Must be 'nse', 'rmse', or both. Got: {metrics}"
        )

    calculate_nse_bool = "nse" in metrics
    calculate_rmse_bool = "rmse" in metrics

    # Load ground truth
    ground_truth_path = Path(ground_truth_path)
    print(f"Loading ground truth from: {ground_truth_path}")

    if ground_truth_path.suffix.lower() == ".parquet":
        ground_truth = pd.read_parquet(ground_truth_path)
    else:
        ground_truth = pd.read_csv(ground_truth_path)

    print(f"Ground truth loaded: {len(ground_truth):,} rows")

    # Check if usage column exists
    has_usage = "Usage" in ground_truth.columns
    if split_by_usage and not has_usage:
        print(
            "WARNING: 'Usage' column not found in ground truth. Will only evaluate overall."
        )
        split_by_usage = False

    if has_usage:
        usage_counts = ground_truth["Usage"].value_counts()
        print("Usage breakdown:")
        for usage, count in usage_counts.items():
            print(f"  {usage}: {count:,} rows ({count / len(ground_truth) * 100:.1f}%)")
    print()

    # If solution_names not provided, use filenames
    if solution_names is None:
        solution_names = [Path(p).stem for p in solution_paths]

    # Validate that we have the same number of names and paths
    if len(solution_names) != len(solution_paths):
        raise ValueError(
            f"Number of solution_names ({len(solution_names)}) must match number of solution_paths ({len(solution_paths)})"
        )

    # Calculate or validate training_std if RMSE is requested
    if calculate_rmse:
        if training_std is None:
            if model1_train_path and model2_train_path:
                print("\nCalculating training standard deviations...")
                training_std = calculate_training_std(
                    model1_train_path=model1_train_path,
                    model2_train_path=model2_train_path,
                )
            else:
                print(
                    "\nWARNING: No training_std provided and no training paths specified."
                )
                print("RMSE will be calculated without standardization.")
                training_std = None

    # Results storage
    results_overall = []
    results_public = [] if split_by_usage else None
    results_private = [] if split_by_usage else None

    print("=" * 80)
    print("EVALUATING SOLUTIONS")
    print("=" * 80)
    print(f"Metrics to evaluate: {', '.join(metrics).upper()}")
    print("=" * 80)

    for idx, (path, name) in enumerate(zip(solution_paths, solution_names), 1):
        print(f"\n[{idx}/{len(solution_paths)}] Evaluating: {name}")
        print(f"  Path: {path}")

        try:
            # Load solution
            path = Path(path)
            if path.suffix.lower() == ".parquet":
                solution = pd.read_parquet(path)
            else:
                solution = pd.read_csv(path)

            print(f"  Loaded: {len(solution):,} rows")

            # Verify same number of rows
            if len(solution) != len(ground_truth):
                print(
                    f"  WARNING: Row count mismatch (solution: {len(solution)}, ground truth: {len(ground_truth)})"
                )

            # Evaluate overall
            print("\n  Overall Performance:")
            overall_metrics = evaluate_single_solution(
                ground_truth,
                solution,
                usage_filter=None,
                training_std=training_std if calculate_rmse else None,
                metrics=metrics,
            )

            if calculate_nse_bool and "nse_score" in overall_metrics:
                print(f"    Overall NSE:  {overall_metrics['nse_score']:.6f}")
            if calculate_rmse_bool and "rmse_score" in overall_metrics:
                print(f"    Overall RMSE: {overall_metrics['rmse_score']:.6f}")

            # Print detailed metrics by model and node type
            print("\n  By Model & Node Type:")

            # Group by model for clearer output
            # Determine which metric to use for finding models
            metric_prefix = "nse" if calculate_nse_bool else "rmse"
            models = sorted(
                set(
                    [
                        int(k.split("_")[2])
                        for k in overall_metrics.keys()
                        if k.startswith(f"{metric_prefix}_model_") and "_node_" not in k
                    ]
                )
            )

            for model_id in models:
                print(f"\n    Model {model_id}:")

                # Overall for this model
                nse_key = f"nse_model_{model_id}_overall"
                rmse_key = f"rmse_model_{model_id}_overall"

                if calculate_nse_bool and nse_key in overall_metrics:
                    print(f"      Overall NSE:  {overall_metrics[nse_key]:.6f}")
                if calculate_rmse_bool and rmse_key in overall_metrics:
                    print(f"      Overall RMSE: {overall_metrics[rmse_key]:.6f}")

                # By node type
                node_types = sorted(
                    set(
                        [
                            int(k.split("_")[-1])
                            for k in overall_metrics.keys()
                            if k.startswith(f"{metric_prefix}_model_{model_id}_node_")
                        ]
                    )
                )

                for node_type in node_types:
                    nse_node_key = f"nse_model_{model_id}_node_{node_type}"
                    rmse_node_key = f"rmse_model_{model_id}_node_{node_type}"

                    node_label = (
                        "1D nodes (surface)"
                        if node_type == 1
                        else "2D nodes (subsurface)"
                    )
                    print(f"      {node_label}:")
                    if calculate_nse_bool and nse_node_key in overall_metrics:
                        print(f"        NSE:  {overall_metrics[nse_node_key]:.6f}")
                    if calculate_rmse_bool and rmse_node_key in overall_metrics:
                        print(f"        RMSE: {overall_metrics[rmse_node_key]:.6f}")

            results_overall.append(
                {"solution_name": name, "solution_path": str(path), **overall_metrics}
            )

            # Evaluate public/private if applicable
            if split_by_usage:
                print("\n  Public Leaderboard Performance:")
                public_metrics = evaluate_single_solution(
                    ground_truth,
                    solution,
                    usage_filter="Public",
                    training_std=training_std if calculate_rmse else None,
                    metrics=metrics,
                )
                if calculate_nse_bool and "nse_score" in public_metrics:
                    print(f"    NSE:  {public_metrics['nse_score']:.6f}")
                if calculate_rmse_bool and "rmse_score" in public_metrics:
                    print(f"    RMSE: {public_metrics['rmse_score']:.6f}")
                print(f"    Rows: {public_metrics['num_predictions']:,}")

                results_public.append(
                    {
                        "solution_name": name,
                        "solution_path": str(path),
                        **public_metrics,
                    }
                )

                print("\n  Private Leaderboard Performance:")
                private_metrics = evaluate_single_solution(
                    ground_truth,
                    solution,
                    usage_filter="Private",
                    training_std=training_std if calculate_rmse else None,
                    metrics=metrics,
                )
                if calculate_nse_bool and "nse_score" in private_metrics:
                    print(f"    NSE:  {private_metrics['nse_score']:.6f}")
                if calculate_rmse_bool and "rmse_score" in private_metrics:
                    print(f"    RMSE: {private_metrics['rmse_score']:.6f}")
                print(f"    Rows: {private_metrics['num_predictions']:,}")

                results_private.append(
                    {
                        "solution_name": name,
                        "solution_path": str(path),
                        **private_metrics,
                    }
                )

        except Exception as e:
            print(f"  ERROR: {e}")
            import traceback

            traceback.print_exc()

            # Add error entries
            error_entry = {
                "solution_name": name,
                "solution_path": str(path),
                "num_predictions": 0,
            }
            if calculate_nse_bool:
                error_entry["nse_score"] = np.nan
            if calculate_rmse_bool:
                error_entry["rmse_score"] = np.nan

            results_overall.append(error_entry.copy())
            if split_by_usage:
                results_public.append(error_entry.copy())
                results_private.append(error_entry.copy())

    # Process results
    output_path = Path(output_csv)
    output_dir = output_path.parent
    output_stem = output_path.stem
    output_suffix = output_path.suffix

    results_dict = {}

    # Determine primary ranking metric (prefer NSE if available)
    primary_metric = "nse" if calculate_nse_bool else "rmse"
    primary_ascending = False if primary_metric == "nse" else True

    # Overall results
    print("\n" + "=" * 80)
    print("PROCESSING OVERALL RESULTS")
    print("=" * 80)

    results_overall_df = pd.DataFrame(results_overall)

    if calculate_nse_bool:
        results_overall_df["nse_rank"] = results_overall_df["nse_score"].rank(
            ascending=False, method="min", na_option="bottom"
        )
    if calculate_rmse:
        results_overall_df["rmse_rank"] = results_overall_df["rmse_score"].rank(
            ascending=True, method="min", na_option="bottom"
        )

    # Sort by primary metric
    results_overall_df = results_overall_df.sort_values(f"{primary_metric}_rank")

    # Reorder columns to show main metrics first, then detailed ones
    main_cols = ["solution_name"]
    if calculate_nse_bool:
        main_cols.extend(["nse_rank", "nse_score"])
    if calculate_rmse_bool:
        main_cols.extend(["rmse_rank", "rmse_score"])
    main_cols.append("num_predictions")

    detailed_cols = [
        col
        for col in results_overall_df.columns
        if col not in main_cols and col != "solution_path"
    ]
    all_cols = main_cols + sorted(detailed_cols) + ["solution_path"]
    results_overall_df = results_overall_df[
        [col for col in all_cols if col in results_overall_df.columns]
    ]

    overall_output_path = output_dir / f"{output_stem}_overall{output_suffix}"
    results_overall_df.to_csv(overall_output_path, index=False)
    results_dict["overall"] = results_overall_df

    print(f"\nRanked by {primary_metric.upper()} (best first):")
    display_cols = [f"{primary_metric}_rank", "solution_name"]
    if calculate_nse_bool:
        display_cols.append("nse_score")
    if calculate_rmse_bool:
        display_cols.append("rmse_score")
    print(
        results_overall_df[
            [col for col in display_cols if col in results_overall_df.columns]
        ].to_string(index=False)
    )
    print(f"\nResults saved to: {overall_output_path.absolute()}")

    # Public results
    if split_by_usage:
        print("\n" + "=" * 80)
        print("PROCESSING PUBLIC LEADERBOARD RESULTS")
        print("=" * 80)

        results_public_df = pd.DataFrame(results_public)

        if calculate_nse_bool:
            results_public_df["nse_rank"] = results_public_df["nse_score"].rank(
                ascending=False, method="min", na_option="bottom"
            )
        if calculate_rmse_bool:
            results_public_df["rmse_rank"] = results_public_df["rmse_score"].rank(
                ascending=True, method="min", na_option="bottom"
            )

        results_public_df = results_public_df.sort_values(f"{primary_metric}_rank")

        # Reorder columns
        main_cols = ["solution_name"]
        if calculate_nse_bool:
            main_cols.extend(["nse_rank", "nse_score"])
        if calculate_rmse_bool:
            main_cols.extend(["rmse_rank", "rmse_score"])
        main_cols.append("num_predictions")

        detailed_cols = [
            col
            for col in results_public_df.columns
            if col not in main_cols and col != "solution_path"
        ]
        all_cols = main_cols + sorted(detailed_cols) + ["solution_path"]
        results_public_df = results_public_df[
            [col for col in all_cols if col in results_public_df.columns]
        ]

        public_output_path = output_dir / f"{output_stem}_public{output_suffix}"
        results_public_df.to_csv(public_output_path, index=False)
        results_dict["public"] = results_public_df

        print(f"\nRanked by {primary_metric.upper()} (best first):")
        display_cols = [f"{primary_metric}_rank", "solution_name"]
        if calculate_nse_bool:
            display_cols.append("nse_score")
        if calculate_rmse_bool:
            display_cols.append("rmse_score")
        print(
            results_public_df[
                [col for col in display_cols if col in results_public_df.columns]
            ].to_string(index=False)
        )
        print(f"\nResults saved to: {public_output_path.absolute()}")

        # Private results
        print("\n" + "=" * 80)
        print("PROCESSING PRIVATE LEADERBOARD RESULTS")
        print("=" * 80)

        results_private_df = pd.DataFrame(results_private)

        if calculate_nse_bool:
            results_private_df["nse_rank"] = results_private_df["nse_score"].rank(
                ascending=False, method="min", na_option="bottom"
            )
        if calculate_rmse_bool:
            results_private_df["rmse_rank"] = results_private_df["rmse_score"].rank(
                ascending=True, method="min", na_option="bottom"
            )

        results_private_df = results_private_df.sort_values(f"{primary_metric}_rank")

        # Reorder columns
        main_cols = ["solution_name"]
        if calculate_nse_bool:
            main_cols.extend(["nse_rank", "nse_score"])
        if calculate_rmse_bool:
            main_cols.extend(["rmse_rank", "rmse_score"])
        main_cols.append("num_predictions")

        detailed_cols = [
            col
            for col in results_private_df.columns
            if col not in main_cols and col != "solution_path"
        ]
        all_cols = main_cols + sorted(detailed_cols) + ["solution_path"]
        results_private_df = results_private_df[
            [col for col in all_cols if col in results_private_df.columns]
        ]

        private_output_path = output_dir / f"{output_stem}_private{output_suffix}"
        results_private_df.to_csv(private_output_path, index=False)
        results_dict["private"] = results_private_df

        print(f"\nRanked by {primary_metric.upper()} (best first):")
        display_cols = [f"{primary_metric}_rank", "solution_name"]
        if calculate_nse_bool:
            display_cols.append("nse_score")
        if calculate_rmse_bool:
            display_cols.append("rmse_score")
        print(
            results_private_df[
                [col for col in display_cols if col in results_private_df.columns]
            ].to_string(index=False)
        )
        print(f"\nResults saved to: {private_output_path.absolute()}")

        # Leaderboard shake-up analysis
        print("\n" + "=" * 80)
        print("LEADERBOARD SHAKE-UP ANALYSIS")
        print("=" * 80)

        # Merge public and private ranks using primary metric
        rank_col = f"{primary_metric}_rank"
        shake_up = results_public_df[["solution_name", rank_col]].merge(
            results_private_df[["solution_name", rank_col]],
            on="solution_name",
            suffixes=("_public", "_private"),
        )
        shake_up["rank_change"] = (
            shake_up[f"{rank_col}_public"] - shake_up[f"{rank_col}_private"]
        )
        shake_up = shake_up.sort_values(f"{rank_col}_private")

        print(f"\nRank Changes (Public → Private) based on {primary_metric.upper()}:")
        print(
            shake_up[
                [
                    "solution_name",
                    f"{rank_col}_public",
                    f"{rank_col}_private",
                    "rank_change",
                ]
            ].to_string(index=False)
        )

        # Save shake-up analysis
        shake_up_path = output_dir / f"{output_stem}_shakeup{output_suffix}"
        shake_up.to_csv(shake_up_path, index=False)
        print(f"\nShake-up analysis saved to: {shake_up_path.absolute()}")

        # Identify biggest movers
        biggest_improver = shake_up.loc[shake_up["rank_change"].idxmax()]
        biggest_decliner = shake_up.loc[shake_up["rank_change"].idxmin()]

        if biggest_improver["rank_change"] > 0:
            print(f"\n📈 Biggest Improver: {biggest_improver['solution_name']}")
            print(
                f"   Public rank #{int(biggest_improver[f'{rank_col}_public'])} → Private rank #{int(biggest_improver[f'{rank_col}_private'])} (+{int(biggest_improver['rank_change'])} positions)"
            )

        if biggest_decliner["rank_change"] < 0:
            print(f"\n📉 Biggest Decliner: {biggest_decliner['solution_name']}")
            print(
                f"   Public rank #{int(biggest_decliner[f'{rank_col}_public'])} → Private rank #{int(biggest_decliner[f'{rank_col}_private'])} ({int(biggest_decliner['rank_change'])} positions)"
            )

    print("\n" + "=" * 80 + "\n")

    # Return format depends on split_by_usage
    if split_by_usage:
        return results_overall_df, results_public_df, results_private_df
    else:
        return results_overall_df


if __name__ == "__main__":
    # # Example usage
    # solution_path = "/Users/jiayulim/Documents/GitHub/dual_flood_gnn/kaggle_submission/solutions.csv"

    # # Generate comprehensive baselines and calculate their NSE scores
    # # Set save_parquet=True to save each baseline as a parquet file
    # baseline_results = generate_comprehensive_baselines(
    #     solution_path=solution_path,
    #     output_dir="baselines",
    #     random_seed=42,
    #     save_parquet=True  # Enable parquet saving
    # )

    # parquet_to_csv("/Users/jiayulim/Documents/GitHub/dual_flood_gnn/baselines/temporal_persistence_t10.parquet")

    solution_paths = [
        "submissions/gegerout/49646601_submission.csv",
        "submissions/gegerout/49642289_submission.csv",
        "submissions/kevincao/49688714_sub2.parquet",
        "submissions/kevincao/49605808_sub1.parquet",
        "submissions/timotheehenry/49684010_subm12.csv",
        "submissions/timotheehenry/49683468_subm11.csv",
        "submissions/timotheehenry/49645995_subm6.csv",
        "submissions/thelaplacian/49639878_submission.csv",
        "submissions/thelaplacian/49636087_submission.csv",
        "submissions/mattmotoki/49696172_persistence.parquet",
        "submissions/mattmotoki/49695478_model_decay.parquet",
        "submissions/yonderians/-20_sub.csv",
        "submissions/avasgaire/-1562_sub.csv",
    ]

    # Optional: Provide custom names
    solution_names = [
        "Gegerout Version 30",
        "Gegerout -4.5196 Model",
        "Kevin Cao -6.5594 Model",
        "Kevin Cao -4.5196 Model",
        "Timothee Henry -4.7775 Model 12",
        "Timothee Henry -4.7629 Model 11",
        "Timothee Henry -4.5229 Model 6",
        "TheLaplacian -80000000000+ Model A",
        "TheLaplacian -4.5196 Model B",
        "Matt Motoki -4.7775 Persistence",
        "Matt Motoki -170954133676167.0000 Model Decay",
        "Yonderians -20 Model",
        "Avasgaire -1562 Model",
    ]

    # Evaluate and rank
    overall, public, private = evaluate_and_rank_solutions(
        solution_paths=solution_paths,
        ground_truth_path="kaggle_submission/solutions.csv",
        output_csv="submissions/rmsesolution_rankings.csv",
        solution_names=solution_names,
        split_by_usage=True,
        training_std=None,
        metrics=["rmse"],
        model1_train_path="/Users/jiayulim/Documents/GitHub/dual_flood_gnn/data/Model1/processed/features_csv/train",
        model2_train_path="/Users/jiayulim/Documents/GitHub/dual_flood_gnn/data/Model2/processed/features_csv/train",
    )

    print("\nTop 3 Overall by NSE:")
    print(overall.head(3)[["solution_name", "nse_score", "rmse_score"]])

    print("\nTop 3 Public by NSE:")
    print(public.head(3)[["solution_name", "nse_score", "rmse_score"]])

    print("\nTop 3 Private by NSE:")
    print(private.head(3)[["solution_name", "nse_score", "rmse_score"]])
