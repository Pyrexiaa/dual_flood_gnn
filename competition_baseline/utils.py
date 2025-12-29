from pathlib import Path
import pandas as pd
import numpy as np


def load_static(event_dir):
    static = {}

    static["1d_node"] = pd.read_csv(f"{event_dir}/1d_nodes_static.csv")
    static["1d_node"] = static["1d_node"].rename(
        columns={"position_x": "1d_position_x", "position_y": "1d_position_y"}
    )
    static["1d_edge"] = pd.read_csv(f"{event_dir}/1d_edges_static.csv")
    static["1d_edge"] = static["1d_edge"].rename(columns={"length": "1d_length"})
    static["1d_edge_index"] = pd.read_csv(f"{event_dir}/1d_edge_index.csv")

    static["2d_node"] = pd.read_csv(f"{event_dir}/2d_nodes_static.csv")
    static["2d_node"] = static["2d_node"].rename(
        columns={"position_x": "2d_position_x", "position_y": "2d_position_y"}
    )
    static["2d_edge"] = pd.read_csv(f"{event_dir}/2d_edges_static.csv")
    static["2d_edge"] = static["2d_edge"].rename(columns={"length": "2d_length"})
    static["2d_edge_index"] = pd.read_csv(f"{event_dir}/2d_edge_index.csv")
    static["1d_2d_conn"] = pd.read_csv(f"{event_dir}/1d2d_connections.csv")

    return static


def load_dynamic(event_dir):
    dyn = {}

    dyn["1d_node"] = pd.read_csv(f"{event_dir}/1d_nodes_dynamic_all.csv")
    dyn["1d_edge"] = pd.read_csv(f"{event_dir}/1d_edges_dynamic_all.csv")
    dyn["1d_edge"] = dyn["1d_edge"].rename(
        columns={"flow": "1d_flow", "velocity": "1d_velocity"}
    )

    dyn["2d_node"] = pd.read_csv(f"{event_dir}/2d_nodes_dynamic_all.csv")
    dyn["2d_edge"] = pd.read_csv(f"{event_dir}/2d_edges_dynamic_all.csv")
    dyn["2d_edge"] = dyn["2d_edge"].rename(
        columns={"flow": "2d_flow", "velocity": "2d_velocity"}
    )
    dyn["timesteps"] = pd.read_csv(f"{event_dir}/timesteps.csv")

    return dyn


def attach_timestamps(df, timesteps_df):
    """
    df must contain `timestep_idx`
    timesteps_df contains [timestep_idx, timestamp]
    """
    df_edited = df.rename(columns={"timestep": "timestep_idx"})
    return df_edited.merge(timesteps_df, on="timestep_idx", how="left")


def aggregate_edge_features(edge_dyn, edge_index):
    merged = edge_dyn.merge(edge_index, on="edge_idx")

    agg = (
        merged.groupby(["timestep_idx", "to_node"])
        .agg({"flow": "sum", "velocity": "mean"})
        .reset_index()
        .rename(columns={"to_node": "node_idx"})
    )

    return agg


def build_node_timeseries(node_dyn, edge_agg, node_static):
    df = (
        node_dyn.merge(edge_agg, on=["timestep_idx", "node_idx"], how="left")
        .merge(node_static, on="node_idx", how="left")
        .fillna(0.0)
        .sort_values("timestep_idx")
    )

    return df


def make_samples(node_df, window):
    X, y = [], []

    feature_cols = [
        c
        for c in node_df.columns
        if c not in ["timestep_idx", "timestamp", "node_idx", "water_level"]
    ]

    values = node_df.reset_index(drop=True)

    for t in range(window, len(values) - 1):
        x = values.loc[t - window : t - 1, feature_cols].values.flatten()

        target = values.loc[t + 1, "water_level"]

        X.append(x)
        y.append(target)

    return X, y


def evaluate_predictions(pred_csv_path, split_by_node_type=True, save_metrics=True):
    """
    Evaluate predictions from CSV file and optionally save metrics to CSV.

    Args:
        pred_csv_path: Path to predictions CSV
        split_by_node_type: Whether to show separate metrics for 1D/2D
        save_metrics: Whether to save metrics to CSV file

    Returns:
        dict: Dictionary containing all metrics
    """
    df = pd.read_csv(pred_csv_path)

    print("\n" + "=" * 80)
    print("EVALUATION RESULTS")
    print("=" * 80)

    # Overall metrics
    y_true = df["target_water_level"].values
    y_pred = df["predicted_water_level"].values

    rmse = np.sqrt(np.mean((y_true - y_pred) ** 2))
    mae = np.mean(np.abs(y_true - y_pred))
    mape = np.mean(np.abs((y_true - y_pred) / (np.abs(y_true) + 1e-8))) * 100
    r2 = 1 - (np.sum((y_true - y_pred) ** 2) / np.sum((y_true - y_true.mean()) ** 2))

    print("\nOverall Metrics:")
    print(f"  Samples: {len(df)}")
    print(f"  RMSE: {rmse:.4f}")
    print(f"  MAE:  {mae:.4f}")
    print(f"  MAPE: {mape:.2f}%")
    print(f"  R²:   {r2:.4f}")

    results = {
        "metric_type": "overall",
        "n_samples": len(df),
        "rmse": rmse,
        "mae": mae,
        "mape": mape,
        "r2": r2,
    }

    # Store metrics in list for CSV export
    metrics_list = [results.copy()]

    if split_by_node_type and "node_type" in df.columns:
        print("\n" + "-" * 80)
        print("1D Node Metrics:")
        print("-" * 80)
        df_1d = df[df["node_type"] == 0]
        if len(df_1d) > 0:
            y_true_1d = df_1d["target_water_level"].values
            y_pred_1d = df_1d["predicted_water_level"].values

            rmse_1d = np.sqrt(np.mean((y_true_1d - y_pred_1d) ** 2))
            mae_1d = np.mean(np.abs(y_true_1d - y_pred_1d))
            mape_1d = (
                np.mean(np.abs((y_true_1d - y_pred_1d) / (np.abs(y_true_1d) + 1e-8)))
                * 100
            )
            r2_1d = 1 - (
                np.sum((y_true_1d - y_pred_1d) ** 2)
                / np.sum((y_true_1d - y_true_1d.mean()) ** 2)
            )

            print(f"  Samples: {len(df_1d)}")
            print(f"  RMSE: {rmse_1d:.4f}")
            print(f"  MAE:  {mae_1d:.4f}")
            print(f"  MAPE: {mape_1d:.2f}%")
            print(f"  R²:   {r2_1d:.4f}")

            results["rmse_1d"] = rmse_1d
            results["mae_1d"] = mae_1d
            results["mape_1d"] = mape_1d
            results["r2_1d"] = r2_1d
            results["n_samples_1d"] = len(df_1d)

            # Add to metrics list
            metrics_list.append(
                {
                    "metric_type": "1d_nodes",
                    "n_samples": len(df_1d),
                    "rmse": rmse_1d,
                    "mae": mae_1d,
                    "mape": mape_1d,
                    "r2": r2_1d,
                }
            )
        else:
            print("  No 1D samples in predictions")

        print("\n" + "-" * 80)
        print("2D Node Metrics:")
        print("-" * 80)
        df_2d = df[df["node_type"] == 1]
        if len(df_2d) > 0:
            y_true_2d = df_2d["target_water_level"].values
            y_pred_2d = df_2d["predicted_water_level"].values

            rmse_2d = np.sqrt(np.mean((y_true_2d - y_pred_2d) ** 2))
            mae_2d = np.mean(np.abs(y_true_2d - y_pred_2d))
            mape_2d = (
                np.mean(np.abs((y_true_2d - y_pred_2d) / (np.abs(y_true_2d) + 1e-8)))
                * 100
            )
            r2_2d = 1 - (
                np.sum((y_true_2d - y_pred_2d) ** 2)
                / np.sum((y_true_2d - y_true_2d.mean()) ** 2)
            )

            print(f"  Samples: {len(df_2d)}")
            print(f"  RMSE: {rmse_2d:.4f}")
            print(f"  MAE:  {mae_2d:.4f}")
            print(f"  MAPE: {mape_2d:.2f}%")
            print(f"  R²:   {r2_2d:.4f}")

            results["rmse_2d"] = rmse_2d
            results["mae_2d"] = mae_2d
            results["mape_2d"] = mape_2d
            results["r2_2d"] = r2_2d
            results["n_samples_2d"] = len(df_2d)

            # Add to metrics list
            metrics_list.append(
                {
                    "metric_type": "2d_nodes",
                    "n_samples": len(df_2d),
                    "rmse": rmse_2d,
                    "mae": mae_2d,
                    "mape": mape_2d,
                    "r2": r2_2d,
                }
            )
        else:
            print("  No 2D samples in predictions")

    # Error distribution
    errors = y_true - y_pred
    print("\n" + "-" * 80)
    print("Error Distribution:")
    print("-" * 80)
    print(f"  Mean Error: {errors.mean():.4f}")
    print(f"  Std Error:  {errors.std():.4f}")
    print(f"  Min Error:  {errors.min():.4f}")
    print(f"  Max Error:  {errors.max():.4f}")
    print(f"  25th percentile (|error|): {np.percentile(np.abs(errors), 25):.4f}")
    print(f"  50th percentile (|error|): {np.percentile(np.abs(errors), 50):.4f}")
    print(f"  75th percentile (|error|): {np.percentile(np.abs(errors), 75):.4f}")
    print(f"  95th percentile (|error|): {np.percentile(np.abs(errors), 95):.4f}")

    # Add error distribution to results
    results["mean_error"] = errors.mean()
    results["std_error"] = errors.std()
    results["min_error"] = errors.min()
    results["max_error"] = errors.max()
    results["abs_error_p25"] = np.percentile(np.abs(errors), 25)
    results["abs_error_p50"] = np.percentile(np.abs(errors), 50)
    results["abs_error_p75"] = np.percentile(np.abs(errors), 75)
    results["abs_error_p95"] = np.percentile(np.abs(errors), 95)

    # Add error distribution to overall metrics in list
    metrics_list[0].update(
        {
            "mean_error": errors.mean(),
            "std_error": errors.std(),
            "min_error": errors.min(),
            "max_error": errors.max(),
            "abs_error_p25": np.percentile(np.abs(errors), 25),
            "abs_error_p50": np.percentile(np.abs(errors), 50),
            "abs_error_p75": np.percentile(np.abs(errors), 75),
            "abs_error_p95": np.percentile(np.abs(errors), 95),
        }
    )

    print("=" * 80 + "\n")

    # Save metrics to CSV
    if save_metrics:
        # Create metrics DataFrame
        metrics_df = pd.DataFrame(metrics_list)

        # Generate output filename
        pred_path = Path(pred_csv_path)
        metrics_path = pred_path.parent / f"{pred_path.stem}_metrics.csv"

        metrics_df.to_csv(metrics_path, index=False)
        print(f"✓ Saved metrics to: {metrics_path}\n")

        # Also save a detailed summary with timestamp
        summary_path = pred_path.parent / f"{pred_path.stem}_summary.txt"
        with open(summary_path, "w") as f:
            f.write("=" * 80 + "\n")
            f.write("EVALUATION SUMMARY\n")
            f.write("=" * 80 + "\n")
            f.write(f"Predictions file: {pred_csv_path}\n")
            f.write(f"Generated: {pd.Timestamp.now()}\n\n")

            f.write("OVERALL METRICS\n")
            f.write("-" * 80 + "\n")
            f.write(f"Samples:     {len(df):>12,}\n")
            f.write(f"RMSE:        {rmse:>12.4f}\n")
            f.write(f"MAE:         {mae:>12.4f}\n")
            f.write(f"MAPE:        {mape:>12.2f}%\n")
            f.write(f"R²:          {r2:>12.4f}\n\n")

            if "rmse_1d" in results:
                f.write("1D NODE METRICS\n")
                f.write("-" * 80 + "\n")
                f.write(f"Samples:     {results['n_samples_1d']:>12,}\n")
                f.write(f"RMSE:        {results['rmse_1d']:>12.4f}\n")
                f.write(f"MAE:         {results['mae_1d']:>12.4f}\n")
                f.write(f"MAPE:        {results['mape_1d']:>12.2f}%\n")
                f.write(f"R²:          {results['r2_1d']:>12.4f}\n\n")

            if "rmse_2d" in results:
                f.write("2D NODE METRICS\n")
                f.write("-" * 80 + "\n")
                f.write(f"Samples:     {results['n_samples_2d']:>12,}\n")
                f.write(f"RMSE:        {results['rmse_2d']:>12.4f}\n")
                f.write(f"MAE:         {results['mae_2d']:>12.4f}\n")
                f.write(f"MAPE:        {results['mape_2d']:>12.2f}%\n")
                f.write(f"R²:          {results['r2_2d']:>12.4f}\n\n")

            f.write("ERROR DISTRIBUTION\n")
            f.write("-" * 80 + "\n")
            f.write(f"Mean Error:  {errors.mean():>12.4f}\n")
            f.write(f"Std Error:   {errors.std():>12.4f}\n")
            f.write(f"Min Error:   {errors.min():>12.4f}\n")
            f.write(f"Max Error:   {errors.max():>12.4f}\n")
            f.write(f"25th pct:    {np.percentile(np.abs(errors), 25):>12.4f}\n")
            f.write(f"50th pct:    {np.percentile(np.abs(errors), 50):>12.4f}\n")
            f.write(f"75th pct:    {np.percentile(np.abs(errors), 75):>12.4f}\n")
            f.write(f"95th pct:    {np.percentile(np.abs(errors), 95):>12.4f}\n")
            f.write("=" * 80 + "\n")

        print(f"✓ Saved summary to: {summary_path}\n")

    return results
