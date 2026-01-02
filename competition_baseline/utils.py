from pathlib import Path
import pandas as pd
import numpy as np


def nse(y_true, y_pred):
    """Calculate Nash-Sutcliffe Efficiency"""
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)

    denominator = np.sum((y_true - np.mean(y_true)) ** 2)

    # Safety check (important for dry or constant series)
    if denominator == 0:
        denominator = 0.0001

    return 1 - np.sum((y_true - y_pred) ** 2) / denominator


def nrmse(obs, pred, type="sd"):
    """
    Calculate Normalized RMSE between observed and predicted values.

    Parameters:
    -----------
    obs : array-like
        Observed values
    pred : array-like
        Predicted values
    type : str, optional
        Type of normalization:
        - "sd"      : divide by standard deviation of obs
        - "mean"    : divide by mean of obs
        - "maxmin"  : divide by (max - min) of obs
        - "iq"      : divide by interquartile range of obs
        Default is "sd"

    Returns:
    --------
    nrmse : float
        Normalized RMSE rounded to 5 decimal places
    """
    obs = np.asarray(obs)
    pred = np.asarray(pred)

    rmse = np.sqrt(np.mean((obs - pred) ** 2))

    if type == "sd":
        norm_factor = np.std(obs)
    elif type == "mean":
        norm_factor = np.mean(obs)
    elif type == "maxmin":
        norm_factor = np.max(obs) - np.min(obs)
    elif type == "iq":
        q75, q25 = np.percentile(obs, [75, 25])
        norm_factor = q75 - q25
    else:
        raise ValueError("Wrong type! Must be one of ['sd', 'mean', 'maxmin', 'iq']")

    if norm_factor == 0:
        norm_factor = 0.0001

    nrmse_val = rmse / norm_factor
    return round(nrmse_val, 5)


def load_static(event_dir):
    static = {}

    static["1d_node"] = pd.read_csv(f"{event_dir}/1d_nodes_static.csv")
    static["1d_node"] = static["1d_node"].rename(
        columns={
            "position_x": "1d_position_x",
            "position_y": "1d_position_y",
            "roughness": "1d_roughness",
        }
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

    selected_1d_node_cols = ["timestep", "node_idx", "water_level"]
    dyn["1d_node"] = pd.read_csv(
        f"{event_dir}/1d_nodes_dynamic_all.csv", usecols=selected_1d_node_cols
    )

    selected_1d_edge_cols = [
        "timestep",
        "edge_idx",
    ]
    dyn["1d_edge"] = pd.read_csv(
        f"{event_dir}/1d_edges_dynamic_all.csv", usecols=selected_1d_edge_cols
    )

    selected_2d_node_cols = ["timestep", "node_idx", "rainfall", "water_level"]
    dyn["2d_node"] = pd.read_csv(
        f"{event_dir}/2d_nodes_dynamic_all.csv", usecols=selected_2d_node_cols
    )

    selected_2d_edge_cols = [
        "timestep",
        "edge_idx",
    ]
    dyn["2d_edge"] = pd.read_csv(
        f"{event_dir}/2d_edges_dynamic_all.csv", usecols=selected_2d_edge_cols
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


def evaluate_predictions_hierarchical(
    pred_csv_path,
    split_by_node_type=True,
    save_metrics=True,
    nrmse_types=["sd", "mean", "maxmin", "iq"],
):
    """
    Evaluate predictions with hierarchical NSE calculation:
    1. Per-node NSE
    2. Per-event NSE (average of per-node NSE)
    3. Overall NSE (average of per-event NSE)

    Also computes normalized RMSE with multiple normalization methods.

    Args:
        pred_csv_path: Path to predictions CSV
        split_by_node_type: Whether to show separate metrics for 1D/2D
        save_metrics: Whether to save metrics to CSV file
        nrmse_types: List of NRMSE normalization types to calculate

    Returns:
        dict: Dictionary containing all metrics
    """
    df = pd.read_csv(pred_csv_path)

    print("\n" + "=" * 80)
    print("HIERARCHICAL EVALUATION RESULTS")
    print("=" * 80)
    print("NSE Calculation: Per-Node → Per-Event → Overall")
    print("=" * 80)

    # Check required columns
    required_cols = [
        "node_id",
        "event_id",
        "target_water_level",
        "predicted_water_level",
    ]
    missing_cols = [col for col in required_cols if col not in df.columns]
    if missing_cols:
        raise ValueError(f"Missing required columns: {missing_cols}")

    # Store metrics in list for CSV export
    metrics_list = []

    # ============================================================================
    # OVERALL METRICS (All data pooled)
    # ============================================================================
    y_true = df["target_water_level"].values
    y_pred = df["predicted_water_level"].values

    rmse = np.sqrt(np.mean((y_true - y_pred) ** 2))
    mae = np.mean(np.abs(y_true - y_pred))
    mape = np.mean(np.abs((y_true - y_pred) / (np.abs(y_true) + 1e-8))) * 100
    r2 = 1 - (np.sum((y_true - y_pred) ** 2) / np.sum((y_true - y_true.mean()) ** 2))

    # Calculate NRMSE for all types
    nrmse_dict = {}
    for norm_type in nrmse_types:
        nrmse_dict[f"nrmse_{norm_type}"] = nrmse(y_true, y_pred, type=norm_type)

    print("\nPooled Metrics (Traditional - all samples together):")
    print(f"  Samples: {len(df)}")
    print(f"  RMSE: {rmse:.4f}")
    print(f"  MAE:  {mae:.4f}")
    print(f"  MAPE: {mape:.2f}%")
    print(f"  R²:   {r2:.4f}")
    for norm_type in nrmse_types:
        print(f"  NRMSE ({norm_type}): {nrmse_dict[f'nrmse_{norm_type}']:.5f}")

    # ============================================================================
    # HIERARCHICAL NSE CALCULATION
    # ============================================================================
    print("\n" + "=" * 80)
    print("HIERARCHICAL NSE CALCULATION")
    print("=" * 80)

    # Step 1: Calculate per-node NSE for each event
    event_nse_list = []
    event_details = []

    unique_events = sorted(df["event_id"].unique())

    for event_id in unique_events:
        df_event = df[df["event_id"] == event_id]
        unique_nodes = df_event["node_id"].unique()

        node_nse_list = []

        for node_id in unique_nodes:
            df_node = df_event[df_event["node_id"] == node_id]

            if len(df_node) > 1:  # Need at least 2 points for NSE
                y_true_node = df_node["target_water_level"].values
                y_pred_node = df_node["predicted_water_level"].values

                nse_node = nse(y_true_node, y_pred_node)
                node_nse_list.append(nse_node)

        # Step 2: Average per-node NSE to get per-event NSE
        if node_nse_list:
            event_nse = np.mean(node_nse_list)
            event_nse_list.append(event_nse)

            event_details.append(
                {
                    "event_id": event_id,
                    "n_nodes": len(node_nse_list),
                    "n_samples": len(df_event),
                    "event_nse": event_nse,
                    "min_node_nse": np.min(node_nse_list),
                    "max_node_nse": np.max(node_nse_list),
                    "std_node_nse": np.std(node_nse_list),
                }
            )

    # Step 3: Average per-event NSE to get overall NSE
    overall_nse_hierarchical = np.mean(event_nse_list) if event_nse_list else np.nan

    print("\nHierarchical NSE Results:")
    print(f"  Total events: {len(unique_events)}")
    print(f"  Total unique nodes: {df['node_id'].nunique()}")
    print(
        f"  Average nodes per event: {np.mean([e['n_nodes'] for e in event_details]):.1f}"
    )
    print(f"\n  Overall NSE (hierarchical): {overall_nse_hierarchical:.4f}")
    print(
        f"  Event NSE range: [{np.min(event_nse_list):.4f}, {np.max(event_nse_list):.4f}]"
    )
    print(f"  Event NSE std: {np.std(event_nse_list):.4f}")

    # Store overall results
    overall_results = {
        "metric_type": "overall",
        "n_samples": len(df),
        "n_events": len(unique_events),
        "n_nodes": df["node_id"].nunique(),
        "rmse": rmse,
        "mae": mae,
        "mape": mape,
        "r2": r2,
        "nse_hierarchical": overall_nse_hierarchical,
        **nrmse_dict,
    }
    metrics_list.append(overall_results)

    # ============================================================================
    # PER-EVENT METRICS
    # ============================================================================
    print("\n" + "=" * 80)
    print("PER-EVENT SUMMARY (First 10 events)")
    print("=" * 80)
    print(
        f"{'Event':>8} | {'Samples':>8} | {'Nodes':>6} | {'NSE':>8} | {'Node NSE Range':>20}"
    )
    print("-" * 80)

    for i, event_info in enumerate(event_details[:10]):
        print(
            f"{event_info['event_id']:>8} | {event_info['n_samples']:>8} | "
            f"{event_info['n_nodes']:>6} | {event_info['event_nse']:>8.4f} | "
            f"[{event_info['min_node_nse']:>6.4f}, {event_info['max_node_nse']:>6.4f}]"
        )

    if len(event_details) > 10:
        print(f"... and {len(event_details) - 10} more events")

    # ============================================================================
    # NODE TYPE SPLIT (if requested)
    # ============================================================================
    if split_by_node_type and "node_type" in df.columns:
        # 1D NODES
        print("\n" + "=" * 80)
        print("1D NODE METRICS")
        print("=" * 80)
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

            # NRMSE for 1D
            nrmse_1d_dict = {}
            for norm_type in nrmse_types:
                nrmse_1d_dict[f"nrmse_{norm_type}"] = nrmse(
                    y_true_1d, y_pred_1d, type=norm_type
                )

            # Hierarchical NSE for 1D nodes
            event_nse_1d_list = []
            for event_id in unique_events:
                df_event_1d = df_1d[df_1d["event_id"] == event_id]
                if len(df_event_1d) == 0:
                    continue

                unique_nodes_1d = df_event_1d["node_id"].unique()
                node_nse_1d_list = []

                for node_id in unique_nodes_1d:
                    df_node = df_event_1d[df_event_1d["node_id"] == node_id]
                    if len(df_node) > 1:
                        y_true_node = df_node["target_water_level"].values
                        y_pred_node = df_node["predicted_water_level"].values
                        nse_node = nse(y_true_node, y_pred_node)
                        node_nse_1d_list.append(nse_node)

                if node_nse_1d_list:
                    event_nse_1d_list.append(np.mean(node_nse_1d_list))

            nse_1d_hierarchical = (
                np.mean(event_nse_1d_list) if event_nse_1d_list else np.nan
            )

            print(f"  Samples: {len(df_1d)}")
            print(f"  Unique nodes: {df_1d['node_id'].nunique()}")
            print(f"  RMSE: {rmse_1d:.4f}")
            print(f"  MAE:  {mae_1d:.4f}")
            print(f"  MAPE: {mape_1d:.2f}%")
            print(f"  R²:   {r2_1d:.4f}")
            print(f"  NSE (hierarchical): {nse_1d_hierarchical:.4f}")
            for norm_type in nrmse_types:
                print(
                    f"  NRMSE ({norm_type}): {nrmse_1d_dict[f'nrmse_{norm_type}']:.5f}"
                )

            metrics_list.append(
                {
                    "metric_type": "1d_nodes",
                    "n_samples": len(df_1d),
                    "n_events": len(event_nse_1d_list),
                    "n_nodes": df_1d["node_id"].nunique(),
                    "rmse": rmse_1d,
                    "mae": mae_1d,
                    "mape": mape_1d,
                    "r2": r2_1d,
                    "nse_hierarchical": nse_1d_hierarchical,
                    **nrmse_1d_dict,
                }
            )
        else:
            print("  No 1D samples in predictions")

        # 2D NODES
        print("\n" + "=" * 80)
        print("2D NODE METRICS")
        print("=" * 80)
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

            # NRMSE for 2D
            nrmse_2d_dict = {}
            for norm_type in nrmse_types:
                nrmse_2d_dict[f"nrmse_{norm_type}"] = nrmse(
                    y_true_2d, y_pred_2d, type=norm_type
                )

            # Hierarchical NSE for 2D nodes
            event_nse_2d_list = []
            for event_id in unique_events:
                df_event_2d = df_2d[df_2d["event_id"] == event_id]
                if len(df_event_2d) == 0:
                    continue

                unique_nodes_2d = df_event_2d["node_id"].unique()
                node_nse_2d_list = []

                for node_id in unique_nodes_2d:
                    df_node = df_event_2d[df_event_2d["node_id"] == node_id]
                    if len(df_node) > 1:
                        y_true_node = df_node["target_water_level"].values
                        y_pred_node = df_node["predicted_water_level"].values
                        nse_node = nse(y_true_node, y_pred_node)
                        node_nse_2d_list.append(nse_node)

                if node_nse_2d_list:
                    event_nse_2d_list.append(np.mean(node_nse_2d_list))

            nse_2d_hierarchical = (
                np.mean(event_nse_2d_list) if event_nse_2d_list else np.nan
            )

            print(f"  Samples: {len(df_2d)}")
            print(f"  Unique nodes: {df_2d['node_id'].nunique()}")
            print(f"  RMSE: {rmse_2d:.4f}")
            print(f"  MAE:  {mae_2d:.4f}")
            print(f"  MAPE: {mape_2d:.2f}%")
            print(f"  R²:   {r2_2d:.4f}")
            print(f"  NSE (hierarchical): {nse_2d_hierarchical:.4f}")
            for norm_type in nrmse_types:
                print(
                    f"  NRMSE ({norm_type}): {nrmse_2d_dict[f'nrmse_{norm_type}']:.5f}"
                )

            metrics_list.append(
                {
                    "metric_type": "2d_nodes",
                    "n_samples": len(df_2d),
                    "n_events": len(event_nse_2d_list),
                    "n_nodes": df_2d["node_id"].nunique(),
                    "rmse": rmse_2d,
                    "mae": mae_2d,
                    "mape": mape_2d,
                    "r2": r2_2d,
                    "nse_hierarchical": nse_2d_hierarchical,
                    **nrmse_2d_dict,
                }
            )
        else:
            print("  No 2D samples in predictions")

    # ============================================================================
    # ERROR DISTRIBUTION
    # ============================================================================
    errors = y_true - y_pred
    print("\n" + "=" * 80)
    print("ERROR DISTRIBUTION")
    print("=" * 80)
    print(f"  Mean Error: {errors.mean():.4f}")
    print(f"  Std Error:  {errors.std():.4f}")
    print(f"  Min Error:  {errors.min():.4f}")
    print(f"  Max Error:  {errors.max():.4f}")
    print(f"  25th percentile (|error|): {np.percentile(np.abs(errors), 25):.4f}")
    print(f"  50th percentile (|error|): {np.percentile(np.abs(errors), 50):.4f}")
    print(f"  75th percentile (|error|): {np.percentile(np.abs(errors), 75):.4f}")
    print(f"  95th percentile (|error|): {np.percentile(np.abs(errors), 95):.4f}")

    print("=" * 80 + "\n")

    # ============================================================================
    # SAVE METRICS
    # ============================================================================
    if save_metrics:
        # Save overall metrics CSV
        metrics_df = pd.DataFrame(metrics_list)
        pred_path = Path(pred_csv_path)
        metrics_path = pred_path.parent / f"{pred_path.stem}_metrics.csv"
        metrics_df.to_csv(metrics_path, index=False)
        print(f"✓ Saved metrics to: {metrics_path}")

        # Save per-event details
        if event_details:
            event_df = pd.DataFrame(event_details)
            event_metrics_path = (
                pred_path.parent / f"{pred_path.stem}_event_metrics.csv"
            )
            event_df.to_csv(event_metrics_path, index=False)
            print(f"✓ Saved per-event metrics to: {event_metrics_path}")

        # Save detailed summary
        summary_path = pred_path.parent / f"{pred_path.stem}_summary.txt"
        with open(summary_path, "w") as f:
            f.write("=" * 80 + "\n")
            f.write("HIERARCHICAL EVALUATION SUMMARY\n")
            f.write("=" * 80 + "\n")
            f.write(f"Predictions file: {pred_csv_path}\n")
            f.write(f"Generated: {pd.Timestamp.now()}\n\n")

            f.write("NSE CALCULATION METHOD\n")
            f.write("-" * 80 + "\n")
            f.write("1. Calculate NSE for each node within each event\n")
            f.write("2. Average per-node NSE to get per-event NSE\n")
            f.write("3. Average per-event NSE to get overall NSE\n\n")

            f.write("OVERALL METRICS\n")
            f.write("-" * 80 + "\n")
            f.write(f"Samples:     {len(df):>12,}\n")
            f.write(f"Events:      {len(unique_events):>12}\n")
            f.write(f"Nodes:       {df['node_id'].nunique():>12}\n")
            f.write(f"RMSE:        {rmse:>12.4f}\n")
            f.write(f"MAE:         {mae:>12.4f}\n")
            f.write(f"MAPE:        {mape:>12.2f}%\n")
            f.write(f"R²:          {r2:>12.4f}\n")
            f.write(f"NSE:         {overall_nse_hierarchical:>12.4f}\n")
            for norm_type in nrmse_types:
                f.write(
                    f"NRMSE ({norm_type}):  {nrmse_dict[f'nrmse_{norm_type}']:>12.5f}\n"
                )
            f.write("\n")

            if len(metrics_list) > 1 and metrics_list[1]["metric_type"] == "1d_nodes":
                f.write("1D NODE METRICS\n")
                f.write("-" * 80 + "\n")
                m = metrics_list[1]
                f.write(f"Samples:     {m['n_samples']:>12,}\n")
                f.write(f"Nodes:       {m['n_nodes']:>12}\n")
                f.write(f"RMSE:        {m['rmse']:>12.4f}\n")
                f.write(f"MAE:         {m['mae']:>12.4f}\n")
                f.write(f"MAPE:        {m['mape']:>12.2f}%\n")
                f.write(f"R²:          {m['r2']:>12.4f}\n")
                f.write(f"NSE:         {m['nse_hierarchical']:>12.4f}\n")
                for norm_type in nrmse_types:
                    f.write(f"NRMSE ({norm_type}):  {m[f'nrmse_{norm_type}']:>12.5f}\n")
                f.write("\n")

            if len(metrics_list) > 2 and metrics_list[2]["metric_type"] == "2d_nodes":
                f.write("2D NODE METRICS\n")
                f.write("-" * 80 + "\n")
                m = metrics_list[2]
                f.write(f"Samples:     {m['n_samples']:>12,}\n")
                f.write(f"Nodes:       {m['n_nodes']:>12}\n")
                f.write(f"RMSE:        {m['rmse']:>12.4f}\n")
                f.write(f"MAE:         {m['mae']:>12.4f}\n")
                f.write(f"MAPE:        {m['mape']:>12.2f}%\n")
                f.write(f"R²:          {m['r2']:>12.4f}\n")
                f.write(f"NSE:         {m['nse_hierarchical']:>12.4f}\n")
                for norm_type in nrmse_types:
                    f.write(f"NRMSE ({norm_type}):  {m[f'nrmse_{norm_type}']:>12.5f}\n")
                f.write("\n")

            f.write("=" * 80 + "\n")

        print(f"✓ Saved summary to: {summary_path}\n")

    return overall_results, event_details
