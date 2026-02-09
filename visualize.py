from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import r2_score
import torch
from torch import Tensor
from argparse import ArgumentParser, Namespace


def NSE(pred: Tensor, target: Tensor) -> Tensor:
    """Nash Sutcliffe Efficiency"""
    model_sse = torch.sum((target - pred) ** 2)
    mean_model_sse = torch.sum((target - target.mean()) ** 2)
    return 1 - (model_sse / mean_model_sse)


def calculate_nse_numpy(pred: np.ndarray, target: np.ndarray) -> float:
    """NSE calculation for numpy arrays"""
    pred_tensor = torch.from_numpy(pred).float()
    target_tensor = torch.from_numpy(target).float()
    return NSE(pred_tensor, target_tensor).item()


def rmse(y_true, y_pred):
    """Calculate Root Mean Squared Error"""
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    return np.sqrt(np.mean((y_true - y_pred) ** 2))


def standardized_rmse(y_true, y_pred, std_dev):
    """Calculate RMSE standardized by provided standard deviation"""
    if std_dev == 0 or np.isnan(std_dev):
        return np.nan

    rmse_val = rmse(y_true, y_pred)
    return rmse_val / std_dev


def plot_individual_node_timeseries(
    csv_path="gru_test_predictions_test.csv",
    models_config=None,
):
    """
    Plot complete time series for individual nodes for each model separately.
    Creates one figure per model with 1D nodes (left) and 2D nodes (right).

    Args:
        csv_path: Path to the predictions CSV file
        models_config: Dictionary with model configurations. Format:
            {
                1: {  # model_id
                    "1d_event_id": 18,
                    "nodes_1d": [12, 15, 11, 0, 14],
                    "2d_event_id": 97,
                    "nodes_2d": [857, 856, 914, 913, 978],
                    'color': 'red',
                    'marker': 'x',
                    'linestyle': '--',
                    'label': 'Model 1'
                },
                2: {
                    "1d_event_id": 65,
                    "nodes_1d": [123, 193, 194, 164, 165],
                    "2d_event_id": 17,
                    "nodes_2d": [1464, 1465, 1463, 1466, 1850],
                    'color': 'green',
                    'marker': '^',
                    'linestyle': ':',
                    'label': 'Model 2'
                }
            }
    """

    # ========================
    # DEFAULT CONFIGURATION
    # ========================
    if models_config is None:
        models_config = {
            1: {
                "1d_event_id": 8,
                "nodes_1d": [9, 4, 12, 0, 11],
                "2d_event_id": 8,
                "nodes_2d": [2854, 563, 3668, 3488, 2960],
                "color": "red",
                "marker": "x",
                "linestyle": "--",
                "label": "Model 1",
            },
            2: {
                "1d_event_id": 65,
                "nodes_1d": [193, 194, 123, 3, 2],
                "2d_event_id": 62,
                "nodes_2d": [493, 4245, 4248, 4077, 1112],
                "color": "green",
                "marker": "^",
                "linestyle": ":",
                "label": "Model 2",
            },
        }

    # Load predictions
    df = pd.read_csv(csv_path)

    csv_path_obj = Path(csv_path)
    output_path = csv_path_obj.with_suffix("") / "individual_node_timeseries_by_model.png"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_dir = csv_path_obj.with_suffix("")

    print(f"Total samples: {len(df)}")
    print(f"Columns: {df.columns.tolist()}")

    # Check available models
    unique_models = df["model_id"].unique()
    print(f"\nAvailable models in data: {sorted(unique_models)}")
    print(f"Models to plot: {sorted(models_config.keys())}")

    # ========================
    # Create separate figure for each model
    # ========================
    for model_id, config in models_config.items():
        print(f"\n{'=' * 50}")
        print(f"Plotting {config['label']} (ID: {model_id})")
        print(f"{'=' * 50}")

        nodes_1d = config["nodes_1d"]
        nodes_2d = config["nodes_2d"]
        event_id_1d = config["1d_event_id"]
        event_id_2d = config["2d_event_id"]

        # Determine number of rows needed (max of 1D and 2D nodes)
        num_rows = max(len(nodes_1d), len(nodes_2d))

        # Create figure for this model
        fig, axes = plt.subplots(num_rows, 2, figsize=(16, 4 * num_rows))

        if num_rows == 1:
            axes = axes.reshape(1, -1)

        # Add overall title for the figure
        fig.suptitle(
            f"{config['label']} - Event 1D {event_id_1d}, Event 2D {event_id_2d}",
            fontsize=14,
            fontweight="bold",
            y=0.995,
        )

        # ========================
        # Plot 1D Nodes (Left Column)
        # ========================
        for row_idx in range(num_rows):
            ax = axes[row_idx, 0]
            # ax.set_ylim([180, 380])

            if row_idx < len(nodes_1d):
                node_id = nodes_1d[row_idx]

                # Get data for this specific node and event
                node_data = df[
                    (df["node_id"] == node_id)
                    & (df["model_id"] == model_id)
                    & (df["event_id"] == event_id_1d)
                    & (df["node_type"] == 1)
                ].sort_values("timestep")

                if len(node_data) > 0:
                    timesteps = node_data["timestep"].values

                    # Plot ground truth
                    ax.plot(
                        timesteps,
                        node_data["target_water_level"],
                        label="Ground Truth",
                        color="blue",
                        marker="o",
                        linewidth=2,
                        markersize=4,
                        alpha=0.8,
                    )

                    # Plot model predictions
                    ax.plot(
                        timesteps,
                        node_data["water_level"],
                        label="Prediction",
                        color=config["color"],
                        marker=config["marker"],
                        linewidth=2,
                        markersize=4,
                        alpha=0.8,
                        linestyle=config["linestyle"],
                    )

                    # Calculate metrics
                    rmse = np.sqrt(
                        np.mean(
                            (node_data["target_water_level"] - node_data["water_level"])
                            ** 2
                        )
                    )
                    mae = np.mean(
                        np.abs(
                            node_data["target_water_level"] - node_data["water_level"]
                        )
                    )
                    r2 = r2_score(
                        node_data["target_water_level"], node_data["water_level"]
                    )
                    nse = calculate_nse_numpy(
                        node_data["water_level"].values,
                        node_data["target_water_level"].values,
                    )

                    ax.set_title(
                        f"1D Node {node_id} | RMSE: {rmse:.4f} | MAE: {mae:.4f} | R²: {r2:.4f} | NSE: {nse:.4f}",
                        fontsize=10,
                        fontweight="bold",
                    )
                    ax.set_xlabel("Timestep", fontsize=10)
                    ax.set_ylabel("Water Level", fontsize=10)
                    ax.legend(loc="best", fontsize=9)
                    ax.grid(True, alpha=0.3)

                    print(f"  ✓ 1D Node {node_id}: {len(node_data)} timesteps")
                else:
                    ax.text(
                        0.5,
                        0.5,
                        f"No data for 1D Node {node_id}",
                        ha="center",
                        va="center",
                        transform=ax.transAxes,
                    )
                    print(f"  ✗ 1D Node {node_id}: No data found")
            else:
                # Empty subplot if no more 1D nodes
                ax.axis("off")

        # ========================
        # Plot 2D Nodes (Right Column)
        # ========================
        for row_idx in range(num_rows):
            ax = axes[row_idx, 1]
            # ax.set_ylim([290, 360])

            if row_idx < len(nodes_2d):
                node_id = nodes_2d[row_idx]

                # Get data for this specific node and event
                node_data = df[
                    (df["node_id"] == node_id)
                    & (df["model_id"] == model_id)
                    & (df["event_id"] == event_id_2d)
                    & (df["node_type"] == 2)
                ].sort_values("timestep")

                if len(node_data) > 0:
                    timesteps = node_data["timestep"].values

                    # Plot ground truth
                    ax.plot(
                        timesteps,
                        node_data["target_water_level"],
                        label="Ground Truth",
                        color="blue",
                        marker="o",
                        linewidth=2,
                        markersize=4,
                        alpha=0.8,
                    )

                    # Plot model predictions
                    ax.plot(
                        timesteps,
                        node_data["water_level"],
                        label="Prediction",
                        color=config["color"],
                        marker=config["marker"],
                        linewidth=2,
                        markersize=4,
                        alpha=0.8,
                        linestyle=config["linestyle"],
                    )

                    # Calculate metrics
                    rmse = np.sqrt(
                        np.mean(
                            (node_data["target_water_level"] - node_data["water_level"])
                            ** 2
                        )
                    )
                    mae = np.mean(
                        np.abs(
                            node_data["target_water_level"] - node_data["water_level"]
                        )
                    )
                    r2 = r2_score(
                        node_data["target_water_level"], node_data["water_level"]
                    )
                    nse = calculate_nse_numpy(
                        node_data["water_level"].values,
                        node_data["target_water_level"].values,
                    )

                    ax.set_title(
                        f"2D Node {node_id} | RMSE: {rmse:.4f} | MAE: {mae:.4f} | R²: {r2:.4f} | NSE: {nse:.4f}",
                        fontsize=10,
                        fontweight="bold",
                    )
                    ax.set_xlabel("Timestep", fontsize=10)
                    ax.set_ylabel("Water Level", fontsize=10)
                    ax.legend(loc="best", fontsize=9)
                    ax.grid(True, alpha=0.3)

                    print(f"  ✓ 2D Node {node_id}: {len(node_data)} timesteps")
                else:
                    ax.text(
                        0.5,
                        0.5,
                        f"No data for 2D Node {node_id}",
                        ha="center",
                        va="center",
                        transform=ax.transAxes,
                    )
                    print(f"  ✗ 2D Node {node_id}: No data found")
            else:
                # Empty subplot if no more 2D nodes
                ax.axis("off")

        # Save figure for this model
        output_path = output_dir / f"model_{model_id}_node_timeseries.png"
        plt.tight_layout()
        plt.savefig(output_path, dpi=300, bbox_inches="tight")
        print(f"\n✓ {config['label']} plot saved as '{output_path}'")
        plt.close()

    print(f"\n{'=' * 50}")
    print("All plots completed!")
    print(f"{'=' * 50}")


def analyze_node_statistics(csv_path="gru_test_predictions_test.csv"):
    """
    Print statistics about predictions per node.
    Shows both aggregated metrics AND per-node metric distributions.
    """
    df = pd.read_csv(csv_path)

    print("=" * 80)
    print("NODE STATISTICS")
    print("=" * 80)

    # Overall statistics
    print(f"\nTotal samples: {len(df)}")
    print(f"Total unique nodes: {df['node_id'].nunique()}")
    print(f"Timestep range: {df['timestep'].min()} to {df['timestep'].max()}")

    # By node type
    for node_type in [1, 2]:
        node_type_str = "1D" if node_type == 1 else "2D"
        df_type = df[df["node_type"] == node_type]

        print(f"\n{node_type_str} Nodes:")
        print(f"  Total samples: {len(df_type)}")
        print(f"  Unique nodes: {df_type['node_id'].nunique()}")

        # Samples per node statistics
        samples_per_node = df_type.groupby("node_id").size()
        print("  Timesteps per node:")
        print(f"    Mean: {samples_per_node.mean():.1f}")
        print(f"    Min: {samples_per_node.min()}")
        print(f"    Max: {samples_per_node.max()}")
        print(f"    Std: {samples_per_node.std():.1f}")

        # AGGREGATED metrics (treating all samples as one pool)
        rmse_agg = np.sqrt(
            np.mean((df_type["target_water_level"] - df_type["water_level"]) ** 2)
        )
        mae_agg = np.mean(
            np.abs(df_type["target_water_level"] - df_type["water_level"])
        )
        r2_agg = r2_score(df_type["target_water_level"], df_type["water_level"])
        nse_agg = calculate_nse_numpy(
            df_type["water_level"].values,
            df_type["target_water_level"].values,
        )

        print("\n  AGGREGATED Metrics (all samples pooled):")
        print(f"    RMSE: {rmse_agg:.4f}")
        print(f"    MAE: {mae_agg:.4f}")
        print(f"    R²: {r2_agg:.4f}")
        print(f"    NSE: {nse_agg:.4f}")

        # PER-NODE metrics (calculate for each node separately)
        print("\n  PER-NODE Metric Distributions:")
        per_node_rmse = []
        per_node_mae = []
        per_node_r2 = []
        per_node_nse = []

        for node_id in df_type["node_id"].unique():
            node_data = df_type[df_type["node_id"] == node_id]

            # Calculate metrics for this node
            rmse = np.sqrt(
                np.mean(
                    (node_data["target_water_level"] - node_data["water_level"]) ** 2
                )
            )
            mae = np.mean(
                np.abs(node_data["target_water_level"] - node_data["water_level"])
            )

            # Only calculate R² and NSE if we have enough samples and variance
            if len(node_data) > 1:
                target_var = np.var(node_data["target_water_level"])
                if target_var > 1e-10:  # Avoid division by zero
                    r2 = r2_score(
                        node_data["target_water_level"],
                        node_data["water_level"],
                    )
                    nse = calculate_nse_numpy(
                        node_data["water_level"].values,
                        node_data["target_water_level"].values,
                    )
                else:
                    r2 = np.nan  # Skip nodes with no variance
                    nse = np.nan
            else:
                r2 = np.nan
                nse = np.nan

            per_node_rmse.append(rmse)
            per_node_mae.append(mae)
            if not np.isnan(r2):
                per_node_r2.append(r2)
            if not np.isnan(nse):
                per_node_nse.append(nse)

        # Print distribution statistics
        print("    RMSE per node:")
        print(f"      Mean: {np.mean(per_node_rmse):.4f}")
        print(f"      Median: {np.median(per_node_rmse):.4f}")
        print(f"      Min: {np.min(per_node_rmse):.4f}")
        print(f"      Max: {np.max(per_node_rmse):.4f}")
        print(f"      Std: {np.std(per_node_rmse):.4f}")

        print("    MAE per node:")
        print(f"      Mean: {np.mean(per_node_mae):.4f}")
        print(f"      Median: {np.median(per_node_mae):.4f}")
        print(f"      Min: {np.min(per_node_mae):.4f}")
        print(f"      Max: {np.max(per_node_mae):.4f}")
        print(f"      Std: {np.std(per_node_mae):.4f}")

        if len(per_node_r2) > 0:
            print("    R² per node:")
            print(f"      Mean: {np.mean(per_node_r2):.4f}")
            print(f"      Median: {np.median(per_node_r2):.4f}")
            print(f"      Min: {np.min(per_node_r2):.4f}")
            print(f"      Max: {np.max(per_node_r2):.4f}")
            print(f"      Std: {np.std(per_node_r2):.4f}")

            # Count problematic nodes
            poor_nodes = sum(1 for r2 in per_node_r2 if r2 < 0.5)
            negative_r2_nodes = sum(1 for r2 in per_node_r2 if r2 < 0)
            print(
                f"      Nodes with R² < 0.5: {poor_nodes} ({poor_nodes / len(per_node_r2) * 100:.1f}%)"
            )
            print(
                f"      Nodes with R² < 0 (negative): {negative_r2_nodes} ({negative_r2_nodes / len(per_node_r2) * 100:.1f}%)"
            )

        if len(per_node_nse) > 0:
            print("    NSE per node:")
            print(f"      Mean: {np.mean(per_node_nse):.4f}")
            print(f"      Median: {np.median(per_node_nse):.4f}")
            print(f"      Min: {np.min(per_node_nse):.4f}")
            print(f"      Max: {np.max(per_node_nse):.4f}")
            print(f"      Std: {np.std(per_node_nse):.4f}")

            # Count problematic nodes
            poor_nse_nodes = sum(1 for nse in per_node_nse if nse < 0.5)
            negative_nse_nodes = sum(1 for nse in per_node_nse if nse < 0)
            print(
                f"      Nodes with NSE < 0.5: {poor_nse_nodes} ({poor_nse_nodes / len(per_node_nse) * 100:.1f}%)"
            )
            print(
                f"      Nodes with NSE < 0 (negative): {negative_nse_nodes} ({negative_nse_nodes / len(per_node_nse) * 100:.1f}%)"
            )


def plot_per_node_metric_distributions(csv_path="gru_test_predictions_test.csv"):
    """
    Plot histograms showing the distribution of metrics across individual nodes.
    This helps identify if some nodes perform much worse than others.
    """
    df = pd.read_csv(csv_path)

    csv_path_obj = Path(csv_path)
    output_path = csv_path_obj.with_suffix("") / "per_node_metric_distributions.png"
    output_path.parent.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(2, 4, figsize=(18, 10))

    for node_type_idx, node_type in enumerate([1, 2]):
        node_type_str = "1D" if node_type == 1 else "2D"
        df_type = df[df["node_type"] == node_type]

        per_node_rmse = []
        per_node_mae = []
        per_node_r2 = []
        per_node_nse = []

        # Calculate per-node metrics
        for node_id in df_type["node_id"].unique():
            node_data = df_type[df_type["node_id"] == node_id]

            rmse = np.sqrt(
                np.mean(
                    (node_data["target_water_level"] - node_data["water_level"]) ** 2
                )
            )
            mae = np.mean(
                np.abs(node_data["target_water_level"] - node_data["water_level"])
            )
            nse = calculate_nse_numpy(
                node_data["water_level"].values,
                node_data["target_water_level"].values,
            )

            if len(node_data) > 1:
                target_var = np.var(node_data["target_water_level"])
                if target_var > 1e-10:
                    r2 = r2_score(
                        node_data["target_water_level"],
                        node_data["water_level"],
                    )
                    per_node_r2.append(r2)

            per_node_rmse.append(rmse)
            per_node_mae.append(mae)
            per_node_nse.append(nse)

            per_node_nse_array = np.asarray(per_node_nse)
            mask = np.isfinite(per_node_nse_array)
            per_node_nse_clean = per_node_nse_array[mask]

        # Plot RMSE distribution
        ax_rmse = axes[node_type_idx, 0]
        ax_rmse.hist(per_node_rmse, bins=30, color="blue", alpha=0.7, edgecolor="black")
        ax_rmse.axvline(
            np.mean(per_node_rmse),
            color="red",
            linestyle="--",
            linewidth=2,
            label=f"Mean: {np.mean(per_node_rmse):.4f}",
        )
        ax_rmse.axvline(
            np.median(per_node_rmse),
            color="green",
            linestyle="--",
            linewidth=2,
            label=f"Median: {np.median(per_node_rmse):.4f}",
        )
        ax_rmse.set_title(
            f"{node_type_str} Nodes - RMSE Distribution", fontweight="bold"
        )
        ax_rmse.set_xlabel("RMSE")
        ax_rmse.set_ylabel("Number of Nodes")
        ax_rmse.legend()
        ax_rmse.grid(True, alpha=0.3)

        # Plot MAE distribution
        ax_mae = axes[node_type_idx, 1]
        ax_mae.hist(per_node_mae, bins=30, color="orange", alpha=0.7, edgecolor="black")
        ax_mae.axvline(
            np.mean(per_node_mae),
            color="red",
            linestyle="--",
            linewidth=2,
            label=f"Mean: {np.mean(per_node_mae):.4f}",
        )
        ax_mae.axvline(
            np.median(per_node_mae),
            color="green",
            linestyle="--",
            linewidth=2,
            label=f"Median: {np.median(per_node_mae):.4f}",
        )
        ax_mae.set_title(f"{node_type_str} Nodes - MAE Distribution", fontweight="bold")
        ax_mae.set_xlabel("MAE")
        ax_mae.set_ylabel("Number of Nodes")
        ax_mae.legend()
        ax_mae.grid(True, alpha=0.3)

        # Plot R² distribution
        ax_r2 = axes[node_type_idx, 2]
        if len(per_node_r2) > 0:
            ax_r2.hist(
                per_node_r2, bins=30, color="green", alpha=0.7, edgecolor="black"
            )
            ax_r2.axvline(
                np.mean(per_node_r2),
                color="red",
                linestyle="--",
                linewidth=2,
                label=f"Mean: {np.mean(per_node_r2):.4f}",
            )
            ax_r2.axvline(
                np.median(per_node_r2),
                color="blue",
                linestyle="--",
                linewidth=2,
                label=f"Median: {np.median(per_node_r2):.4f}",
            )
            ax_r2.axvline(0, color="black", linestyle="-", linewidth=1, alpha=0.5)

            # Highlight poor performance regions
            poor_count = sum(1 for r2 in per_node_r2 if r2 < 0.5)
            if poor_count > 0:
                ax_r2.axvspan(
                    -1,
                    0.5,
                    alpha=0.2,
                    color="red",
                    label=f"R² < 0.5: {poor_count} nodes",
                )

            ax_r2.set_title(
                f"{node_type_str} Nodes - R² Distribution", fontweight="bold"
            )
            ax_r2.set_xlabel("R²")
            ax_r2.set_ylabel("Number of Nodes")
            ax_r2.legend()
            ax_r2.grid(True, alpha=0.3)
        else:
            ax_r2.text(
                0.5,
                0.5,
                "No R² data",
                ha="center",
                va="center",
                transform=ax_r2.transAxes,
            )

        # Plot NSE distribution
        ax_nse = axes[node_type_idx, 3]
        if len(per_node_nse_clean) > 0:
            ax_nse.hist(
                per_node_nse_clean,
                bins=30,
                color="purple",
                alpha=0.7,
                edgecolor="black",
            )
            ax_nse.axvline(
                np.mean(per_node_nse_clean),
                color="red",
                linestyle="--",
                linewidth=2,
                label=f"Mean: {np.mean(per_node_nse_clean):.4f}",
            )
            ax_nse.axvline(
                np.median(per_node_nse_clean),
                color="blue",
                linestyle="--",
                linewidth=2,
                label=f"Median: {np.median(per_node_nse_clean):.4f}",
            )
            ax_nse.axvline(0, color="black", linestyle="-", linewidth=1, alpha=0.5)

            # Highlight poor performance regions
            poor_count = sum(1 for nse in per_node_nse_clean if nse < 0.5)
            if poor_count > 0:
                ax_nse.axvspan(
                    -1,
                    0.5,
                    alpha=0.2,
                    color="red",
                    label=f"NSE < 0.5: {poor_count} nodes",
                )

            ax_nse.set_title(
                f"{node_type_str} Nodes - NSE Distribution", fontweight="bold"
            )
            ax_nse.set_xlabel("NSE")
            ax_nse.set_ylabel("Number of Nodes")
            ax_nse.legend()
            ax_nse.grid(True, alpha=0.3)
        else:
            ax_nse.text(
                0.5,
                0.5,
                "No NSE data",
                ha="center",
                va="center",
                transform=ax_nse.transAxes,
            )

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    print("✓ Metric distributions saved as 'per_node_metric_distributions.png'")
    plt.close()


def explain_rmse_calculation(
    csv_path="gru_test_predictions_test.csv",
    show_top_contributors=20,
    std_dev_dict=None,
):
    """
    Detailed breakdown of how hierarchical RMSE is calculated.
    Shows per-node RMSE, per-event RMSE, and overall metrics.

    Args:
        csv_path: Path to predictions CSV file
        show_top_contributors: Number of nodes to show in detailed breakdown
        std_dev_dict: Dictionary of standard deviations for standardization
                     Format: {(model_id, node_type): std_dev}
    """
    # Default std_dev values from the evaluation script
    if std_dev_dict is None:
        std_dev_dict = {
            (1, 1): 16.877747,  # Model 1, Node type 1
            (1, 2): 14.378797,  # Model 1, Node type 2
            (2, 1): 3.191784,  # Model 2, Node type 1
            (2, 2): 2.727131,  # Model 2, Node type 2
        }

    df = pd.read_csv(csv_path)

    # Check if event_id exists, if not create a single event
    if "event_id" not in df.columns:
        df["event_id"] = 1
        print("Note: No event_id column found, treating all data as single event\n")

    # Check if model_id exists, if not default to 1
    if "model_id" not in df.columns:
        df["model_id"] = 1
        print("Note: No model_id column found, defaulting to model_id=1\n")

    print("=" * 80)
    print("HIERARCHICAL RMSE CALCULATION BREAKDOWN")
    print("=" * 80)
    print("\nCalculation Structure:")
    print("  1. Per-node RMSE: RMSE for each node within an event")
    print("  2. Per-event Standardized RMSE: Average of standardized per-node RMSEs")
    print("  3. Per-model RMSE: Average across all events for each model")
    print("  4. Final Score: Average across all models")
    print("\nFormulas:")
    print("  RMSE = √(mean((y_true - y_pred)²))")
    print("  Standardized RMSE = RMSE / std_dev")
    print("  std_dev is specific to (model_id, node_type) combination\n")

    # Overall statistics
    print("=" * 80)
    print("DATASET OVERVIEW")
    print("=" * 80)
    print(f"Total samples: {len(df):,}")
    print(f"Models: {sorted(df['model_id'].unique())}")
    print(f"Events: {sorted(df['event_id'].unique())}")
    print(f"Node types: {sorted(df['node_type'].unique())}")
    print(f"Unique nodes: {df['node_id'].nunique()}")

    # Process each model
    model_scores = []

    for model_id in sorted(df["model_id"].unique()):
        df_model = df[df["model_id"] == model_id]

        print("\n" + "=" * 80)
        print(f"MODEL {model_id} ANALYSIS")
        print("=" * 80)

        event_std_rmses = []

        # Process each event
        for event_id in sorted(df_model["event_id"].unique()):
            df_event = df_model[df_model["event_id"] == event_id]

            print(f"\n--- Event {event_id} (Model {model_id}) ---")
            print(f"Total samples in event: {len(df_event):,}")

            # Process each node type
            node_type_std_rmses = []

            for node_type in [1, 2]:
                df_type = df_event[df_event["node_type"] == node_type]

                if len(df_type) == 0:
                    continue

                node_type_str = "1D" if node_type == 1 else "2D"
                std_dev = std_dev_dict.get((model_id, node_type), np.nan)

                print(f"\n  {node_type_str} Nodes:")
                print(f"    Standard deviation: {std_dev:.6f}")
                print(f"    Total samples: {len(df_type):,}")
                print(f"    Unique nodes: {df_type['node_id'].nunique()}")

                # Calculate per-node RMSE
                node_rmses = []
                node_std_rmses = []
                node_details = []

                for node_id in df_type["node_id"].unique():
                    node_data = df_type[df_type["node_id"] == node_id]

                    if len(node_data) <= 1:
                        continue

                    targets = node_data["target_water_level"].values
                    preds = node_data["water_level"].values

                    # Calculate RMSE for this node
                    node_rmse = rmse(targets, preds)
                    node_rmses.append(node_rmse)

                    # Calculate standardized RMSE
                    node_std_rmse = standardized_rmse(targets, preds, std_dev)
                    if not np.isnan(node_std_rmse):
                        node_std_rmses.append(node_std_rmse)

                    # Collect details
                    node_details.append(
                        {
                            "node_id": node_id,
                            "n_samples": len(targets),
                            "target_mean": targets.mean(),
                            "pred_mean": preds.mean(),
                            "target_std": targets.std(),
                            "pred_std": preds.std(),
                            "rmse": node_rmse,
                            "std_rmse": node_std_rmse,
                            "mae": np.mean(np.abs(targets - preds)),
                        }
                    )

                if node_std_rmses:
                    # Average standardized RMSE for this node type
                    avg_std_rmse = np.mean(node_std_rmses)
                    node_type_std_rmses.append(avg_std_rmse)

                    print(
                        f"    Per-node RMSE range: {min(node_rmses):.4f} to {max(node_rmses):.4f}"
                    )
                    print(
                        f"    Per-node Std RMSE range: {min(node_std_rmses):.4f} to {max(node_std_rmses):.4f}"
                    )
                    print(
                        f"    Average Std RMSE for {node_type_str}: {avg_std_rmse:.6f}"
                    )

                    # Show top contributors (worst performing nodes)
                    if len(node_details) > 0:
                        details_df = pd.DataFrame(node_details)
                        details_df = details_df.sort_values("std_rmse", ascending=False)

                        print(
                            f"\n    Top {min(show_top_contributors, len(details_df))} worst performing nodes:"
                        )
                        print(
                            f"    {'Node':>6} | {'Samples':>7} | {'RMSE':>8} | {'Std RMSE':>9} | {'MAE':>8}"
                        )
                        print("    " + "-" * 58)

                        for idx, row in details_df.head(
                            show_top_contributors
                        ).iterrows():
                            print(
                                f"    {row['node_id']:6.0f} | {row['n_samples']:7.0f} | "
                                f"{row['rmse']:8.4f} | {row['std_rmse']:9.6f} | {row['mae']:8.4f}"
                            )

            # Calculate event-level standardized RMSE
            if node_type_std_rmses:
                event_std_rmse = np.mean(node_type_std_rmses)
                event_std_rmses.append(event_std_rmse)
                print(f"\n  Event {event_id} Standardized RMSE: {event_std_rmse:.6f}")

        # Calculate model-level score
        if event_std_rmses:
            model_score = np.mean(event_std_rmses)
            model_scores.append(model_score)

            print(f"\n{'=' * 80}")
            print(f"MODEL {model_id} SUMMARY:")
            print(f"  Events processed: {len(event_std_rmses)}")
            print(
                f"  Event Std RMSE range: {min(event_std_rmses):.6f} to {max(event_std_rmses):.6f}"
            )
            print(f"  Model {model_id} Average Standardized RMSE: {model_score:.6f}")
            print(f"{'=' * 80}")

    # Final score
    if model_scores:
        final_score = np.mean(model_scores)

        print("\n" + "=" * 80)
        print("FINAL SCORE")
        print("=" * 80)
        print(f"Number of models: {len(model_scores)}")
        for i, score in enumerate(model_scores, 1):
            print(f"Model {i} Standardized RMSE: {score:.6f}")
        print(f"\nFinal Score (average across models): {final_score:.6f}")
        print("=" * 80)

        return final_score
    else:
        print("\n⚠️ Could not calculate final score - no valid events processed")
        return np.nan
    
def plot_multi_csv_comparison(
    csv_paths,
    csv_labels=None,
    csv_colors=None,
    csv_linestyles=None,
    csv_markers=None,
):
    """
    Compare predictions from multiple CSV files for specific nodes.
    Creates 3 figures, each comparing 5 different results + ground truth.
    
    Args:
        csv_paths: List of paths to 5 CSV files
        csv_labels: List of 5 labels for each CSV (default: "Result 1", "Result 2", etc.)
        csv_colors: List of 5 colors for each CSV (default: red, green, orange, purple, brown)
        csv_linestyles: List of 5 linestyles (default: '-', '--', '-.', ':', '-')
        csv_markers: List of 5 markers (default: 'x', '^', 's', 'D', 'v')
    """
    
    # ========================
    # DEFAULT CONFIGURATION
    # ========================
    if len(csv_paths) != 5:
        raise ValueError(f"Expected 5 CSV paths, got {len(csv_paths)}")
    
    if csv_labels is None:
        csv_labels = [f"Result {i+1}" for i in range(5)]
    
    if csv_colors is None:
        csv_colors = ['red', 'green', 'orange', 'purple', 'brown']
    
    if csv_linestyles is None:
        csv_linestyles = ['-', '--', '-.', ':', '-']
    
    if csv_markers is None:
        csv_markers = ['x', '^', 's', 'D', 'v']
    
    # Fixed parameters for all plots
    model_id = 1
    event_id = 8
    
    # Define the 3 nodes to plot
    nodes_to_plot = [
        {"node_type": 1, "node_id": 9, "title": "1D Node 9"},
        {"node_type": 2, "node_id": 2854, "title": "2D Node 2854"},
        {"node_type": 2, "node_id": 2960, "title": "2D Node 2960"},
    ]
    
    # Output directory (based on first CSV path)
    csv_path_obj = Path(csv_paths[0])
    output_dir = csv_path_obj.parent / "multi_csv_comparison_test"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"\n{'=' * 60}")
    print(f"Multi-CSV Comparison for Model {model_id}, Event {event_id}")
    print(f"{'=' * 60}")
    
    # Load all CSV files
    dfs = []
    for i, csv_path in enumerate(csv_paths):
        df = pd.read_csv(csv_path)
        dfs.append(df)
        print(f"  Loaded {csv_labels[i]}: {len(df)} samples from {csv_path}")
    
    # ========================
    # Create a figure for each node
    # ========================
    for node_config in nodes_to_plot:
        node_type = node_config["node_type"]
        node_id = node_config["node_id"]
        node_title = node_config["title"]
        
        print(f"\n{'-' * 60}")
        print(f"Plotting {node_title} (Type {node_type}, ID {node_id})")
        print(f"{'-' * 60}")
        
        # Create figure
        fig, ax = plt.subplots(figsize=(14, 6))
        
        ground_truth_plotted = False
        
        # Plot data from each CSV
        for i, (df, label, color, linestyle, marker) in enumerate(
            zip(dfs, csv_labels, csv_colors, csv_linestyles, csv_markers)
        ):
            # Get data for this specific node
            node_data = df[
                (df["node_id"] == node_id)
                & (df["model_id"] == model_id)
                & (df["event_id"] == event_id)
                & (df["node_type"] == node_type)
            ].sort_values("timestep")
            
            if len(node_data) > 0:
                timesteps = node_data["timestep"].values
                
                # Plot ground truth only once (from first CSV)
                if not ground_truth_plotted:
                    ax.plot(
                        timesteps,
                        node_data["target_water_level"],
                        label="Ground Truth",
                        color="blue",
                        marker="o",
                        linewidth=2.5,
                        markersize=5,
                        alpha=0.9,
                        zorder=10,  # Draw on top
                    )
                    ground_truth_plotted = True
                
                # Plot predictions from this CSV
                ax.plot(
                    timesteps,
                    node_data["water_level"],
                    label=label,
                    color=color,
                    marker=marker,
                    linewidth=2,
                    markersize=4,
                    alpha=0.8,
                    linestyle=linestyle,
                )
                
                # Calculate metrics for this CSV
                rmse = np.sqrt(
                    np.mean(
                        (node_data["target_water_level"] - node_data["water_level"]) ** 2
                    )
                )
                mae = np.mean(
                    np.abs(node_data["target_water_level"] - node_data["water_level"])
                )
                r2 = r2_score(
                    node_data["target_water_level"], node_data["water_level"]
                )
                
                print(f"  ✓ {label}: {len(node_data)} timesteps | RMSE: {rmse:.4f} | MAE: {mae:.4f} | R²: {r2:.4f}")
            else:
                print(f"  ✗ {label}: No data found")
        
        # Set title and labels
        ax.set_title(
            f"{node_title} - Model {model_id}, Event {event_id} - Multi-CSV Comparison",
            fontsize=13,
            fontweight="bold",
            pad=15,
        )
        ax.set_xlabel("Timestep", fontsize=11)
        ax.set_ylabel("Water Level", fontsize=11)
        
        # Legend with better positioning
        ax.legend(
            loc="best",
            fontsize=10,
            framealpha=0.9,
            edgecolor='black',
        )
        
        ax.grid(True, alpha=0.3, linestyle='--')
        
        # Optional: Uncomment to set y-axis limits
        # if node_type == 1:
        #     ax.set_ylim([180, 380])
        # else:
        #     ax.set_ylim([290, 360])
        
        # Save figure
        output_filename = f"comparison_model{model_id}_event{event_id}_type{node_type}_node{node_id}.png"
        output_path = output_dir / output_filename
        
        plt.tight_layout()
        plt.savefig(output_path, dpi=300, bbox_inches="tight")
        print(f"  ✓ Saved: {output_path}")
        plt.close()
    
    print(f"\n{'=' * 60}")
    print(f"All comparison plots completed!")
    print(f"Output directory: {output_dir}")
    print(f"{'=' * 60}")


def concatenate_ground_truth(csv_file, gt_file, output_path):
    """
    Concatenate ground truth data to predictions file.
    Handles both CSV and Parquet formats for input files.
    Always saves output as CSV.
    
    Args:
        csv_file: Path to original predictions file (CSV or Parquet)
        gt_file: Path to ground truth file (CSV or Parquet)
        output_path: Path to save the combined CSV file
    """
    if csv_file.endswith('.parquet'):
        ori_file = pd.read_parquet(csv_file)
    else:
        ori_file = pd.read_csv(csv_file)
    
    if gt_file.endswith('.parquet'):
        gt_file_data = pd.read_parquet(gt_file)
    else:
        gt_file_data = pd.read_csv(gt_file)
    
    ori_file["timestep"] = gt_file_data["timestep"]
    ori_file["target_water_level"] = gt_file_data["target_water_level"]
    
    ori_file.to_csv(output_path, index=False)
    print(f"✓ Saved combined data to {output_path}")

def parse_args() -> Namespace:
    parser = ArgumentParser(description="")
    parser.add_argument(
        "--input_csv",
        type=str,
        required=True,
        help="Path to model1 and model2 csv file",
    )
    parser.add_argument(
        "--gt_csv", type=str, required=True, help="Path to csv file having ground truth"
    )
    parser.add_argument(
        "--output_csv",
        type=str,
        required=True,
        help="Path to concatenated input and gt csv",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=("cuda" if torch.cuda.is_available() else "cpu"),
        help="Device to run on",
    )
    return parser.parse_args()


# def main():
#     args = parse_args()

#     concatenate_ground_truth(
#         args.input_csv,
#         args.gt_csv,
#         args.output_csv
#     )

#     print("Analyzing node-level predictions...")
#     print("\n" + "=" * 80)
#     print("NODE STATISTICS (Aggregated vs Per-Node)")
#     print("=" * 80)
#     analyze_node_statistics(csv_path=args.output_csv)

#     print("\n" + "=" * 80)
#     print("PER-NODE METRIC DISTRIBUTIONS")
#     print("=" * 80)
#     plot_per_node_metric_distributions(csv_path=args.output_csv)

#     print("\n" + "=" * 80)
#     print("PLOTTING INDIVIDUAL NODE TIME SERIES (10 random nodes)")
#     print("=" * 80)
#     plot_individual_node_timeseries(csv_path=args.output_csv, num_nodes=10)

#     print("\n" + "=" * 80)
#     print("EXAMPLE: Plot a specific node")
#     print("=" * 80)
#     print("To plot a specific node, use:")
#     print("  plot_specific_node('gru_test_predictions_test.csv', node_id=42)")

#     explain_rmse_calculation(csv_path=args.output_csv, show_top_contributors=20)

#     print("\n✓ All visualizations complete!")


def main():
    # input_csv = "kaggle_submissions/node_only_8_sorted.csv"
    # gt_csv = "kaggle_submissions/ground_truth.csv"
    # output_csv = "kaggle_submissions/node_only_8_sorted_gt.csv"

    # concatenate_ground_truth(
    #     input_csv,
    #     gt_csv,
    #     output_csv
    # )

    # print("Analyzing node-level predictions...")
    # print("\n" + "=" * 80)
    # print("NODE STATISTICS (Aggregated vs Per-Node)")
    # print("=" * 80)
    # analyze_node_statistics(csv_path=output_csv)

    # print("\n" + "=" * 80)
    # print("PER-NODE METRIC DISTRIBUTIONS")
    # print("=" * 80)
    # plot_per_node_metric_distributions(csv_path=output_csv)

    # print("\n" + "=" * 80)
    # print("PLOTTING INDIVIDUAL NODE TIME SERIES (10 random nodes)")
    # print("=" * 80)
    # plot_individual_node_timeseries(csv_path=output_csv)

    # print("\n" + "=" * 80)
    # print("EXAMPLE: Plot a specific node")
    # print("=" * 80)
    # print("To plot a specific node, use:")
    # print("  plot_specific_node('gru_test_predictions_test.csv', node_id=42)")

    # explain_rmse_calculation(csv_path=output_csv, show_top_contributors=20)

    # Define your 5 CSV file paths
    csv_paths = [
        "kaggle_submissions/muhammedkaya_gt.csv",
        "kaggle_submissions/timotheehenry_gt.csv",
        "kaggle_submissions/nomagic_gt.csv",
        "kaggle_submissions/dmytro_gt.csv",
        "kaggle_submissions/gegerout_gt.csv",
    ]
    
    # Optional: Customize labels and styles
    csv_labels = [
        "Top 1 - 0.0562",
        "Top 2 - 0.0571",
        "Top 3 - 0.0643",
        "Top 4 - 0.0766",
        "Top 5 - 0.0799",
    ]
    
    csv_colors = ['red', 'green', 'orange', 'purple', 'brown']
    csv_linestyles = ['-', '--', '-.', ':', '-']
    csv_markers = ['x', '^', 's', 'D', 'v']
    
    # Run the comparison
    plot_multi_csv_comparison(
        csv_paths=csv_paths,
        csv_labels=csv_labels,
        csv_colors=csv_colors,
        csv_linestyles=csv_linestyles,
        csv_markers=csv_markers,
    )

    print("\n✓ All visualizations complete!")


if __name__ == "__main__":
    main()
