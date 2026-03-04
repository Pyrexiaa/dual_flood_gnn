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
    pred_csv_path,
    gt_csv_path,
    show_top_contributors=20,
    std_dev_dict=None,
):
    """
    Detailed breakdown of how hierarchical RMSE is calculated, split into 6 readings:
      1. Public Overall Standardized RMSE
      2. Private Overall Standardized RMSE
      3. Public 1D (node_type=1) Standardized RMSE
      4. Public 2D (node_type=2) Standardized RMSE
      5. Private 1D (node_type=1) Standardized RMSE
      6. Private 2D (node_type=2) Standardized RMSE

    Args:
        pred_csv_path: Path to predictions CSV file
                       Expected columns: row_id, model_id, event_id, node_type, node_id, water_level
        gt_csv_path:   Path to ground truth CSV file
                       Expected columns: row_id, model_id, event_id, node_type, node_id,
                                         target_water_level, Usage
        show_top_contributors: Number of worst-performing nodes to show per group
        std_dev_dict: Dictionary of standard deviations for standardization
                      Format: {(model_id, node_type): std_dev}

    Returns:
        dict with keys:
            public_overall, private_overall,
            public_1d, public_2d,
            private_1d, private_2d
    """
    # ------------------------------------------------------------------ #
    # Default std_dev values
    # ------------------------------------------------------------------ #
    if std_dev_dict is None:
        std_dev_dict = {
            (1, 1): 16.877747,
            (1, 2): 14.378797,
            (2, 1): 3.191784,
            (2, 2): 2.727131,
        }

    # ------------------------------------------------------------------ #
    # Load & merge
    # ------------------------------------------------------------------ #
    pred_df = pd.read_csv(pred_csv_path)
    gt_df = pd.read_csv(gt_csv_path)
    gt_df.rename(columns={"water_level": "target_water_level"}, inplace=True)

    # Identify the join keys available in both files
    join_keys = ["row_id"]
    for col in ["model_id", "event_id", "node_type", "node_id"]:
        if col in pred_df.columns and col in gt_df.columns:
            join_keys.append(col)

    # Keep only necessary columns from each side before merging
    gt_cols = join_keys + [
        c for c in ["target_water_level", "Usage"] if c in gt_df.columns
    ]
    pred_cols = join_keys + ["water_level"]

    df = pd.merge(
        pred_df[pred_cols],
        gt_df[gt_cols],
        on=join_keys,
        how="inner",
    )

    # Fallbacks if certain columns are absent
    if "event_id" not in df.columns:
        df["event_id"] = 1
        print("Note: No event_id column - treating all data as single event\n")
    if "model_id" not in df.columns:
        df["model_id"] = 1
        print("Note: No model_id column - defaulting to model_id=1\n")
    if "Usage" not in df.columns:
        df["Usage"] = "Public"
        print("Note: No Usage column - treating all rows as Public\n")

    print("=" * 80)
    print("HIERARCHICAL RMSE CALCULATION BREAKDOWN")
    print("=" * 80)
    print(f"Total rows after merge  : {len(df):,}")
    print(f"Models                  : {sorted(df['model_id'].unique())}")
    print(f"Events                  : {sorted(df['event_id'].unique())}")
    print(f"Node types              : {sorted(df['node_type'].unique())}")
    print(f"Unique nodes            : {df['node_id'].nunique()}")
    print(f"Usage split             : {df['Usage'].value_counts().to_dict()}")

    # ------------------------------------------------------------------ #
    # Helper: compute hierarchical standardized RMSE for a sub-dataframe
    # Hierarchy: model -> event -> node_type -> node -> average back up
    # ------------------------------------------------------------------ #
    def compute_hierarchical_std_rmse(sub_df, label=""):
        """Returns (overall_score, score_1d, score_2d)."""
        if sub_df.empty:
            return np.nan, np.nan, np.nan

        model_scores = []
        model_scores_1d = []
        model_scores_2d = []

        for model_id in sorted(sub_df["model_id"].unique()):
            df_model = sub_df[sub_df["model_id"] == model_id]
            event_std_rmses = []
            event_std_rmses_1d = []
            event_std_rmses_2d = []

            for event_id in sorted(df_model["event_id"].unique()):
                df_event = df_model[df_model["event_id"] == event_id]
                node_type_std_rmses = []

                nt_scores = {}  # node_type -> avg_std_rmse

                for node_type in [1, 2]:
                    df_type = df_event[df_event["node_type"] == node_type]
                    if df_type.empty:
                        continue

                    std_dev = std_dev_dict.get((model_id, node_type), np.nan)
                    node_std_rmses = []

                    for node_id in df_type["node_id"].unique():
                        node_data = df_type[df_type["node_id"] == node_id]
                        if len(node_data) <= 1:
                            continue
                        targets = node_data["target_water_level"].values
                        preds = node_data["water_level"].values
                        nsr = standardized_rmse(targets, preds, std_dev)
                        if not np.isnan(nsr):
                            node_std_rmses.append(nsr)

                    if node_std_rmses:
                        avg = np.mean(node_std_rmses)
                        node_type_std_rmses.append(avg)
                        nt_scores[node_type] = avg

                if node_type_std_rmses:
                    event_std_rmses.append(np.mean(node_type_std_rmses))
                if 1 in nt_scores:
                    event_std_rmses_1d.append(nt_scores[1])
                if 2 in nt_scores:
                    event_std_rmses_2d.append(nt_scores[2])

            if event_std_rmses:
                model_scores.append(np.mean(event_std_rmses))
            if event_std_rmses_1d:
                model_scores_1d.append(np.mean(event_std_rmses_1d))
            if event_std_rmses_2d:
                model_scores_2d.append(np.mean(event_std_rmses_2d))

        overall = np.mean(model_scores) if model_scores else np.nan
        score_1d = np.mean(model_scores_1d) if model_scores_1d else np.nan
        score_2d = np.mean(model_scores_2d) if model_scores_2d else np.nan
        return overall, score_1d, score_2d

    # ------------------------------------------------------------------ #
    # Detailed per-node breakdown (mirrors original function)
    # ------------------------------------------------------------------ #
    def print_detailed_breakdown(sub_df, label):
        print("\n" + "=" * 80)
        print(f"DETAILED BREAKDOWN — {label.upper()}")
        print("=" * 80)

        for model_id in sorted(sub_df["model_id"].unique()):
            df_model = sub_df[sub_df["model_id"] == model_id]
            print(f"\n  Model {model_id}")

            for event_id in sorted(df_model["event_id"].unique()):
                df_event = df_model[df_model["event_id"] == event_id]
                print(f"\n    Event {event_id}  ({len(df_event):,} samples)")

                for node_type in [1, 2]:
                    df_type = df_event[df_event["node_type"] == node_type]
                    if df_type.empty:
                        continue

                    nt_str = "1D" if node_type == 1 else "2D"
                    std_dev = std_dev_dict.get((model_id, node_type), np.nan)
                    node_details = []

                    for node_id in df_type["node_id"].unique():
                        nd = df_type[df_type["node_id"] == node_id]
                        if len(nd) <= 1:
                            continue
                        t = nd["target_water_level"].values
                        p = nd["water_level"].values
                        nr = rmse(t, p)
                        ns = standardized_rmse(t, p, std_dev)
                        node_details.append(
                            {
                                "node_id": node_id,
                                "n_samples": len(t),
                                "rmse": nr,
                                "std_rmse": ns,
                                "mae": np.mean(np.abs(t - p)),
                            }
                        )

                    if not node_details:
                        continue

                    details_df = pd.DataFrame(node_details).sort_values(
                        "std_rmse", ascending=False
                    )
                    avg_std = details_df["std_rmse"].mean()

                    print(f"\n      {nt_str} Nodes  |  std_dev={std_dev:.6f}  |  "
                          f"avg_std_rmse={avg_std:.6f}")
                    print(
                        f"      {'Node':>6} | {'Samples':>7} | {'RMSE':>8} | "
                        f"{'Std RMSE':>9} | {'MAE':>8}"
                    )
                    print("      " + "-" * 55)
                    for _, row in details_df.head(show_top_contributors).iterrows():
                        print(
                            f"      {row['node_id']:6.0f} | {row['n_samples']:7.0f} | "
                            f"{row['rmse']:8.4f} | {row['std_rmse']:9.6f} | "
                            f"{row['mae']:8.4f}"
                        )

    # ------------------------------------------------------------------ #
    # Split by Usage
    # ------------------------------------------------------------------ #
    public_df = df[df["Usage"].str.strip().str.lower() == "public"]
    private_df = df[df["Usage"].str.strip().str.lower() == "private"]

    # print_detailed_breakdown(public_df, "Public")
    # print_detailed_breakdown(private_df, "Private")

    # ------------------------------------------------------------------ #
    # Compute the 6 scores
    # ------------------------------------------------------------------ #
    pub_overall, pub_1d, pub_2d = compute_hierarchical_std_rmse(public_df, "Public")
    prv_overall, prv_1d, prv_2d = compute_hierarchical_std_rmse(private_df, "Private")

    results = {
        "public_overall": pub_overall,
        "private_overall": prv_overall,
        "public_1d": pub_1d,
        "public_2d": pub_2d,
        "private_1d": prv_1d,
        "private_2d": prv_2d,
    }

    # ------------------------------------------------------------------ #
    # Print summary
    # ------------------------------------------------------------------ #
    print("\n" + "=" * 80)
    print("FINAL 6 RMSE READINGS")
    print("=" * 80)
    print(f"  {'Metric':<30} {'Standardized RMSE':>20}")
    print("  " + "-" * 52)
    print(f"  {'Public  — Overall':<30} {pub_overall:>20.6f}")
    print(f"  {'Public  — 1D (node_type=1)':<30} {pub_1d:>20.6f}")
    print(f"  {'Public  — 2D (node_type=2)':<30} {pub_2d:>20.6f}")
    print(f"  {'Private — Overall':<30} {prv_overall:>20.6f}")
    print(f"  {'Private — 1D (node_type=1)':<30} {prv_1d:>20.6f}")
    print(f"  {'Private — 2D (node_type=2)':<30} {prv_2d:>20.6f}")
    print("=" * 80)

    return results
    
def plot_multi_csv_comparison(
    csv_paths,
    csv_labels=None,
    csv_colors=None,
    csv_linestyles=None,
    csv_markers=None,
    output_dir=None,
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
        output_dir: Directory to save the plots (default: same directory as first CSV)
    """
    
    # ========================
    # DEFAULT CONFIGURATION
    # ========================
    if len(csv_paths) != 4:
        raise ValueError(f"Expected 4 CSV paths, got {len(csv_paths)}")
    
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
    if output_dir is None:
        output_dir = csv_path_obj.parent / "multi_csv_comparison_test"
        output_dir.mkdir(parents=True, exist_ok=True)
    else:
        output_dir = csv_path_obj.parent / f"{output_dir}"
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
            ]

            # Handle missing timestep column
            if "timestep" in df.columns:
                node_data = node_data.sort_values("timestep")
                timesteps = node_data["timestep"].values
            else:
                # Preserve original row order and generate synthetic timesteps
                node_data = node_data.copy()
                node_data = node_data.reset_index(drop=True)
                timesteps = np.arange(len(node_data))
            
            if len(node_data) > 0:
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
        xlabel = "Timestep" if "timestep" in dfs[0].columns else "Sequence Index"
        ax.set_xlabel(xlabel, fontsize=11)
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
    
    # Drop existing columns if they already exist
    if "target_water_level" in ori_file.columns:
        ori_file.drop(columns=["target_water_level"], inplace=True)
        print("⚠ Removed existing column: target_water_level")

    # ori_file["timestep"] = gt_file_data["timestep"]
    ori_file["target_water_level"] = gt_file_data["water_level"]
    
    ori_file.to_csv(output_path, index=False)
    print(f"✓ Saved combined data to {output_path}")
    
def sort_and_reset_csv(input_csv: str, output_csv: str):
    """
    Sorts a CSV by model_id, event_id, node_type, node_id,
    resets row_id from 0, and saves to a new file.

    Args:
        input_csv:  Path to the input CSV file.
        output_csv: Path to save the sorted CSV file.
    """
    df = pd.read_csv(input_csv)

    sort_keys = ["model_id", "event_id", "node_type", "node_id"]

    # Verify all sort keys exist
    for key in sort_keys:
        if key not in df.columns:
            raise ValueError(f"Missing sort key '{key}' in file")

    df_sorted = df.sort_values(by=sort_keys, ascending=True).reset_index(drop=True)

    # Reset row_id from 0
    df_sorted["row_id"] = df_sorted.index

    # Reorder columns to ensure row_id is first
    cols = ["row_id"] + [col for col in df_sorted.columns if col != "row_id"]
    df_sorted = df_sorted[cols]

    df_sorted.to_csv(output_csv, index=False)

    print(f"Original rows : {len(df)}")
    print(f"Sorted by     : {sort_keys}")
    print(f"Saved to      : {output_csv}")

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
#     plot_individual_node_timeseries(csv_path=args.output_csv)

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

    explain_rmse_calculation(
        pred_csv_path="kaggle_submissions/node_only_1_new_sorted.csv",
        gt_csv_path="kaggle_submissions/solutions_remapped_sorted.csv",
        show_top_contributors=20,
    )

    # # Define your 5 CSV file paths
    # csv_paths = [
    #     "kaggle_submissions/node_only_1_sorted_2.csv",
    #     "kaggle_submissions/node_only_1_v2_sorted_2.csv",
    #     "kaggle_submissions/node_edge_only_1_sorted_2.csv",
    #     "kaggle_submissions/node_only_1_hgnn_sorted_2.csv",
    # ]
    
    # # Optional: Customize labels and styles
    # csv_labels = [
    #     "Node Only - 6.2211",
    #     "Simplified Node Only - 36.5512",
    #     "Node Edge - 5.9299",
    #     "HGNN Node Edge - 1.6953",
    # ]
    
    # csv_colors = ['red', 'green', 'orange', "purple"]
    # csv_linestyles = ['-', '--', '-.', ':']
    # csv_markers = ['x', '^', 's', 'D']
    
    # # Run the comparison
    # plot_multi_csv_comparison(
    #     csv_paths=csv_paths,
    #     csv_labels=csv_labels,
    #     csv_colors=csv_colors,
    #     csv_linestyles=csv_linestyles,
    #     csv_markers=csv_markers,
    #     output_dir="multi_csv_comparison_baselines"
    # )

    # print("\n✓ All visualizations complete!")


if __name__ == "__main__":
    main()
