import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import r2_score
import torch
from torch import Tensor


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


def plot_individual_node_timeseries(
    csv_path="gru_test_predictions_test.csv", num_nodes=10
):
    """
    Plot complete time series for randomly selected individual nodes.
    Shows ground truth vs prediction across all timesteps for each node.

    Args:
        csv_path: Path to the predictions CSV file
        num_nodes: Number of random nodes to plot (5 from 1D, 5 from 2D)
    """
    # Load predictions
    df = pd.read_csv(csv_path)

    print(f"Total samples: {len(df)}")
    print(f"Columns: {df.columns.tolist()}")

    # Separate by node type
    df_1d = df[df["node_type"] == 0]
    df_2d = df[df["node_type"] == 1]

    print(f"\n1D nodes (type 0): {len(df_1d)} samples")
    print(f"2D nodes (type 1): {len(df_2d)} samples")

    # Get unique nodes
    unique_nodes_1d = df_1d["node_id"].unique()
    unique_nodes_2d = df_2d["node_id"].unique()

    print(f"\nUnique 1D nodes: {len(unique_nodes_1d)}")
    print(f"Unique 2D nodes: {len(unique_nodes_2d)}")

    # Randomly select nodes
    np.random.seed(42)
    num_per_type = num_nodes // 2
    selected_1d = np.random.choice(
        unique_nodes_1d, min(num_per_type, len(unique_nodes_1d)), replace=False
    )
    selected_2d = np.random.choice(
        unique_nodes_2d, min(num_per_type, len(unique_nodes_2d)), replace=False
    )

    print(f"\nSelected {len(selected_1d)} random 1D nodes: {selected_1d}")
    print(f"Selected {len(selected_2d)} random 2D nodes: {selected_2d}")

    # Create subplots - 2 columns (1D and 2D), rows = num_per_type
    fig, axes = plt.subplots(num_per_type, 2, figsize=(16, 4 * num_per_type))

    # Ensure axes is 2D even if only one row
    if num_per_type == 1:
        axes = axes.reshape(1, -1)

    # ========================
    # Plot 1D Nodes (Left Column)
    # ========================
    for idx, node_id in enumerate(selected_1d):
        ax = axes[idx, 0]

        # Get data for this specific node, sorted by timestep
        node_data = df_1d[df_1d["node_id"] == node_id].sort_values("timestep")

        if len(node_data) > 0:
            timesteps = node_data["timestep"].values

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
            ax.plot(
                timesteps,
                node_data["predicted_water_level"],
                label="Prediction",
                color="red",
                marker="x",
                linewidth=2,
                markersize=4,
                alpha=0.8,
                linestyle="--",
            )

            # Calculate metrics for this node
            rmse = np.sqrt(
                np.mean(
                    (
                        node_data["target_water_level"]
                        - node_data["predicted_water_level"]
                    )
                    ** 2
                )
            )
            mae = np.mean(
                np.abs(
                    node_data["target_water_level"] - node_data["predicted_water_level"]
                )
            )
            r2 = r2_score(
                node_data["target_water_level"], node_data["predicted_water_level"]
            )
            nse = calculate_nse_numpy(
                node_data["predicted_water_level"].values,
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
        else:
            ax.text(
                0.5,
                0.5,
                f"No data for node {node_id}",
                ha="center",
                va="center",
                transform=ax.transAxes,
            )

    # ========================
    # Plot 2D Nodes (Right Column)
    # ========================
    for idx, node_id in enumerate(selected_2d):
        ax = axes[idx, 1]

        # Get data for this specific node, sorted by timestep
        node_data = df_2d[df_2d["node_id"] == node_id].sort_values("timestep")

        if len(node_data) > 0:
            timesteps = node_data["timestep"].values

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
            ax.plot(
                timesteps,
                node_data["predicted_water_level"],
                label="Prediction",
                color="red",
                marker="x",
                linewidth=2,
                markersize=4,
                alpha=0.8,
                linestyle="--",
            )

            # Calculate metrics for this node
            rmse = np.sqrt(
                np.mean(
                    (
                        node_data["target_water_level"]
                        - node_data["predicted_water_level"]
                    )
                    ** 2
                )
            )
            mae = np.mean(
                np.abs(
                    node_data["target_water_level"] - node_data["predicted_water_level"]
                )
            )
            r2 = r2_score(
                node_data["target_water_level"], node_data["predicted_water_level"]
            )
            nse = calculate_nse_numpy(
                node_data["predicted_water_level"].values,
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
        else:
            ax.text(
                0.5,
                0.5,
                f"No data for node {node_id}",
                ha="center",
                va="center",
                transform=ax.transAxes,
            )

    plt.tight_layout()
    plt.savefig("individual_node_timeseries.png", dpi=300, bbox_inches="tight")
    print("\n✓ Plot saved as 'individual_node_timeseries.png'")
    plt.close()


def plot_specific_node(csv_path, node_id, save_path=None):
    """
    Plot time series for a specific node.

    Args:
        csv_path: Path to the predictions CSV file
        node_id: Specific node ID to plot
        save_path: Optional path to save the figure
    """
    df = pd.read_csv(csv_path)

    # Get data for this specific node
    node_data = df[df["node_id"] == node_id].sort_values("timestep")

    if len(node_data) == 0:
        print(f"No data found for node_id={node_id}")
        return

    node_type = node_data["node_type"].iloc[0]
    node_type_str = "1D" if node_type == 0 else "2D"

    plt.figure(figsize=(12, 6))

    timesteps = node_data["timestep"].values
    plt.plot(
        timesteps,
        node_data["target_water_level"],
        label="Ground Truth",
        color="blue",
        marker="o",
        linewidth=2.5,
        markersize=5,
        alpha=0.8,
    )
    plt.plot(
        timesteps,
        node_data["predicted_water_level"],
        label="Prediction",
        color="red",
        marker="x",
        linewidth=2.5,
        markersize=5,
        alpha=0.8,
        linestyle="--",
    )

    # Calculate metrics
    rmse = np.sqrt(
        np.mean(
            (node_data["target_water_level"] - node_data["predicted_water_level"]) ** 2
        )
    )
    mae = np.mean(
        np.abs(node_data["target_water_level"] - node_data["predicted_water_level"])
    )
    r2 = r2_score(node_data["target_water_level"], node_data["predicted_water_level"])
    nse = calculate_nse_numpy(
        node_data["predicted_water_level"].values,
        node_data["target_water_level"].values,
    )

    plt.title(
        f"{node_type_str} Node {node_id} - Water Level Prediction\n"
        f"RMSE: {rmse:.4f} | MAE: {mae:.4f} | R²: {r2:.4f} | NSE: {nse:.4f}",
        fontsize=14,
        fontweight="bold",
    )
    plt.xlabel("Timestep", fontsize=12)
    plt.ylabel("Water Level", fontsize=12)
    plt.legend(loc="best", fontsize=11)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        print(f"✓ Plot saved to {save_path}")

    plt.close()


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
    for node_type in [0, 1]:
        node_type_str = "1D" if node_type == 0 else "2D"
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
            np.mean(
                (df_type["target_water_level"] - df_type["predicted_water_level"]) ** 2
            )
        )
        mae_agg = np.mean(
            np.abs(df_type["target_water_level"] - df_type["predicted_water_level"])
        )
        r2_agg = r2_score(
            df_type["target_water_level"], df_type["predicted_water_level"]
        )
        nse_agg = calculate_nse_numpy(
            df_type["predicted_water_level"].values,
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
                    (
                        node_data["target_water_level"]
                        - node_data["predicted_water_level"]
                    )
                    ** 2
                )
            )
            mae = np.mean(
                np.abs(
                    node_data["target_water_level"] - node_data["predicted_water_level"]
                )
            )

            # Only calculate R² and NSE if we have enough samples and variance
            if len(node_data) > 1:
                target_var = np.var(node_data["target_water_level"])
                if target_var > 1e-10:  # Avoid division by zero
                    r2 = r2_score(
                        node_data["target_water_level"],
                        node_data["predicted_water_level"],
                    )
                    nse = calculate_nse_numpy(
                        node_data["predicted_water_level"].values,
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

    fig, axes = plt.subplots(2, 4, figsize=(18, 10))

    for node_type_idx, node_type in enumerate([0, 1]):
        node_type_str = "1D" if node_type == 0 else "2D"
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
                    (
                        node_data["target_water_level"]
                        - node_data["predicted_water_level"]
                    )
                    ** 2
                )
            )
            mae = np.mean(
                np.abs(
                    node_data["target_water_level"] - node_data["predicted_water_level"]
                )
            )
            nse = calculate_nse_numpy(
                node_data["predicted_water_level"].values,
                node_data["target_water_level"].values,
            )

            if len(node_data) > 1:
                target_var = np.var(node_data["target_water_level"])
                if target_var > 1e-10:
                    r2 = r2_score(
                        node_data["target_water_level"],
                        node_data["predicted_water_level"],
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
    plt.savefig("per_node_metric_distributions.png", dpi=300, bbox_inches="tight")
    print("✓ Metric distributions saved as 'per_node_metric_distributions.png'")
    plt.close()

def explain_nse_calculation(csv_path="gru_test_predictions_test.csv", show_top_contributors=20):
    """
    Detailed breakdown of how aggregated NSE achieves high scores despite poor per-node NSE.
    Shows which nodes contribute most to the numerator and denominator.
    """
    df = pd.read_csv(csv_path)
    
    print("=" * 80)
    print("DETAILED NSE CALCULATION BREAKDOWN")
    print("=" * 80)
    print("\nNSE = 1 - (SS_res / SS_tot)")
    print("  SS_res = Σ(target - pred)²     (residual sum of squares)")
    print("  SS_tot = Σ(target - mean)²     (total sum of squares)")
    print("\nFor AGGREGATED: mean = global mean across all samples")
    print("For PER-NODE: mean = that node's mean across its timesteps\n")
    
    for node_type in [0, 1]:
        node_type_str = "1D" if node_type == 0 else "2D"
        df_type = df[df['node_type'] == node_type]
        
        print("=" * 80)
        print(f"{node_type_str} NODES ANALYSIS")
        print("=" * 80)
        
        # AGGREGATED NSE CALCULATION
        all_targets = df_type['target_water_level'].values
        all_preds = df_type['predicted_water_level'].values
        
        global_mean = all_targets.mean()
        ss_res_agg = np.sum((all_targets - all_preds) ** 2)
        ss_tot_agg = np.sum((all_targets - global_mean) ** 2)
        nse_agg = 1 - (ss_res_agg / ss_tot_agg)
        
        print(f"\n1. AGGREGATED NSE (all {len(all_targets):,} samples pooled):")
        print(f"   Global mean: {global_mean:.4f}")
        print(f"   SS_res: {ss_res_agg:.2e}")
        print(f"   SS_tot: {ss_tot_agg:.2e}")
        print(f"   NSE: {nse_agg:.4f}")
        
        # PER-NODE CONTRIBUTIONS
        print(f"\n2. PER-NODE BREAKDOWN:")
        
        node_contributions = []
        
        for node_id in df_type['node_id'].unique():
            node_data = df_type[df_type['node_id'] == node_id]
            
            targets = node_data['target_water_level'].values
            preds = node_data['predicted_water_level'].values
            
            # Per-node NSE (using node's own mean)
            node_mean = targets.mean()
            ss_res_node = np.sum((targets - preds) ** 2)
            ss_tot_node = np.sum((targets - node_mean) ** 2)
            nse_node = 1 - (ss_res_node / ss_tot_node) if ss_tot_node > 1e-10 else np.nan
            
            # Contribution to aggregated calculation (using GLOBAL mean)
            ss_res_contrib = ss_res_node  # Same as per-node
            ss_tot_contrib = np.sum((targets - global_mean) ** 2)  # Different!
            
            # Variance measures
            target_variance = np.var(targets)
            pred_variance = np.var(preds)
            target_std = np.std(targets)
            pred_std = np.std(preds)
            
            node_contributions.append({
                'node_id': node_id,
                'n_samples': len(targets),
                'node_mean': node_mean,
                'deviation_from_global': abs(node_mean - global_mean),
                'target_variance': target_variance,
                'pred_variance': pred_variance,
                'target_std': target_std,
                'pred_std': pred_std,
                'ss_res_node': ss_res_node,
                'ss_tot_node': ss_tot_node,
                'ss_res_contrib': ss_res_contrib,
                'ss_tot_contrib': ss_tot_contrib,
                'nse_node': nse_node,
                'ss_tot_contrib_pct': (ss_tot_contrib / ss_tot_agg) * 100
            })
        
        # Convert to DataFrame for analysis
        contrib_df = pd.DataFrame(node_contributions)
        contrib_df = contrib_df.sort_values('ss_tot_contrib_pct', ascending=False)
        
        print(f"\n   Total nodes: {len(contrib_df)}")
        print(f"   Range of node means: {contrib_df['node_mean'].min():.2f} to {contrib_df['node_mean'].max():.2f}")
        print(f"   Global mean: {global_mean:.2f}")
        
        # Show top contributors to SS_tot
        print(f"\n3. TOP {show_top_contributors} NODES CONTRIBUTING TO HIGH AGGREGATED NSE:")
        print(f"   (These nodes have water levels far from global mean)")
        print(f"\n   {'Node':>6} | {'Mean':>8} | {'Δ Global':>8} | {'Var':>8} | {'SS_tot %':>9} | {'NSE_node':>9}")
        print("   " + "-" * 72)
        
        for idx, row in contrib_df.head(show_top_contributors).iterrows():
            print(f"   {row['node_id']:6.0f} | {row['node_mean']:8.2f} | "
                  f"{row['deviation_from_global']:8.2f} | {row['target_variance']:8.2f} | "
                  f"{row['ss_tot_contrib_pct']:8.2f}% | {row['nse_node']:9.4f}")
        
        # Cumulative contribution analysis
        cumsum_pct = contrib_df['ss_tot_contrib_pct'].cumsum()
        top_10_pct = cumsum_pct.iloc[9] if len(cumsum_pct) > 9 else cumsum_pct.iloc[-1]
        top_100_pct = cumsum_pct.iloc[99] if len(cumsum_pct) > 99 else cumsum_pct.iloc[-1]
        
        print(f"\n4. CUMULATIVE CONTRIBUTION ANALYSIS:")
        print(f"   Top 10 nodes account for: {top_10_pct:.2f}% of SS_tot")
        print(f"   Top 100 nodes account for: {top_100_pct:.2f}% of SS_tot")
        
        # Variance analysis
        print(f"\n5. VARIANCE ANALYSIS (Why aggregated NSE is high):")
        
        # Between-node variance vs within-node variance
        between_node_var = np.var(contrib_df['node_mean'].values)
        avg_within_node_var = contrib_df['target_variance'].mean()
        
        print(f"   Between-node variance (variance of node means): {between_node_var:.2f}")
        print(f"   Average within-node variance: {avg_within_node_var:.2f}")
        print(f"   Ratio (between/within): {between_node_var / avg_within_node_var if avg_within_node_var > 0 else np.inf:.2f}x")
        
        if between_node_var > 10 * avg_within_node_var:
            print(f"\n   ⚠️  Between-node variance is {between_node_var / avg_within_node_var:.1f}x larger!")
            print(f"   This means most variance comes from DIFFERENCES BETWEEN NODES,")
            print(f"   not from TEMPORAL CHANGES WITHIN NODES.")
            print(f"   Your model learned per-node constants, which captures between-node")
            print(f"   variance but completely misses temporal dynamics!")
        
        # Prediction variance analysis
        print(f"\n6. PREDICTION VARIANCE (Is model predicting constants?):")
        
        low_pred_var_nodes = contrib_df[contrib_df['pred_std'] < 0.1 * contrib_df['target_std']]
        print(f"   Nodes with pred_std < 10% of target_std: {len(low_pred_var_nodes)} "
              f"({len(low_pred_var_nodes)/len(contrib_df)*100:.1f}%)")
        
        if len(low_pred_var_nodes) > 0.5 * len(contrib_df):
            print(f"   ⚠️  Over 50% of nodes have nearly constant predictions!")
            print(f"   This confirms the model is NOT learning temporal patterns.")
        
        # Show worst performing nodes
        print(f"\n7. WORST PERFORMING NODES (lowest per-node NSE):")
        print(f"\n   {'Node':>6} | {'Mean':>8} | {'Target σ':>9} | {'Pred σ':>9} | {'NSE_node':>9}")
        print("   " + "-" * 58)
        
        worst_nodes = contrib_df.nsmallest(10, 'nse_node')
        for idx, row in worst_nodes.iterrows():
            print(f"   {row['node_id']:6.0f} | {row['node_mean']:8.2f} | "
                  f"{row['target_std']:9.4f} | {row['pred_std']:9.4f} | {row['nse_node']:9.4f}")
        
        print(f"\n{'=' * 80}\n")
    
    print("\nKEY INSIGHTS:")
    print("=" * 80)
    print("1. Aggregated NSE uses GLOBAL MEAN across all nodes")
    print("   → Nodes far from global mean contribute huge SS_tot values")
    print("   → These dominate the denominator, making NSE artificially high")
    print("\n2. Per-node NSE uses EACH NODE'S OWN MEAN")
    print("   → Only measures temporal prediction within that node")
    print("   → Reveals the model isn't learning time dynamics")
    print("\n3. If between-node variance >> within-node variance:")
    print("   → Aggregated metrics will look good even with constant predictions")
    print("   → Always use PER-NODE metrics for time series evaluation!")
    print("=" * 80)

if __name__ == "__main__":
    csv_file = "Model2_trained/gru_test_predictions_test.csv"

    print("Analyzing node-level predictions...")
    print("\n" + "=" * 80)
    print("NODE STATISTICS (Aggregated vs Per-Node)")
    print("=" * 80)
    analyze_node_statistics(csv_path=csv_file)

    print("\n" + "=" * 80)
    print("PER-NODE METRIC DISTRIBUTIONS")
    print("=" * 80)
    plot_per_node_metric_distributions(csv_path=csv_file)

    print("\n" + "=" * 80)
    print("PLOTTING INDIVIDUAL NODE TIME SERIES (10 random nodes)")
    print("=" * 80)
    plot_individual_node_timeseries(csv_path=csv_file, num_nodes=10)

    print("\n" + "=" * 80)
    print("EXAMPLE: Plot a specific node")
    print("=" * 80)
    print("To plot a specific node, use:")
    print("  plot_specific_node('gru_test_predictions_test.csv', node_id=42)")

    explain_nse_calculation(csv_path=csv_file, show_top_contributors=20)

    print("\n✓ All visualizations complete!")
