import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from typing import Union, List, Tuple
import warnings
warnings.filterwarnings('ignore')

# Set style
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (15, 10)


def plot_prediction_vs_truth_scatter(
    ground_truth: pd.DataFrame,
    prediction: pd.DataFrame,
    sample_size: int = 100000,
    save_path: str = None,
    title: str = "Predictions vs Ground Truth"
):
    """
    Create scatter plot with hexbin for density visualization
    
    Args:
        ground_truth: Ground truth DataFrame
        prediction: Prediction DataFrame
        sample_size: Number of points to sample (None for all data)
        save_path: Path to save figure
        title: Plot title
    """
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    
    # Align data
    y_true = ground_truth['water_level'].values
    y_pred = prediction['water_level'].values
    
    # Sample if needed
    if sample_size and len(y_true) > sample_size:
        indices = np.random.choice(len(y_true), sample_size, replace=False)
        y_true_plot = y_true[indices]
        y_pred_plot = y_pred[indices]
        sample_note = f" (sampled {sample_size:,} of {len(y_true):,} points)"
    else:
        y_true_plot = y_true
        y_pred_plot = y_pred
        sample_note = f" ({len(y_true):,} points)"
    
    # Plot 1: Hexbin (density heatmap)
    hb = axes[0].hexbin(y_true_plot, y_pred_plot, gridsize=50, cmap='YlOrRd', 
                        mincnt=1, edgecolors='none')
    axes[0].plot([y_true.min(), y_true.max()], 
                 [y_true.min(), y_true.max()], 
                 'k--', lw=2, label='Perfect Prediction')
    axes[0].set_xlabel('Ground Truth Water Level', fontsize=12)
    axes[0].set_ylabel('Predicted Water Level', fontsize=12)
    axes[0].set_title(f'Density Plot{sample_note}', fontsize=14)
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    plt.colorbar(hb, ax=axes[0], label='Count')
    
    # Plot 2: Residual plot
    residuals = y_pred_plot - y_true_plot
    axes[1].hexbin(y_true_plot, residuals, gridsize=50, cmap='RdBu_r', 
                   mincnt=1, edgecolors='none')
    axes[1].axhline(y=0, color='k', linestyle='--', lw=2, label='Zero Error')
    axes[1].set_xlabel('Ground Truth Water Level', fontsize=12)
    axes[1].set_ylabel('Residual (Predicted - Truth)', fontsize=12)
    axes[1].set_title('Residual Plot', fontsize=14)
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    plt.colorbar(hb, ax=axes[1], label='Count')
    
    plt.suptitle(title, fontsize=16, y=1.02)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved scatter plot to: {save_path}")
    
    plt.close()
    
    # Print statistics
    rmse = np.sqrt(np.mean((y_pred - y_true) ** 2))
    mae = np.mean(np.abs(y_pred - y_true))
    r2 = 1 - np.sum((y_true - y_pred) ** 2) / np.sum((y_true - np.mean(y_true)) ** 2)
    
    print(f"\nOverall Statistics:")
    print(f"  RMSE: {rmse:.4f}")
    print(f"  MAE:  {mae:.4f}")
    print(f"  R²:   {r2:.4f}")


def plot_time_series_sample(
    ground_truth: pd.DataFrame,
    prediction: pd.DataFrame,
    n_events: int = 6,
    nodes_per_event: int = 3,
    save_path: str = None
):
    """
    Plot time series for random sample of events and nodes
    
    Args:
        ground_truth: Ground truth DataFrame
        prediction: Prediction DataFrame
        n_events: Number of events to plot
        nodes_per_event: Number of nodes to show per event
        save_path: Path to save figure
    """
    # Merge data
    key_cols = ['model_id', 'event_id', 'node_type', 'node_id']
    merged = ground_truth[key_cols + ['water_level']].copy()
    merged = merged.sort_values(key_cols).reset_index(drop=True)
    
    pred_sorted = prediction.sort_values(key_cols).reset_index(drop=True)
    merged['water_level_pred'] = pred_sorted['water_level'].values
    
    # Sample random events
    unique_events = merged.groupby(['model_id', 'event_id']).size().reset_index()[['model_id', 'event_id']]
    sampled_events = unique_events.sample(n=min(n_events, len(unique_events)), random_state=42)
    
    fig, axes = plt.subplots(n_events, 1, figsize=(16, 4 * n_events))
    if n_events == 1:
        axes = [axes]
    
    for idx, (ax, (_, event_row)) in enumerate(zip(axes, sampled_events.iterrows())):
        model_id = event_row['model_id']
        event_id = event_row['event_id']
        
        # Get event data
        event_data = merged[
            (merged['model_id'] == model_id) & 
            (merged['event_id'] == event_id)
        ].copy()
        
        # Sample nodes if too many
        unique_nodes = event_data.groupby(['node_type', 'node_id']).size().reset_index()[['node_type', 'node_id']]
        if len(unique_nodes) > nodes_per_event:
            sampled_nodes = unique_nodes.sample(n=nodes_per_event, random_state=42)
        else:
            sampled_nodes = unique_nodes
        
        # Plot each sampled node
        colors = plt.cm.tab10(np.linspace(0, 1, len(sampled_nodes)))
        
        for color, (_, node_row) in zip(colors, sampled_nodes.iterrows()):
            node_type = node_row['node_type']
            node_id = node_row['node_id']
            
            node_data = event_data[
                (event_data['node_type'] == node_type) & 
                (event_data['node_id'] == node_id)
            ].reset_index(drop=True)
            
            if len(node_data) > 0:
                x = np.arange(len(node_data))
                label = f"Node {node_id} (Type {node_type})"
                
                # Plot ground truth
                ax.plot(x, node_data['water_level'], 
                       color=color, linewidth=2, label=f"{label} - Truth", alpha=0.7)
                
                # Plot prediction
                ax.plot(x, node_data['water_level_pred'], 
                       color=color, linewidth=2, linestyle='--', 
                       label=f"{label} - Pred", alpha=0.7)
        
        ax.set_xlabel('Time Step', fontsize=11)
        ax.set_ylabel('Water Level', fontsize=11)
        ax.set_title(f'Model {model_id}, Event {event_id}', fontsize=12, fontweight='bold')
        ax.legend(loc='best', fontsize=9, ncol=2)
        ax.grid(True, alpha=0.3)
    
    plt.suptitle('Sample Time Series: Ground Truth vs Predictions', 
                 fontsize=16, y=1.00)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved time series plot to: {save_path}")
    
    plt.close()


def plot_performance_by_category(
    ground_truth: pd.DataFrame,
    prediction: pd.DataFrame,
    save_path: str = None
):
    """
    Plot performance metrics broken down by model, node type, and event
    
    Args:
        ground_truth: Ground truth DataFrame
        prediction: Prediction DataFrame
        save_path: Path to save figure
    """
    # Merge data
    key_cols = ['model_id', 'event_id', 'node_type', 'node_id']
    merged = ground_truth[key_cols + ['water_level']].copy()
    merged = merged.sort_values(key_cols).reset_index(drop=True)
    
    pred_sorted = prediction.sort_values(key_cols).reset_index(drop=True)
    merged['water_level_pred'] = pred_sorted['water_level'].values
    merged['residual'] = merged['water_level_pred'] - merged['water_level']
    merged['abs_error'] = np.abs(merged['residual'])
    
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    
    # 1. Error distribution by model
    for model_id in sorted(merged['model_id'].unique()):
        model_data = merged[merged['model_id'] == model_id]['abs_error']
        axes[0, 0].hist(model_data, bins=50, alpha=0.6, label=f'Model {model_id}')
    axes[0, 0].set_xlabel('Absolute Error', fontsize=11)
    axes[0, 0].set_ylabel('Frequency', fontsize=11)
    axes[0, 0].set_title('Error Distribution by Model', fontsize=12, fontweight='bold')
    axes[0, 0].legend()
    axes[0, 0].set_yscale('log')
    axes[0, 0].grid(True, alpha=0.3)
    
    # 2. Error distribution by node type
    for node_type in sorted(merged['node_type'].unique()):
        node_data = merged[merged['node_type'] == node_type]['abs_error']
        axes[0, 1].hist(node_data, bins=50, alpha=0.6, label=f'Type {node_type}')
    axes[0, 1].set_xlabel('Absolute Error', fontsize=11)
    axes[0, 1].set_ylabel('Frequency', fontsize=11)
    axes[0, 1].set_title('Error Distribution by Node Type', fontsize=12, fontweight='bold')
    axes[0, 1].legend()
    axes[0, 1].set_yscale('log')
    axes[0, 1].grid(True, alpha=0.3)
    
    # 3. MAE by model and node type
    mae_by_category = merged.groupby(['model_id', 'node_type'])['abs_error'].mean().reset_index()
    mae_pivot = mae_by_category.pivot(index='model_id', columns='node_type', values='abs_error')
    x = np.arange(len(mae_pivot))
    width = 0.35
    
    for i, node_type in enumerate(mae_pivot.columns):
        axes[0, 2].bar(x + i * width, mae_pivot[node_type], width, 
                      label=f'Type {node_type}', alpha=0.8)
    
    axes[0, 2].set_xlabel('Model ID', fontsize=11)
    axes[0, 2].set_ylabel('Mean Absolute Error', fontsize=11)
    axes[0, 2].set_title('MAE by Model and Node Type', fontsize=12, fontweight='bold')
    axes[0, 2].set_xticks(x + width / 2)
    axes[0, 2].set_xticklabels(mae_pivot.index)
    axes[0, 2].legend()
    axes[0, 2].grid(True, alpha=0.3, axis='y')
    
    # 4. Error by water level magnitude
    merged['water_level_bin'] = pd.cut(merged['water_level'], bins=20)
    error_by_magnitude = merged.groupby('water_level_bin')['abs_error'].agg(['mean', 'std']).reset_index()
    bin_centers = error_by_magnitude['water_level_bin'].apply(lambda x: x.mid)
    
    axes[1, 0].errorbar(bin_centers, error_by_magnitude['mean'], 
                       yerr=error_by_magnitude['std'], 
                       fmt='o-', capsize=5, capthick=2, alpha=0.7)
    axes[1, 0].set_xlabel('Ground Truth Water Level', fontsize=11)
    axes[1, 0].set_ylabel('Mean Absolute Error', fontsize=11)
    axes[1, 0].set_title('Error vs Water Level Magnitude', fontsize=12, fontweight='bold')
    axes[1, 0].grid(True, alpha=0.3)
    
    # 5. Residual distribution
    axes[1, 1].hist(merged['residual'], bins=100, alpha=0.7, edgecolor='black')
    axes[1, 1].axvline(x=0, color='red', linestyle='--', linewidth=2, label='Zero Error')
    axes[1, 1].set_xlabel('Residual (Pred - Truth)', fontsize=11)
    axes[1, 1].set_ylabel('Frequency', fontsize=11)
    axes[1, 1].set_title('Overall Residual Distribution', fontsize=12, fontweight='bold')
    axes[1, 1].legend()
    axes[1, 1].set_yscale('log')
    axes[1, 1].grid(True, alpha=0.3)
    
    # 6. Q-Q plot for residuals
    from scipy import stats
    sample_residuals = merged['residual'].sample(n=min(10000, len(merged)), random_state=42)
    stats.probplot(sample_residuals, dist="norm", plot=axes[1, 2])
    axes[1, 2].set_title('Q-Q Plot (Residual Normality)', fontsize=12, fontweight='bold')
    axes[1, 2].grid(True, alpha=0.3)
    
    plt.suptitle('Performance Analysis by Category', fontsize=16, y=1.00)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved performance analysis to: {save_path}")
    
    plt.close()


def plot_event_performance_distribution(
    ground_truth: pd.DataFrame,
    prediction: pd.DataFrame,
    save_path: str = None
):
    """
    Show distribution of performance across events (to identify problematic events)
    
    Args:
        ground_truth: Ground truth DataFrame
        prediction: Prediction DataFrame
        save_path: Path to save figure
    """
    # Merge data
    key_cols = ['model_id', 'event_id', 'node_type', 'node_id']
    merged = ground_truth[key_cols + ['water_level']].copy()
    merged = merged.sort_values(key_cols).reset_index(drop=True)
    
    pred_sorted = prediction.sort_values(key_cols).reset_index(drop=True)
    merged['water_level_pred'] = pred_sorted['water_level'].values
    
    # Calculate metrics per event
    event_metrics = []
    for (model_id, event_id), event_df in merged.groupby(['model_id', 'event_id']):
        y_true = event_df['water_level'].values
        y_pred = event_df['water_level_pred'].values
        
        rmse = np.sqrt(np.mean((y_true - y_pred) ** 2))
        mae = np.mean(np.abs(y_true - y_pred))
        
        # Calculate NSE if possible
        denominator = np.sum((y_true - np.mean(y_true)) ** 2)
        if denominator > 0:
            nse = 1 - np.sum((y_true - y_pred) ** 2) / denominator
        else:
            nse = np.nan
        
        event_metrics.append({
            'model_id': model_id,
            'event_id': event_id,
            'rmse': rmse,
            'mae': mae,
            'nse': nse,
            'n_points': len(event_df)
        })
    
    event_metrics_df = pd.DataFrame(event_metrics)
    
    fig, axes = plt.subplots(2, 2, figsize=(16, 10))
    
    # 1. RMSE distribution by model
    for model_id in sorted(event_metrics_df['model_id'].unique()):
        model_data = event_metrics_df[event_metrics_df['model_id'] == model_id]['rmse']
        axes[0, 0].hist(model_data, bins=30, alpha=0.6, label=f'Model {model_id}')
    axes[0, 0].set_xlabel('Event RMSE', fontsize=11)
    axes[0, 0].set_ylabel('Number of Events', fontsize=11)
    axes[0, 0].set_title('Event-Level RMSE Distribution', fontsize=12, fontweight='bold')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    
    # 2. NSE distribution by model
    for model_id in sorted(event_metrics_df['model_id'].unique()):
        model_data = event_metrics_df[event_metrics_df['model_id'] == model_id]['nse'].dropna()
        axes[0, 1].hist(model_data, bins=30, alpha=0.6, label=f'Model {model_id}')
    axes[0, 1].set_xlabel('Event NSE', fontsize=11)
    axes[0, 1].set_ylabel('Number of Events', fontsize=11)
    axes[0, 1].set_title('Event-Level NSE Distribution', fontsize=12, fontweight='bold')
    axes[0, 1].axvline(x=0, color='red', linestyle='--', linewidth=2, alpha=0.7)
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)
    
    # 3. Worst performing events
    worst_events = event_metrics_df.nlargest(20, 'rmse')
    colors = ['red' if mid == worst_events.iloc[0]['model_id'] else 'blue' 
              for mid in worst_events['model_id']]
    
    axes[1, 0].barh(range(len(worst_events)), worst_events['rmse'], color=colors, alpha=0.7)
    axes[1, 0].set_yticks(range(len(worst_events)))
    axes[1, 0].set_yticklabels([f"M{row['model_id']}-E{row['event_id']}" 
                                 for _, row in worst_events.iterrows()], fontsize=9)
    axes[1, 0].set_xlabel('RMSE', fontsize=11)
    axes[1, 0].set_title('Top 20 Worst Events (by RMSE)', fontsize=12, fontweight='bold')
    axes[1, 0].grid(True, alpha=0.3, axis='x')
    axes[1, 0].invert_yaxis()
    
    # 4. Best performing events
    best_events = event_metrics_df[event_metrics_df['nse'].notna()].nlargest(20, 'nse')
    colors = ['green' if mid == best_events.iloc[0]['model_id'] else 'orange' 
              for mid in best_events['model_id']]
    
    axes[1, 1].barh(range(len(best_events)), best_events['nse'], color=colors, alpha=0.7)
    axes[1, 1].set_yticks(range(len(best_events)))
    axes[1, 1].set_yticklabels([f"M{row['model_id']}-E{row['event_id']}" 
                                 for _, row in best_events.iterrows()], fontsize=9)
    axes[1, 1].set_xlabel('NSE', fontsize=11)
    axes[1, 1].set_title('Top 20 Best Events (by NSE)', fontsize=12, fontweight='bold')
    axes[1, 1].grid(True, alpha=0.3, axis='x')
    axes[1, 1].invert_yaxis()
    
    plt.suptitle('Event-Level Performance Analysis', fontsize=16, y=1.00)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved event performance plot to: {save_path}")
    
    plt.close()
    
    # Print summary statistics
    print("\nEvent-Level Performance Summary:")
    print(f"Total events: {len(event_metrics_df)}")
    print(f"\nRMSE Statistics:")
    print(event_metrics_df.groupby('model_id')['rmse'].describe())
    print(f"\nNSE Statistics:")
    print(event_metrics_df.groupby('model_id')['nse'].describe())


def generate_all_plots(
    ground_truth_path: Union[str, Path],
    prediction_path: Union[str, Path],
    output_dir: str = "plots",
    solution_name: str = "solution"
):
    """
    Generate all visualization plots for a solution
    
    Args:
        ground_truth_path: Path to ground truth file
        prediction_path: Path to prediction file
        output_dir: Directory to save plots
        solution_name: Name for this solution (used in filenames)
    """
    # Create output directory
    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True, parents=True)
    
    # Load data
    print(f"Loading ground truth from: {ground_truth_path}")
    if Path(ground_truth_path).suffix.lower() == '.parquet':
        ground_truth = pd.read_parquet(ground_truth_path)
    else:
        ground_truth = pd.read_csv(ground_truth_path)
    
    print(f"Loading prediction from: {prediction_path}")
    if Path(prediction_path).suffix.lower() == '.parquet':
        prediction = pd.read_parquet(prediction_path)
    else:
        prediction = pd.read_csv(prediction_path)
    
    print(f"Ground truth: {len(ground_truth):,} rows")
    print(f"Prediction: {len(prediction):,} rows\n")
    
    # Generate plots
    print("Generating scatter plots...")
    plot_prediction_vs_truth_scatter(
        ground_truth, prediction,
        sample_size=1000000,
        save_path=output_dir / f"{solution_name}_scatter.png",
        title=f"{solution_name}: Predictions vs Ground Truth"
    )
    
    print("\nGenerating time series samples...")
    plot_time_series_sample(
        ground_truth, prediction,
        n_events=6,
        nodes_per_event=3,
        save_path=output_dir / f"{solution_name}_timeseries.png"
    )
    
    print("\nGenerating performance analysis...")
    plot_performance_by_category(
        ground_truth, prediction,
        save_path=output_dir / f"{solution_name}_performance.png"
    )
    
    print("\nGenerating event performance distribution...")
    plot_event_performance_distribution(
        ground_truth, prediction,
        save_path=output_dir / f"{solution_name}_event_performance.png"
    )
    
    print(f"\n✓ All plots saved to: {output_dir.absolute()}")

def plot_submission_comparison(
    ground_truth_path: str,
    submission1_path: str,
    submission2_path: str,
    submission1_name: str = "Submission 1",
    submission2_name: str = "Submission 2",
    node_id: int = 16,
    node_type: int = 1,
    n_events: int = 2,
    save_path: str = None
):
    """
    Plot comparison of two submissions against ground truth for specific node
    
    Args:
        ground_truth_path: Path to ground truth file
        submission1_path: Path to first submission file
        submission2_path: Path to second submission file
        submission1_name: Display name for first submission
        submission2_name: Display name for second submission
        node_id: Node ID to plot (default: 16)
        node_type: Node type to plot (default: 1 for 1D nodes)
        n_events: Number of random events to plot (default: 2)
        save_path: Path to save figure
    """
    # Load data
    if Path(ground_truth_path).suffix.lower() == '.parquet':
        ground_truth = pd.read_parquet(ground_truth_path)
    else:
        ground_truth = pd.read_csv(ground_truth_path)
    
    if Path(submission1_path).suffix.lower() == '.parquet':
        submission1 = pd.read_parquet(submission1_path)
    else:
        submission1 = pd.read_csv(submission1_path)
    
    if Path(submission2_path).suffix.lower() == '.parquet':
        submission2 = pd.read_parquet(submission2_path)
    else:
        submission2 = pd.read_csv(submission2_path)
    
    # Filter for specific node
    key_cols = ['model_id', 'event_id', 'node_type', 'node_id']
    
    gt_filtered = ground_truth[
        (ground_truth['node_id'] == node_id) & 
        (ground_truth['node_type'] == node_type)
    ].copy()
    
    sub1_filtered = submission1[
        (submission1['node_id'] == node_id) & 
        (submission1['node_type'] == node_type)
    ].copy()
    
    sub2_filtered = submission2[
        (submission2['node_id'] == node_id) & 
        (submission2['node_type'] == node_type)
    ].copy()
    
    # Sample random events
    unique_events = gt_filtered.groupby(['model_id', 'event_id']).size().reset_index()[['model_id', 'event_id']]
    
    if len(unique_events) < n_events:
        print(f"Warning: Only {len(unique_events)} events available for node {node_id} (type {node_type})")
        n_events = len(unique_events)
    
    sampled_events = unique_events.sample(n=n_events, random_state=42)
    
    # Create subplots
    fig, axes = plt.subplots(n_events, 1, figsize=(14, 5 * n_events))
    if n_events == 1:
        axes = [axes]
    
    for idx, (ax, (_, event_row)) in enumerate(zip(axes, sampled_events.iterrows())):
        model_id = event_row['model_id']
        event_id = event_row['event_id']
        
        # Get event data for this specific node
        gt_event = gt_filtered[
            (gt_filtered['model_id'] == model_id) & 
            (gt_filtered['event_id'] == event_id)
        ].sort_values(key_cols).reset_index(drop=True)
        
        sub1_event = sub1_filtered[
            (sub1_filtered['model_id'] == model_id) & 
            (sub1_filtered['event_id'] == event_id)
        ].sort_values(key_cols).reset_index(drop=True)
        
        sub2_event = sub2_filtered[
            (sub2_filtered['model_id'] == model_id) & 
            (sub2_filtered['event_id'] == event_id)
        ].sort_values(key_cols).reset_index(drop=True)
        
        if len(gt_event) > 0:
            x = np.arange(len(gt_event))
            
            # Plot ground truth
            ax.plot(x, gt_event['water_level'], 
                   color='black', linewidth=2.5, label='Ground Truth', 
                   alpha=0.8, zorder=3)
            
            # Plot submission 1
            if len(sub1_event) > 0:
                ax.plot(x, sub1_event['water_level'], 
                       color='#1f77b4', linewidth=2, linestyle='--', 
                       label=submission1_name, alpha=0.7, zorder=2)
            
            # Plot submission 2
            if len(sub2_event) > 0:
                ax.plot(x, sub2_event['water_level'], 
                       color='#ff7f0e', linewidth=2, linestyle='-.', 
                       label=submission2_name, alpha=0.7, zorder=1)
            
            # Calculate metrics for this event
            if len(sub1_event) > 0 and len(sub2_event) > 0:
                rmse1 = np.sqrt(np.mean((gt_event['water_level'].values - sub1_event['water_level'].values) ** 2))
                rmse2 = np.sqrt(np.mean((gt_event['water_level'].values - sub2_event['water_level'].values) ** 2))
                
                # Add metrics to title
                ax.set_title(
                    f'Model {model_id}, Event {event_id}, Node {node_id} (Type {node_type})\n'
                    f'{submission1_name} RMSE: {rmse1:.4f} | {submission2_name} RMSE: {rmse2:.4f}',
                    fontsize=12, fontweight='bold'
                )
            else:
                ax.set_title(
                    f'Model {model_id}, Event {event_id}, Node {node_id} (Type {node_type})',
                    fontsize=12, fontweight='bold'
                )
            
            ax.set_xlabel('Time Step', fontsize=11)
            ax.set_ylabel('Water Level', fontsize=11)
            ax.legend(loc='best', fontsize=10)
            ax.grid(True, alpha=0.3)
        else:
            ax.text(0.5, 0.5, f'No data for Model {model_id}, Event {event_id}',
                   ha='center', va='center', transform=ax.transAxes)
            ax.set_title(f'Model {model_id}, Event {event_id}', fontsize=12)
    
    plt.suptitle(f'Submission Comparison: Node {node_id} (Type {node_type})', 
                 fontsize=16, fontweight='bold', y=1.00)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved comparison plot to: {save_path}")
    else:
        plt.show()
    
    plt.close()


# Example usage:
if __name__ == "__main__":
    # Single solution

    # submission_dict = {
    #     "gegerout": {
    #         "Gegerout_Version_30": "submissions/gegerout/49646601_submission.csv",
    #         "Gegerout_-4.5196_Model": "submissions/gegerout/49642289_submission.csv"
    #     },
    #     "kevincao": {
    #         "KevinCao_-6.5594_Model": "submissions/kevincao/49688714_sub2.parquet",
    #         "KevinCao_-4.5196_Model": "submissions/kevincao/49605808_sub1.parquet"
    #     },
    #     "timotheehenry": {
    #         "TimotheeHenry_-4.7775_Model12": "submissions/timotheehenry/49684010_subm12.csv",
    #         "TimotheeHenry_-4.7629_Model11": "submissions/timotheehenry/49683468_subm11.csv",
    #         "TimotheeHenry_-4.5229_Model6": "submissions/timotheehenry/49645995_subm6.csv",
    #     },
    #     "thelaplacian": {
    #         "TheLaplacian_-80000000000_ModelA": "submissions/thelaplacian/49639878_submission.csv",
    #         "TheLaplacian_-4.5196_ModelB": "submissions/thelaplacian/49636087_submission.csv",
    #     },
    #     "mattmotoki": {
    #         "MattMotoki_-4.7775_Persistence": "submissions/mattmotoki/49696172_persistence.parquet",
    #         "MattMotoki_-170954133676167_ModelDecay": "submissions/mattmotoki/49695478_model_decay.parquet",
    #     }
    # }

    # for key, values in submission_dict.items():
    #     for solution_name, prediction_path in values.items():
    #         print(f"\n\n=== Generating plots for {key} - {solution_name} ===")
    #         generate_all_plots(
    #             ground_truth_path="kaggle_submission/solutions.csv",
    #             prediction_path=prediction_path,
    #             output_dir=f"submissions/{key}_plots",
    #             solution_name=solution_name
    #         )
    
    plot_submission_comparison(
        ground_truth_path="kaggle_submission/solutions.csv",
        submission1_path="submissions/gegerout/49646601_submission.csv",
        submission2_path="submissions/gegerout/49642289_submission.csv",
        submission1_name="RMSE-based prediction",
        submission2_name="NSE-based prediction",
        node_id=16,
        node_type=1,
        n_events=10,
        save_path="submissions/comparison_gegerout_nse_rmse.png"
    )