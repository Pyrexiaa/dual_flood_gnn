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


def calculate_event_nse(event_df, y_pred_col='water_level_pred'):
    """
    Calculate hierarchical NSE for a single event
    
    Args:
        event_df: DataFrame for one event with columns: node_type, node_id, water_level (truth), water_level_pred
        y_pred_col: Name of prediction column
    
    Returns:
        float: Event NSE score
    """
    node_types = event_df['node_type'].values
    node_ids = event_df['node_id'].values
    y_true = event_df['water_level'].values
    y_pred = event_df[y_pred_col].values
    
    # Find unique 1D and 2D nodes
    mask_1d = node_types == 1
    mask_2d = node_types == 2
    
    unique_1d_nodes = np.unique(node_ids[mask_1d]) if mask_1d.any() else np.array([])
    unique_2d_nodes = np.unique(node_ids[mask_2d]) if mask_2d.any() else np.array([])
    
    # Calculate NSE for 1D nodes (predict 0D nodes)
    nse_1d_list = []
    for node_id in unique_1d_nodes:
        mask = (node_ids == node_id) & (node_types == 0)
        if np.sum(mask) > 1:
            node_nse = nse(y_true[mask], y_pred[mask])
            if not np.isnan(node_nse):
                nse_1d_list.append(node_nse)
    
    # Calculate NSE for 2D nodes (predict 1D nodes)
    nse_2d_list = []
    for node_id in unique_2d_nodes:
        mask = (node_ids == node_id) & (node_types == 1)
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
    merged = solution_df.copy()
    merged['water_level_pred'] = prediction_df['water_level'].values
    
    # Calculate NSE per event, then average per model
    model_event_nses = defaultdict(list)
    
    for (model_id, event_id), event_df in merged.groupby(['model_id', 'event_id']):
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


def generate_comprehensive_baselines(
    solution_path,
    output_dir="baselines",
    random_seed=42
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
    
    Returns:
        dict: Dictionary with baseline names and their NSE scores
    """
    np.random.seed(random_seed)
    
    # Load solution
    solution_path = Path(solution_path)
    print(f"Loading solution from: {solution_path}")
    
    if solution_path.suffix.lower() == '.parquet':
        solution = pd.read_parquet(solution_path)
    else:
        solution = pd.read_csv(solution_path)
    
    print(f"Loaded {len(solution):,} rows")
    print(f"Columns: {list(solution.columns)}")
    
    # Create output directory
    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True)
    
    # Required columns for submission
    required_cols = ['model_id', 'event_id', 'node_type', 'node_id', 'water_level']
    
    # Check if solution has row_id
    has_row_id = 'row_id' in solution.columns
    if has_row_id:
        submission_cols = ['row_id'] + required_cols
    else:
        submission_cols = required_cols
    
    results = {}
    
    print("\n" + "="*80)
    print("GENERATING COMPREHENSIVE BASELINES")
    print("="*80)
    
    # ============================================================================
    # NAIVE BASELINES
    # ============================================================================
    print("\n" + "="*80)
    print("NAIVE BASELINES")
    print("="*80)
    
    # 1. All Zeros
    print("\n[1/12] All Zeros")
    baseline = solution[submission_cols].copy()
    baseline['water_level'] = 0.0
    try:
        score = calculate_nse_score(solution, baseline)
        results['naive_all_zeros'] = score
        print(f"  NSE Score: {score:.6f}")
    except Exception as e:
        print(f"  Error: {e}")
        results['naive_all_zeros'] = None
    
    # 2. Random Per-Event
    print("\n[2/12] Random Per-Event (uniform within event min-max)")
    baseline = solution[submission_cols].copy()
    def random_per_event(group):
        min_val = group['water_level'].min()
        max_val = group['water_level'].max()
        if min_val == max_val:
            max_val = min_val + 0.01
        group['water_level'] = np.random.uniform(min_val, max_val, size=len(group))
        return group
    baseline = baseline.groupby(['model_id', 'event_id'], group_keys=False).apply(random_per_event)
    try:
        score = calculate_nse_score(solution, baseline)
        results['naive_random_per_event'] = score
        print(f"  NSE Score: {score:.6f}")
    except Exception as e:
        print(f"  Error: {e}")
        results['naive_random_per_event'] = None
    
    # 3. Mean Per-Event
    print("\n[3/12] Mean Per-Event")
    baseline = solution[submission_cols].copy()
    def mean_per_event(group):
        mean_val = group['water_level'].mean()
        group['water_level'] = mean_val
        return group
    baseline = baseline.groupby(['model_id', 'event_id'], group_keys=False).apply(mean_per_event)
    try:
        score = calculate_nse_score(solution, baseline)
        results['naive_mean_per_event'] = score
        print(f"  NSE Score: {score:.6f}")
    except Exception as e:
        print(f"  Error: {e}")
        results['naive_mean_per_event'] = None
    
    # ============================================================================
    # LOCATION-AWARE BASELINES
    # ============================================================================
    print("\n" + "="*80)
    print("LOCATION-AWARE BASELINES")
    print("="*80)
    
    # 4. Random Per-Node
    print("\n[4/12] Random Per-Node (uniform within node min-max)")
    baseline = solution[submission_cols].copy()
    def random_per_node(group):
        min_val = group['water_level'].min()
        max_val = group['water_level'].max()
        if min_val == max_val:
            max_val = min_val + 0.01
        group['water_level'] = np.random.uniform(min_val, max_val, size=len(group))
        return group
    baseline = baseline.groupby(['model_id', 'event_id', 'node_id'], group_keys=False).apply(random_per_node)
    try:
        score = calculate_nse_score(solution, baseline)
        results['location_random_per_node'] = score
        print(f"  NSE Score: {score:.6f}")
    except Exception as e:
        print(f"  Error: {e}")
        results['location_random_per_node'] = None
    
    # 5. Mean Per-Node
    print("\n[5/12] Mean Per-Node")
    baseline = solution[submission_cols].copy()
    def mean_per_node(group):
        mean_val = group['water_level'].mean()
        group['water_level'] = mean_val
        return group
    baseline = baseline.groupby(['model_id', 'event_id', 'node_id'], group_keys=False).apply(mean_per_node)
    try:
        score = calculate_nse_score(solution, baseline)
        results['location_mean_per_node'] = score
        print(f"  NSE Score: {score:.6f}")
    except Exception as e:
        print(f"  Error: {e}")
        results['location_mean_per_node'] = None
    
    # 6. Random Per-Node-Type
    print("\n[6/12] Random Per-Node-Type (separate ranges for 1D vs 2D)")
    baseline = solution[submission_cols].copy()
    def random_per_node_type(group):
        min_val = group['water_level'].min()
        max_val = group['water_level'].max()
        if min_val == max_val:
            max_val = min_val + 0.01
        group['water_level'] = np.random.uniform(min_val, max_val, size=len(group))
        return group
    baseline = baseline.groupby(['model_id', 'event_id', 'node_type'], group_keys=False).apply(random_per_node_type)
    try:
        score = calculate_nse_score(solution, baseline)
        results['location_random_per_node_type'] = score
        print(f"  NSE Score: {score:.6f}")
    except Exception as e:
        print(f"  Error: {e}")
        results['location_random_per_node_type'] = None
    
    # 7. Mean Per-Node-Type
    print("\n[7/12] Mean Per-Node-Type")
    baseline = solution[submission_cols].copy()
    def mean_per_node_type(group):
        mean_val = group['water_level'].mean()
        group['water_level'] = mean_val
        return group
    baseline = baseline.groupby(['model_id', 'event_id', 'node_type'], group_keys=False).apply(mean_per_node_type)
    try:
        score = calculate_nse_score(solution, baseline)
        results['location_mean_per_node_type'] = score
        print(f"  NSE Score: {score:.6f}")
    except Exception as e:
        print(f"  Error: {e}")
        results['location_mean_per_node_type'] = None
    
    # ============================================================================
    # TEMPORAL BASELINES
    # ============================================================================
    print("\n" + "="*80)
    print("TEMPORAL BASELINES")
    print("="*80)
    
    # 8. Persistence (Timestep 10)
    print("\n[8/12] Persistence (repeat timestep 10 value)")
    baseline = solution[submission_cols].copy()
    def persistence_t10(group):
        if len(group) <= 10:
            return group
        timestep_10_value = group.iloc[10]['water_level']
        group.loc[group.index[11:], 'water_level'] = timestep_10_value
        return group
    baseline = baseline.groupby(['model_id', 'event_id', 'node_id'], group_keys=False).apply(persistence_t10)
    try:
        score = calculate_nse_score(solution, baseline)
        results['temporal_persistence_t10'] = score
        print(f"  NSE Score: {score:.6f}")
    except Exception as e:
        print(f"  Error: {e}")
        results['temporal_persistence_t10'] = None
    
    # 9. Last Known Value (Sequential)
    print("\n[9/12] Last Known Value (each timestep = previous timestep)")
    baseline = solution[submission_cols].copy()
    def last_value(group):
        if len(group) > 1:
            # Shift values forward by 1 (each timestep uses previous value)
            group['water_level'] = group['water_level'].shift(1)
            # First timestep keeps its original value
            group.iloc[0, group.columns.get_loc('water_level')] = group.iloc[0]['water_level']
        return group
    baseline = baseline.groupby(['model_id', 'event_id', 'node_id'], group_keys=False).apply(last_value)
    try:
        score = calculate_nse_score(solution, baseline)
        results['temporal_last_value'] = score
        print(f"  NSE Score: {score:.6f}")
    except Exception as e:
        print(f"  Error: {e}")
        results['temporal_last_value'] = None
    
    # 10. Linear Trend (extrapolate from timesteps 0-10)
    print("\n[10/12] Linear Trend (fit line on timesteps 0-10, extrapolate)")
    baseline = solution[submission_cols].copy()
    def linear_trend(group):
        if len(group) <= 10:
            return group
        # Fit linear trend on first 11 timesteps (0-10)
        x = np.arange(11)
        y = group.iloc[:11]['water_level'].values
        # Simple linear regression
        coeffs = np.polyfit(x, y, 1)
        slope, intercept = coeffs[0], coeffs[1]
        # Extrapolate for all timesteps
        x_all = np.arange(len(group))
        group['water_level'] = slope * x_all + intercept
        return group
    baseline = baseline.groupby(['model_id', 'event_id', 'node_id'], group_keys=False).apply(linear_trend)
    try:
        score = calculate_nse_score(solution, baseline)
        results['temporal_linear_trend'] = score
        print(f"  NSE Score: {score:.6f}")
    except Exception as e:
        print(f"  Error: {e}")
        results['temporal_linear_trend'] = None
    
    # ============================================================================
    # HIERARCHICAL BASELINES
    # ============================================================================
    print("\n" + "="*80)
    print("HIERARCHICAL BASELINES")
    print("="*80)
    
    # 11. Mean Per-Model
    print("\n[11/12] Mean Per-Model")
    baseline = solution[submission_cols].copy()
    def mean_per_model(group):
        mean_val = group['water_level'].mean()
        group['water_level'] = mean_val
        return group
    baseline = baseline.groupby(['model_id'], group_keys=False).apply(mean_per_model)
    try:
        score = calculate_nse_score(solution, baseline)
        results['hierarchical_mean_per_model'] = score
        print(f"  NSE Score: {score:.6f}")
    except Exception as e:
        print(f"  Error: {e}")
        results['hierarchical_mean_per_model'] = None
    
    # 12. Global Mean
    print("\n[12/12] Global Mean (single mean for all predictions)")
    baseline = solution[submission_cols].copy()
    global_mean = solution['water_level'].mean()
    baseline['water_level'] = global_mean
    try:
        score = calculate_nse_score(solution, baseline)
        results['hierarchical_global_mean'] = score
        print(f"  NSE Score: {score:.6f}")
    except Exception as e:
        print(f"  Error: {e}")
        results['hierarchical_global_mean'] = None
    
    # ============================================================================
    # SUMMARY & SAVE RESULTS
    # ============================================================================
    print("\n" + "="*80)
    print("BASELINE RESULTS SUMMARY")
    print("="*80)
    
    # Organize results by category
    categories = {
        'Naive': ['naive_all_zeros', 'naive_random_per_event', 'naive_mean_per_event'],
        'Location-Aware': ['location_random_per_node', 'location_mean_per_node', 
                          'location_random_per_node_type', 'location_mean_per_node_type'],
        'Temporal': ['temporal_persistence_t10', 'temporal_last_value', 'temporal_linear_trend'],
        'Hierarchical': ['hierarchical_mean_per_model', 'hierarchical_global_mean']
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
    results_df = pd.DataFrame([
        {'baseline_method': name, 'nse_score': score, 
         'category': next((cat for cat, names in categories.items() if name in names), 'Other')}
        for name, score in results.items()
    ])
    results_df = results_df.sort_values('nse_score', ascending=False)
    
    results_path = output_dir / "comprehensive_baseline_results.csv"
    results_df.to_csv(results_path, index=False)
    
    print(f"\n{'='*80}")
    print(f"Results saved to: {results_path.absolute()}")
    
    # Find best baseline
    valid_results = [(name, score) for name, score in results.items() if score is not None]
    if valid_results:
        best_name, best_score = max(valid_results, key=lambda x: x[1])
        print(f"\nBest Baseline: {best_name}")
        print(f"Best NSE Score: {best_score:.6f}")
    
    print("="*80 + "\n")
    
    return results


if __name__ == "__main__":
    # Example usage
    solution_path = "/Users/jiayulim/Documents/GitHub/dual_flood_gnn/kaggle_submission/solutions.csv"
    
    # Generate comprehensive baselines and calculate their NSE scores
    baseline_results = generate_comprehensive_baselines(
        solution_path=solution_path,
        output_dir="baselines",
        random_seed=42
    )
    