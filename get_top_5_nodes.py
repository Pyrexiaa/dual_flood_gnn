import pandas as pd
import numpy as np

# -----------------------------
# Load CSV
# -----------------------------
input_csv = "kaggle_submissions/solutions.csv"
output_csv = "top5_oscillatory_nodes_same_event.csv"

df = pd.read_csv(input_csv)
df = df.sort_values(['model_id', 'event_id', 'node_type', 'node_id', 'row_id'])


# =============================
# Oscillation Detection Functions
# =============================
def count_direction_changes(series):
    """Count zigzag patterns"""
    if len(series) < 3:
        return 0
    diff = np.diff(series)
    signs = np.sign(diff)
    signs_nonzero = signs[signs != 0]
    if len(signs_nonzero) < 2:
        return 0
    return np.sum(np.diff(signs_nonzero) != 0)


def count_peaks_and_troughs(series, prominence=None):
    """Count local maxima and minima"""
    if len(series) < 3:
        return 0
    
    # Auto-calculate prominence if not provided
    if prominence is None:
        prominence = (np.max(series) - np.min(series)) * 0.05  # 5% of range
    
    peaks = 0
    troughs = 0
    
    for i in range(1, len(series) - 1):
        if series[i] > series[i-1] and series[i] > series[i+1]:
            if series[i] - min(series[i-1], series[i+1]) >= prominence:
                peaks += 1
        elif series[i] < series[i-1] and series[i] < series[i+1]:
            if min(series[i-1], series[i+1]) - series[i] >= prominence:
                troughs += 1
    
    return peaks + troughs


def mean_abs_second_derivative(series):
    """Measure average curvature"""
    if len(series) < 3:
        return 0
    first_deriv = np.diff(series)
    second_deriv = np.diff(first_deriv)
    return np.mean(np.abs(second_deriv))


def normalized_total_variation(series):
    """Total movement normalized by range"""
    if len(series) < 2:
        return 0
    tv = np.sum(np.abs(np.diff(series)))
    value_range = np.max(series) - np.min(series)
    return tv / value_range if value_range > 0 else 0


def composite_oscillation_score(series):
    """
    Prioritizes LARGE movements over high-frequency oscillations.
    Returns a score roughly in 0-100 range.
    """
    if len(series) < 3:
        return 0
    
    # Range-based metrics (amplitude)
    value_range = np.max(series) - np.min(series)
    std_dev = np.std(series)
    
    # Movement metrics
    norm_tv = normalized_total_variation(series)
    mean_abs_change = np.mean(np.abs(np.diff(series)))
    
    # Oscillation metrics (less weight)
    dir_changes = count_direction_changes(series) / max(len(series) - 2, 1)
    peaks = count_peaks_and_troughs(series, prominence=value_range*0.1) / max(len(series) - 2, 1)
    
    # Weighted combination - PRIORITIZE AMPLITUDE
    score = (
        0.40 * value_range +              # Total range of values (HIGH WEIGHT)
        0.25 * std_dev * 5 +               # Variability
        0.15 * mean_abs_change * 10 +     # Average step size
        0.10 * norm_tv * 10 +              # Normalized total variation
        0.05 * dir_changes * 100 +         # Direction changes (LOW WEIGHT)
        0.05 * peaks * 100                 # Peaks/troughs (LOW WEIGHT)
    )

    return score


# =============================
# Compute Metrics Per Event
# =============================
print("Computing oscillation metrics for each node in each event...")

node_event_metrics = []

for (model_id, event_id, node_type, node_id), group in df.groupby(
    ['model_id', 'event_id', 'node_type', 'node_id']
):
    water_levels = group.sort_values('row_id')['water_level'].values
    
    metrics = {
        'model_id': model_id,
        'event_id': event_id,
        'node_type': node_type,
        'node_id': node_id,
        'timesteps': len(water_levels),
        'oscillation_score': composite_oscillation_score(water_levels),
        'direction_changes': count_direction_changes(water_levels),
        'peaks_troughs': count_peaks_and_troughs(water_levels),
        'mean_curvature': mean_abs_second_derivative(water_levels),
        'normalized_tv': normalized_total_variation(water_levels),
        'std_dev': np.std(water_levels),
        'value_range': np.max(water_levels) - np.min(water_levels),
        'min_value': np.min(water_levels),
        'max_value': np.max(water_levels)
    }
    
    node_event_metrics.append(metrics)

metrics_df = pd.DataFrame(node_event_metrics)

# =============================
# Rank Within Each Event
# =============================
print("Ranking nodes within each event...")

metrics_df['rank'] = (
    metrics_df
    .groupby(['model_id', 'event_id', 'node_type'])['oscillation_score']
    .rank(method='dense', ascending=False)
)

# =============================
# Keep Top 5 Per Event
# =============================
top5_per_event = metrics_df[metrics_df['rank'] <= 5].copy()

# Sort for readability
top5_per_event = top5_per_event.sort_values(
    ['model_id', 'event_id', 'node_type', 'rank']
)

# =============================
# Find THE Event with Highest Overall Oscillation
# =============================
print("Finding events with highest overall oscillation...")

# Calculate average oscillation score per event
event_avg_oscillation = (
    top5_per_event
    .groupby(['model_id', 'event_id', 'node_type'])
    .agg({
        'oscillation_score': 'mean',
        'node_id': 'count'  # Should be 5 for top 5
    })
    .rename(columns={
        'oscillation_score': 'avg_top5_oscillation',
        'node_id': 'num_nodes'
    })
    .reset_index()
)

# For each model and node_type, find the event with highest average oscillation
best_events = (
    event_avg_oscillation
    .loc[event_avg_oscillation.groupby(['model_id', 'node_type'])['avg_top5_oscillation'].idxmax()]
)

print("\n" + "="*80)
print("BEST EVENT (highest average oscillation among top 5 nodes):")
print("="*80)
print(best_events.to_string(index=False))

# =============================
# Get Top 5 Nodes from Best Events
# =============================
best_event_top5 = top5_per_event.merge(
    best_events[['model_id', 'event_id', 'node_type']],
    on=['model_id', 'event_id', 'node_type'],
    how='inner'
)

# =============================
# Save Results
# =============================
# Save all top 5 per event
top5_per_event.to_csv('top5_oscillatory_nodes_all_events.csv', index=False)

# Save only the best event's top 5
best_event_top5.to_csv(output_csv, index=False)

print("\n" + "="*80)
print(f"Top 5 nodes from BEST events saved to: {output_csv}")
print(f"All top 5 nodes (all events) saved to: top5_oscillatory_nodes_all_events.csv")
print("="*80)

print("\nBest Event's Top 5 Nodes:")
print(best_event_top5[['model_id', 'event_id', 'node_type', 'node_id', 'rank', 
                        'oscillation_score', 'direction_changes', 'peaks_troughs']].to_string(index=False))

# =============================
# Summary Statistics
# =============================
print("\n" + "="*80)
print("SUMMARY BY MODEL AND NODE_TYPE:")
print("="*80)

summary = best_event_top5.groupby(['model_id', 'node_type', 'event_id']).agg({
    'node_id': lambda x: list(x),
    'oscillation_score': ['mean', 'min', 'max'],
    'peaks_troughs': 'mean',
    'direction_changes': 'mean'
}).round(2)

print(summary)