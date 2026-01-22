import pandas as pd
import numpy as np
import pandas.api.types
import gc
from collections import defaultdict
from pathlib import Path
import psutil
import os
import time


class ParticipantVisibleError(Exception):
    pass


def get_memory_usage_gb():
    """Get current memory usage in GB"""
    process = psutil.Process(os.getpid())
    return process.memory_info().rss / 1024 / 1024 / 1024


def log_memory(label="", threshold_gb=12.0):
    """Log memory usage and warn if approaching threshold"""
    mem_gb = get_memory_usage_gb()
    percentage = (mem_gb / threshold_gb) * 100
    status = "⚠️ WARNING" if mem_gb > threshold_gb * 0.9 else "✓"
    print(
        f"[{status}] {label}: {mem_gb:.2f} GB ({percentage:.1f}% of {threshold_gb}GB limit)"
    )
    return mem_gb


def nse(y_true, y_pred):
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)

    denominator = np.sum((y_true - np.mean(y_true)) ** 2)

    if denominator == 0:
        return np.nan

    return 1 - np.sum((y_true - y_pred) ** 2) / denominator


def calculate_event_nse_from_arrays(node_types, node_ids, y_true, y_pred):
    mask_1d = node_types == 1
    mask_2d = node_types == 2

    unique_1d_nodes = np.unique(node_ids[mask_1d]) if mask_1d.any() else np.array([])
    unique_2d_nodes = np.unique(node_ids[mask_2d]) if mask_2d.any() else np.array([])

    nse_1d_list = []
    for node_id in unique_1d_nodes:
        mask = (node_ids == node_id) & (node_types == 0)
        if np.sum(mask) > 1:
            node_nse = nse(y_true[mask], y_pred[mask])
            if not np.isnan(node_nse):
                nse_1d_list.append(node_nse)

    nse_2d_list = []
    for node_id in unique_2d_nodes:
        mask = (node_ids == node_id) & (node_types == 1)
        if np.sum(mask) > 1:
            node_nse = nse(y_true[mask], y_pred[mask])
            if not np.isnan(node_nse):
                nse_2d_list.append(node_nse)

    nse_1d_avg = np.mean(nse_1d_list) if nse_1d_list else np.nan
    nse_2d_avg = np.mean(nse_2d_list) if nse_2d_list else np.nan
    valid_nse = [x for x in [nse_1d_avg, nse_2d_avg] if not np.isnan(x)]

    return np.mean(valid_nse) if valid_nse else np.nan


def load_file(filepath, required_cols):
    """
    Load file in Parquet or CSV format, selecting only required columns
    """
    filepath = Path(filepath)

    if filepath.suffix.lower() == ".parquet":
        return pd.read_parquet(filepath, columns=required_cols, engine="pyarrow")
    elif filepath.suffix.lower() == ".csv":
        return pd.read_csv(filepath, usecols=required_cols)
    else:
        raise ValueError(f"Unsupported file format: {filepath.suffix}")


def score_from_files(
    solution_path: str, submission_path: str, row_id_column_name: str = "row_id"
) -> float:
    """
    Calculate NSE score from file paths (supports both Parquet and CSV)
    """
    log_memory("Initial memory")

    required_cols = ["model_id", "event_id", "node_type", "node_id", "water_level"]

    print(f"Loading solution from: {solution_path}")
    solution = load_file(solution_path, required_cols)
    print(f"  Loaded {len(solution):,} rows")
    log_memory("After loading solution")

    print(f"Loading submission from: {submission_path}")
    submission_cols = required_cols.copy()
    submission = load_file(submission_path, submission_cols)
    print(f"  Loaded {len(submission):,} rows")
    log_memory("After loading submission")

    score_result = score(solution, submission, row_id_column_name)
    log_memory("After scoring")

    return score_result


def score(
    solution: pd.DataFrame,
    submission: pd.DataFrame,
    row_id_column_name: str,
    log_event_memory: bool = True,
    memory_log_interval: int = 100,
) -> float:
    """
    Memory-efficient hierarchical NSE calculation optimized for speed

    Args:
        log_event_memory: Whether to log memory during event processing
        memory_log_interval: Log memory every N events (to reduce overhead)
    """
    required_cols = ["model_id", "event_id", "node_type", "node_id", "water_level"]
    key_cols = ["model_id", "event_id", "node_type", "node_id"]

    # Validation checks (same as before)
    missing_cols = [col for col in required_cols if col not in submission.columns]
    if missing_cols:
        raise ParticipantVisibleError(
            f"Submission missing required column(s): {', '.join(missing_cols)}"
        )

    cols_to_check = [col for col in submission.columns if col != row_id_column_name]
    extra_cols = [col for col in cols_to_check if col not in required_cols]
    if extra_cols:
        raise ParticipantVisibleError(
            f"Submission has unexpected column(s): {', '.join(extra_cols)}"
        )

    if len(submission) != len(solution):
        raise ParticipantVisibleError(
            f"Submission has {len(submission)} rows but expected {len(solution)} rows"
        )

    unique_models_sub = submission["model_id"].unique()
    if len(unique_models_sub) != 3:
        raise ParticipantVisibleError(
            f"Submission must have exactly 3 unique model_ids, but found {len(unique_models_sub)}"
        )

    unique_models_sol = solution["model_id"].unique()
    if set(unique_models_sub) != set(unique_models_sol):
        raise ParticipantVisibleError(
            "Submission model_ids do not match solution model_ids"
        )

    if not pandas.api.types.is_numeric_dtype(submission["water_level"]):
        raise ParticipantVisibleError(
            'Submission column "water_level" must be numeric'
        )

    if submission["water_level"].isna().any():
        raise ParticipantVisibleError(
            "Submission contains NaN value(s) in water_level column"
        )

    if not np.isfinite(submission["water_level"]).all():
        raise ParticipantVisibleError(
            "Submission contains infinite value(s) in water_level column"
        )

    valid_node_types = {0, 1, 2}
    submission_node_types = set(submission["node_type"].unique())
    solution_node_types = set(solution["node_type"].unique())

    invalid_node_types = submission_node_types - valid_node_types
    if invalid_node_types:
        raise ParticipantVisibleError(
            f"Submission has invalid node_type value(s): {sorted(list(invalid_node_types))}"
        )

    if submission_node_types != solution_node_types:
        raise ParticipantVisibleError(
            f"Submission node_types {sorted(list(submission_node_types))} do not match solution node_types {sorted(list(solution_node_types))}"
        )

    print("\n=== Starting Optimized Event Processing ===")
    start_time = time.time()

    # Select only needed columns
    sol_cols = key_cols + ["water_level"]
    solution_work = solution[sol_cols].copy()
    submission_work = submission[sol_cols].copy()

    del solution, submission
    gc.collect()

    # Sort by group keys for efficient iteration
    solution_work = solution_work.sort_values(key_cols)
    submission_work = submission_work.sort_values(key_cols)

    # Verify structure match
    for col in key_cols:
        if not solution_work[col].equals(submission_work[col]):
            raise ParticipantVisibleError(
                "Submission does not match solution structure"
            )

    # OPTIMIZATION: Use itertools.groupby instead of pandas groupby
    # This avoids creating group indices and is much faster

    model_event_nses = defaultdict(list)

    # Create group key tuples
    sol_keys = list(zip(solution_work["model_id"], solution_work["event_id"]))
    sub_keys = list(zip(submission_work["model_id"], submission_work["event_id"]))

    # Get indices for each group
    sol_indices = solution_work.index.values
    sub_indices = submission_work.index.values

    # Convert to numpy arrays for faster slicing
    sol_values = {
        "node_type": solution_work["node_type"].values,
        "node_id": solution_work["node_id"].values,
        "water_level": solution_work["water_level"].values,
    }
    sub_values = submission_work["water_level"].values

    del solution_work, submission_work
    gc.collect()

    # Process events using itertools.groupby (much faster than pandas groupby)
    event_count = 0
    prev_key = None
    start_idx = 0

    for i, key in enumerate(sol_keys):
        if key != prev_key:
            if prev_key is not None:
                # Process previous group
                end_idx = i

                node_types = sol_values["node_type"][start_idx:end_idx]
                node_ids = sol_values["node_id"][start_idx:end_idx]
                y_true = sol_values["water_level"][start_idx:end_idx]
                y_pred = sub_values[start_idx:end_idx]

                event_nse = calculate_event_nse_from_arrays(
                    node_types, node_ids, y_true, y_pred
                )

                if not np.isnan(event_nse):
                    model_event_nses[prev_key[0]].append(event_nse)

                event_count += 1

                # Log memory periodically
                if log_event_memory and event_count % memory_log_interval == 0:
                    elapsed = time.time() - start_time
                    rate = event_count / elapsed
                    log_memory(
                        f"Processed {event_count} events ({rate:.1f} events/sec)"
                    )

            prev_key = key
            start_idx = i

    # Process last group
    if prev_key is not None:
        end_idx = len(sol_keys)

        node_types = sol_values["node_type"][start_idx:end_idx]
        node_ids = sol_values["node_id"][start_idx:end_idx]
        y_true = sol_values["water_level"][start_idx:end_idx]
        y_pred = sub_values[start_idx:end_idx]

        event_nse = calculate_event_nse_from_arrays(
            node_types, node_ids, y_true, y_pred
        )

        if not np.isnan(event_nse):
            model_event_nses[prev_key[0]].append(event_nse)

        event_count += 1

    elapsed = time.time() - start_time
    print(
        f"\n✓ Processed {event_count} events in {elapsed:.2f}s ({event_count / elapsed:.1f} events/sec)"
    )

    del sol_values, sub_values, sol_keys, sub_keys, sol_indices, sub_indices
    gc.collect()

    # Calculate final scores
    model_scores = []

    for model_id in sorted(model_event_nses.keys()):
        event_nses = model_event_nses[model_id]

        if not event_nses:
            raise ParticipantVisibleError(
                f"Could not calculate NSE for any events in model_id {model_id}"
            )

        model_score = np.mean(event_nses)
        model_scores.append(model_score)
        print(
            f"Model {model_id}: NSE = {model_score:.6f} (from {len(event_nses)} events)"
        )

    if len(model_scores) != 3:
        raise ParticipantVisibleError(
            f"Expected 3 model scores but calculated {len(model_scores)}"
        )

    final_score = np.mean(model_scores)

    if not np.isfinite(final_score):
        raise ParticipantVisibleError("Calculated NSE is not finite")

    return float(final_score)


if __name__ == "__main__":
    final_score = score_from_files(
        solution_path="/Users/jiayulim/Documents/GitHub/dual_flood_gnn/kaggle_submission/solutions.csv",
        submission_path="/Users/jiayulim/Documents/GitHub/dual_flood_gnn/kaggle_submission/solution_without_usage.parquet",
        row_id_column_name="row_id",
    )

    print(f"\n{'=' * 50}")
    print(f"Final NSE Score: {final_score:.6f}")
    print(f"{'=' * 50}")
