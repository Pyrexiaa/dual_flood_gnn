import pandas as pd
import numpy as np
import pandas.api.types
import gc
from collections import defaultdict
import time
from concurrent.futures import ProcessPoolExecutor
import multiprocessing as mp


class ParticipantVisibleError(Exception):
    pass


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


def process_event_batch(batch_data):
    """Process a batch of events in parallel (worker function)"""
    batch_results = []

    for event_data in batch_data:
        node_types, node_ids, y_true, y_pred, model_id = event_data
        event_nse = calculate_event_nse_from_arrays(
            node_types, node_ids, y_true, y_pred
        )

        if not np.isnan(event_nse):
            batch_results.append((model_id, event_nse))

    return batch_results


def score(
    solution: pd.DataFrame,
    submission: pd.DataFrame,
    row_id_column_name: str,
    use_parallel: bool = True,
    n_workers: int = 4,
    batch_size: int = 30,
) -> float:
    """
    Memory-efficient hierarchical NSE calculation with optional parallel processing

    Args:
        use_parallel: Whether to use parallel processing (faster but uses more memory)
        n_workers: Number of parallel workers
        batch_size: Number of events per batch for parallel processing
        memory_log_interval: Log memory every N events
    """
    required_cols = ["model_id", "event_id", "node_type", "node_id", "water_level"]
    key_cols = ["model_id", "event_id", "node_type", "node_id"]

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
        raise ParticipantVisibleError('Submission column "water_level" must be numeric')

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

    start_time = time.time()

    sol_cols = key_cols + ["water_level"]
    solution_work = solution[sol_cols].copy()
    submission_work = submission[sol_cols].copy()

    del solution, submission
    gc.collect()

    solution_work = solution_work.sort_values(key_cols)
    submission_work = submission_work.sort_values(key_cols)

    for col in key_cols:
        if not solution_work[col].equals(submission_work[col]):
            raise ParticipantVisibleError(
                "Submission does not match solution structure"
            )

    sol_keys = list(zip(solution_work["model_id"], solution_work["event_id"]))

    sol_values = {
        "node_type": solution_work["node_type"].values,
        "node_id": solution_work["node_id"].values,
        "water_level": solution_work["water_level"].values,
        "model_id": solution_work["model_id"].values,
    }
    sub_values = submission_work["water_level"].values

    del solution_work, submission_work
    gc.collect()

    event_boundaries = []
    prev_key = None
    start_idx = 0

    for i, key in enumerate(sol_keys):
        if key != prev_key:
            if prev_key is not None:
                event_boundaries.append((prev_key, start_idx, i))
            prev_key = key
            start_idx = i

    if prev_key is not None:
        event_boundaries.append((prev_key, start_idx, len(sol_keys)))

    total_events = len(event_boundaries)
    print(f"Total events to process: {total_events}")

    model_event_nses = defaultdict(list)

    if use_parallel:
        if n_workers is None:
            n_workers = max(1, mp.cpu_count() - 1)

        print(f"Using {n_workers} workers with batch size {batch_size}")

        batches = []
        current_batch = []

        for event_key, start_idx, end_idx in event_boundaries:
            event_data = (
                sol_values["node_type"][start_idx:end_idx],
                sol_values["node_id"][start_idx:end_idx],
                sol_values["water_level"][start_idx:end_idx],
                sub_values[start_idx:end_idx],
                event_key[0],
            )
            current_batch.append(event_data)

            if len(current_batch) >= batch_size:
                batches.append(current_batch)
                current_batch = []

        if current_batch:
            batches.append(current_batch)

        print(f"Created {len(batches)} batches")

        with ProcessPoolExecutor(max_workers=n_workers) as executor:
            for batch_results in executor.map(process_event_batch, batches):
                for model_id, event_nse in batch_results:
                    model_event_nses[model_id].append(event_nse)

        del batches
        gc.collect()

    else:
        event_count = 0

        for event_key, start_idx, end_idx in event_boundaries:
            node_types = sol_values["node_type"][start_idx:end_idx]
            node_ids = sol_values["node_id"][start_idx:end_idx]
            y_true = sol_values["water_level"][start_idx:end_idx]
            y_pred = sub_values[start_idx:end_idx]
            model_id = event_key[0]

            event_nse = calculate_event_nse_from_arrays(
                node_types, node_ids, y_true, y_pred
            )

            if not np.isnan(event_nse):
                model_event_nses[model_id].append(event_nse)

            event_count += 1

    elapsed = time.time() - start_time
    print(
        f"\n✓ Processed {total_events} events in {elapsed:.2f}s ({total_events / elapsed:.1f} events/sec)"
    )

    del sol_values, sub_values, sol_keys, event_boundaries
    gc.collect()

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
