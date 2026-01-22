import pandas as pd
import numpy as np
from pathlib import Path


def nse(y_true, y_pred):
    """Calculate Nash-Sutcliffe Efficiency"""
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)

    denominator = np.sum((y_true - np.mean(y_true)) ** 2)

    # Safety check (important for dry or constant series)
    if denominator == 0:
        return np.nan

    return 1 - np.sum((y_true - y_pred) ** 2) / denominator


def calculate_hierarchical_nse_with_debug(pred_csv_path, save_details=True):
    """
    Calculate hierarchical NSE with detailed debugging information.

    Returns per-node NSE for both 1D and 2D nodes and identifies
    why final NSE might be NaN.

    Args:
        pred_csv_path: Path to predictions CSV
        save_details: Whether to save detailed per-node NSE to CSV

    Returns:
        dict: Contains overall NSE, per-node details, and debugging info
    """
    df = pd.read_csv(pred_csv_path)

    print("\n" + "=" * 80)
    print("HIERARCHICAL NSE CALCULATION WITH DEBUGGING")
    print("=" * 80)

    # Verify required columns
    required_cols = [
        "node_id",
        "event_id",
        "node_type",
        "target_water_level",
        "predicted_water_level",
    ]
    missing = [col for col in required_cols if col not in df.columns]
    if missing:
        raise ValueError(f"Missing columns: {missing}")

    # Overall statistics
    print("\nDataset Overview:")
    print(f"  Total samples: {len(df)}")
    print(f"  Unique events: {df['event_id'].nunique()}")
    print(f"  Unique nodes: {df['node_id'].nunique()}")
    print(f"  1D nodes (type=0): {len(df[df['node_type'] == 0])}")
    print(f"  2D nodes (type=1): {len(df[df['node_type'] == 1])}")

    # Storage for all per-node NSE values
    all_node_nse_records = []

    # Separate analysis for 1D and 2D
    results = {}

    for node_type_val, node_type_name in [(0, "1D"), (1, "2D")]:
        print(f"\n{'=' * 80}")
        print(f"ANALYZING {node_type_name} NODES (node_type={node_type_val})")
        print(f"{'=' * 80}")

        df_type = df[df["node_type"] == node_type_val]

        if len(df_type) == 0:
            print(f"  No {node_type_name} samples found!")
            results[node_type_name] = {
                "overall_nse": np.nan,
                "reason": "No samples",
                "per_node_nse": [],
            }
            continue

        unique_events = sorted(df_type["event_id"].unique())
        unique_nodes = sorted(df_type["node_id"].unique())

        print(f"\n  Events with {node_type_name} data: {len(unique_events)}")
        print(f"  Unique {node_type_name} nodes: {len(unique_nodes)}")

        # Calculate per-node NSE for each event
        event_nse_list = []
        valid_node_count = 0
        nan_node_count = 0
        insufficient_data_count = 0

        for event_id in unique_events:
            df_event = df_type[df_type["event_id"] == event_id]
            event_nodes = df_event["node_id"].unique()

            node_nse_for_event = []

            for node_id in event_nodes:
                df_node = df_event[df_event["node_id"] == node_id]

                if len(df_node) <= 1:
                    insufficient_data_count += 1
                    all_node_nse_records.append(
                        {
                            "node_type": node_type_name,
                            "event_id": event_id,
                            "node_id": node_id,
                            "n_samples": len(df_node),
                            "nse": np.nan,
                            "reason": "insufficient_data",
                        }
                    )
                    continue

                y_true = df_node["target_water_level"].values
                y_pred = df_node["predicted_water_level"].values

                # Calculate NSE
                nse_val = nse(y_true, y_pred)

                # Record details
                record = {
                    "node_type": node_type_name,
                    "event_id": event_id,
                    "node_id": node_id,
                    "n_samples": len(df_node),
                    "nse": nse_val,
                    "y_true_mean": y_true.mean(),
                    "y_true_std": y_true.std(),
                    "y_pred_mean": y_pred.mean(),
                    "rmse": np.sqrt(np.mean((y_true - y_pred) ** 2)),
                }

                if np.isnan(nse_val):
                    nan_node_count += 1
                    # Check why it's NaN
                    if y_true.std() == 0:
                        record["reason"] = "constant_target"
                    else:
                        record["reason"] = "unknown_nan"
                else:
                    valid_node_count += 1
                    node_nse_for_event.append(nse_val)
                    record["reason"] = "valid"

                all_node_nse_records.append(record)

            # Average per-node NSE to get event NSE
            if node_nse_for_event:
                event_nse = np.mean(node_nse_for_event)
                event_nse_list.append(event_nse)

        # Calculate overall NSE
        if event_nse_list:
            overall_nse = np.mean(event_nse_list)
        else:
            overall_nse = np.nan

        # Summary
        print("\n  Per-Node NSE Summary:")
        print(f"    Valid nodes: {valid_node_count}")
        print(f"    NaN nodes: {nan_node_count}")
        print(f"    Insufficient data (<2 samples): {insufficient_data_count}")
        print(
            f"    Total nodes processed: {valid_node_count + nan_node_count + insufficient_data_count}"
        )

        if valid_node_count > 0:
            valid_nse = [
                r["nse"]
                for r in all_node_nse_records
                if r["node_type"] == node_type_name and not np.isnan(r["nse"])
            ]
            print("\n  Valid NSE Statistics:")
            print(f"    Mean: {np.mean(valid_nse):.4f}")
            print(f"    Std: {np.std(valid_nse):.4f}")
            print(f"    Min: {np.min(valid_nse):.4f}")
            print(f"    Max: {np.max(valid_nse):.4f}")

        print(f"\n  Events with valid NSE: {len(event_nse_list)}")
        print(
            f"  Overall {node_type_name} NSE (hierarchical): {overall_nse:.4f}"
            if not np.isnan(overall_nse)
            else f"  Overall {node_type_name} NSE: NaN"
        )

        # Identify problematic cases
        if nan_node_count > 0:
            print("\n  NaN Breakdown:")
            nan_records = [
                r
                for r in all_node_nse_records
                if r["node_type"] == node_type_name and np.isnan(r["nse"])
            ]

            reason_counts = {}
            for r in nan_records:
                reason = r.get("reason", "unknown")
                reason_counts[reason] = reason_counts.get(reason, 0) + 1

            for reason, count in reason_counts.items():
                print(f"    {reason}: {count}")

        # Show some examples
        if valid_node_count > 0:
            print("\n  Example Valid Nodes (first 5):")
            valid_examples = [
                r
                for r in all_node_nse_records
                if r["node_type"] == node_type_name and not np.isnan(r["nse"])
            ][:5]
            for ex in valid_examples:
                print(
                    f"    Event {ex['event_id']}, Node {ex['node_id']}: "
                    f"NSE={ex['nse']:.4f}, n={ex['n_samples']}"
                )

        if nan_node_count > 0:
            print("\n  Example NaN Nodes (first 5):")
            nan_examples = [
                r
                for r in all_node_nse_records
                if r["node_type"] == node_type_name and np.isnan(r["nse"])
            ][:5]
            for ex in nan_examples:
                print(
                    f"    Event {ex['event_id']}, Node {ex['node_id']}: "
                    f"reason={ex.get('reason', 'unknown')}, n={ex['n_samples']}"
                )

        results[node_type_name] = {
            "overall_nse": overall_nse,
            "valid_nodes": valid_node_count,
            "nan_nodes": nan_node_count,
            "insufficient_data": insufficient_data_count,
            "event_nse_list": event_nse_list,
            "per_node_nse": all_node_nse_records,
        }

    # Save detailed results
    if save_details:
        pred_path = Path(pred_csv_path)

        # Save all per-node NSE
        node_nse_df = pd.DataFrame(all_node_nse_records)
        output_path = pred_path.parent / f"{pred_path.stem}_per_node_nse.csv"
        node_nse_df.to_csv(output_path, index=False)
        print(f"\n✓ Saved per-node NSE details to: {output_path}")

        # Summary report
        summary_path = pred_path.parent / f"{pred_path.stem}_nse_debug.txt"
        with open(summary_path, "w") as f:
            f.write("=" * 80 + "\n")
            f.write("HIERARCHICAL NSE DEBUG REPORT\n")
            f.write("=" * 80 + "\n\n")

            for node_type_name in ["1D", "2D"]:
                if node_type_name in results:
                    res = results[node_type_name]
                    f.write(f"{node_type_name} NODES:\n")
                    f.write(f"  Overall NSE: {res['overall_nse']}\n")
                    f.write(f"  Valid nodes: {res['valid_nodes']}\n")
                    f.write(f"  NaN nodes: {res['nan_nodes']}\n")
                    f.write(f"  Insufficient data: {res['insufficient_data']}\n")
                    f.write("\n")

        print(f"✓ Saved debug summary to: {summary_path}")

    print("\n" + "=" * 80)
    print("FINAL RESULTS")
    print("=" * 80)
    print(f"1D NSE: {results.get('1D', {}).get('overall_nse', 'N/A')}")
    print(f"2D NSE: {results.get('2D', {}).get('overall_nse', 'N/A')}")
    print("=" * 80 + "\n")

    return results


# Example usage
if __name__ == "__main__":
    # Replace with your actual file path
    results = calculate_hierarchical_nse_with_debug(
        "model2_gru_test_predictions_test.csv"
    )

    # Access the results
    print("\nYou can now investigate:")
    print("- results['1D']['per_node_nse'] for all 1D node details")
    print("- results['2D']['per_node_nse'] for all 2D node details")
    print("- Check the saved CSV for complete per-node breakdown")
