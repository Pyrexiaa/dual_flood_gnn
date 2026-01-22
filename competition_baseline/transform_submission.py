import pandas as pd
from pathlib import Path
from typing import Union, List


def transform_csv_to_submission_format(
    csv_path: Union[str, Path],
    model_id: int,
    timestep_threshold: int = 10,
    event_ids: List[int] = None
) -> pd.DataFrame:
    """
    Transform a single CSV file to submission format.
    
    Args:
        csv_path: Path to input CSV file
        model_id: Model ID to assign (1 or 2)
        timestep_threshold: Only include timesteps >= this value (default: 10)
        event_ids: Optional list of event IDs to filter. If provided, only these events will be included.
                   Raises ValueError if any specified event_id is not found in the data.
    
    Returns:
        pd.DataFrame: Transformed dataframe
    """
    # Load CSV
    df = pd.read_csv(csv_path)
    
    print(f"Processing: {csv_path}")
    print(f"  Original rows: {len(df):,}")
    
    # Filter by event_ids if provided
    if event_ids is not None:
        available_event_ids = set(df['event_id'].unique())
        requested_event_ids = set(event_ids)
        
        # Check for missing event_ids
        missing_event_ids = requested_event_ids - available_event_ids
        if missing_event_ids:
            raise ValueError(
                f"Event IDs not found in {csv_path}: {sorted(missing_event_ids)}. "
                f"Available event IDs: {sorted(available_event_ids)}"
            )
        
        print(f"  Filtering for event IDs: {sorted(event_ids)}")
        df = df[df['event_id'].isin(event_ids)].copy()
        print(f"  After event_id filtering: {len(df):,}")
    
    # Filter timesteps >= threshold
    df = df[df['timestep'] >= timestep_threshold].copy()
    print(f"  After filtering timestep >= {timestep_threshold}: {len(df):,}")
    
    # Transform node_type: 0 -> 1, 1 -> 2
    df['node_type'] = df['node_type'] + 1
    
    # Create output dataframe with required columns
    output_df = pd.DataFrame({
        'model_id': model_id,
        'event_id': df['event_id'],
        'node_type': df['node_type'],
        'node_id': df['node_id'],
        'water_level': df['predicted_water_level']
    })
    
    return output_df


def combine_and_transform_csvs(
    csv_paths: List[Union[str, Path]],
    model_ids: List[int],
    output_path: str = "combined_submission.csv",
    timestep_threshold: int = 10,
    event_ids: Union[List[int], List[List[int]]] = None
) -> pd.DataFrame:
    """
    Transform multiple CSV files and combine them into a single submission file.
    
    Args:
        csv_paths: List of paths to input CSV files
        model_ids: List of model IDs corresponding to each CSV (e.g., [1, 2])
        output_path: Path to save the combined output CSV
        timestep_threshold: Only include timesteps >= this value (default: 10)
        event_ids: Optional event ID filtering. Can be:
                   - None: Include all events for all models
                   - List[int]: Same event IDs applied to all models (e.g., [0, 1, 2])
                   - List[List[int]]: Different event IDs for each model (e.g., [[0, 1], [2, 3]])
                   Raises ValueError if any specified event_id is not found in the corresponding CSV.
    
    Returns:
        pd.DataFrame: Combined transformed dataframe with row_id
    """
    if len(csv_paths) != len(model_ids):
        raise ValueError(f"Number of csv_paths ({len(csv_paths)}) must match number of model_ids ({len(model_ids)})")
    
    # Handle event_ids parameter
    if event_ids is None:
        # No filtering for any model
        event_ids_per_model = [None] * len(csv_paths)
    elif isinstance(event_ids[0], list):
        # Different event_ids for each model
        if len(event_ids) != len(csv_paths):
            raise ValueError(f"Number of event_ids lists ({len(event_ids)}) must match number of csv_paths ({len(csv_paths)})")
        event_ids_per_model = event_ids
    else:
        # Same event_ids for all models
        event_ids_per_model = [event_ids] * len(csv_paths)
    
    print("="*80)
    print("TRANSFORMING AND COMBINING CSV FILES")
    print("="*80 + "\n")
    
    all_dfs = []
    
    # Transform each CSV
    for csv_path, model_id, model_event_ids in zip(csv_paths, model_ids, event_ids_per_model):
        transformed_df = transform_csv_to_submission_format(
            csv_path, 
            model_id, 
            timestep_threshold,
            model_event_ids
        )
        all_dfs.append(transformed_df)
        print()
    
    # Combine all dataframes
    combined_df = pd.concat(all_dfs, ignore_index=True)
    
    print(f"Combined total rows: {len(combined_df):,}")
    
    # Add row_id starting from 0
    combined_df.insert(0, 'row_id', range(len(combined_df)))
    
    # Reorder columns to match required format
    combined_df = combined_df[['row_id', 'model_id', 'event_id', 'node_type', 'node_id', 'water_level']]
    
    # Save to CSV
    output_path = Path(output_path)
    combined_df.to_csv(output_path, index=False)
    
    print("\n" + "="*80)
    print("TRANSFORMATION COMPLETE")
    print("="*80)
    print(f"Output saved to: {output_path.absolute()}")
    print(f"Total rows: {len(combined_df):,}")
    print(f"Columns: {list(combined_df.columns)}")
    
    # Print sample of output
    print("\nFirst few rows:")
    print(combined_df.head(10))
    
    print("\nLast few rows:")
    print(combined_df.tail(5))
    
    # Print summary statistics
    print("\n" + "="*80)
    print("SUMMARY")
    print("="*80)
    print(f"Model IDs: {sorted(combined_df['model_id'].unique())}")
    print(f"Events: {combined_df['event_id'].nunique()} unique")
    print(f"Nodes: {combined_df['node_id'].nunique()} unique")
    print(f"Node types: {sorted(combined_df['node_type'].unique())}")
    print(f"Water level range: [{combined_df['water_level'].min():.4f}, {combined_df['water_level'].max():.4f}]")
    print("="*80 + "\n")
    
    return combined_df


# Example usage:
if __name__ == "__main__":
    model1_test_events = [5, 8, 18, 22, 26, 29, 31, 33, 35, 37, 42, 44, 48, 51, 52, 53, 59, 62, 65, 66, 67, 69, 73, 75, 80, 81, 83, 88, 97]
    model2_test_events = [4, 8, 17, 18, 22, 29, 31, 35, 37, 42, 44, 51, 52, 53, 54, 59, 60, 61, 62, 65, 66, 67, 73, 76, 77, 82, 84, 88, 90, 99]

    # Example 1: Transform and combine 2 CSV files
    csv_paths = [
        "submissions/model1_gru_test_predictions_test.csv",
        "submissions/model2_gru_test_predictions_test.csv",
    ]
    
    model_ids = [1, 2]
    
    result = combine_and_transform_csvs(
        csv_paths=csv_paths,
        model_ids=model_ids,
        output_path="submissions/combined_submission.csv",
        timestep_threshold=10,
        event_ids=[model1_test_events, model2_test_events]  # Different event IDs for each model
    )
    
    # # Example 2: If you want to process just one CSV
    # single_result = transform_csv_to_submission_format(
    #     csv_path="submissions/model1_gru_test_predictions_test.csv",
    #     model_id=1,
    #     timestep_threshold=10,
    #     event_ids=model1_test_events,
    # )
    # single_result.insert(0, 'row_id', range(len(single_result)))
    # single_result.to_csv("submissions/model1_single_submission.csv", index=False)