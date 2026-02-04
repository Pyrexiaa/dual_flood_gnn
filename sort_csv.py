import pandas as pd

def reindex_row_id(df):
    """
    Resets the row_id column to start from 0 and increment sequentially.
    
    Parameters:
    df (pd.DataFrame): Input dataframe with row_id column
    
    Returns:
    pd.DataFrame: Dataframe with reindexed row_id
    """
    df_copy = df.copy()
    df_copy['row_id'] = range(len(df_copy))
    return df_copy


def sort_csv(df):
    """
    Sorts the CSV by:
    1. model_id (ascending, so 1 comes first)
    2. event_id (ascending, smallest first)
    3. node_type (ascending, so 1 comes first)
    4. node_id (ascending, smallest first)
    
    Parameters:
    df (pd.DataFrame): Input dataframe
    
    Returns:
    pd.DataFrame: Sorted dataframe
    """
    df_sorted = df.sort_values(
        by=['model_id', 'event_id', 'node_type', 'node_id'],
        ascending=[True, True, True, True]
    ).reset_index(drop=True)
    
    return df_sorted


def process_csv(input_file, output_file=None):
    """
    Complete processing: sorts the CSV and reindexes row_id.
    
    Parameters:
    input_file (str): Path to input CSV file
    output_file (str): Path to output CSV file (optional)
    
    Returns:
    pd.DataFrame: Processed dataframe
    """
    # Read the CSV
    df = pd.read_csv(input_file)
    
    # Sort by specified columns
    df_sorted = sort_csv(df)
    
    # Reindex row_id
    df_final = reindex_row_id(df_sorted)
    
    # Save to file if output path provided
    if output_file:
        df_final.to_csv(output_file, index=False)
        print(f"Processed CSV saved to: {output_file}")
    
    return df_final


# Example usage
if __name__ == "__main__":
     # Example with your data
    input_csv = "kaggle_submissions/node_only_8.csv"
    output_csv = "kaggle_submissions/node_only_8_sorted.csv"  # Replace with desired output path
    
    # Process the CSV
    df_result = process_csv(input_csv, output_csv)
    
    print("\nProcessed Data:")
    print(df_result)

     # Example with your data
    input_csv = "kaggle_submissions/node_only_9.csv"
    output_csv = "kaggle_submissions/node_only_9_sorted.csv"  # Replace with desired output path
    
    # Process the CSV
    df_result = process_csv(input_csv, output_csv)
    
    print("\nProcessed Data:")
    print(df_result)