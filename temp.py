import pandas as pd
import random

def get_random_rows(csv_filepath, n=10, random_seed=None):
    """
    Randomly select n rows from a CSV file.
    
    Parameters:
    -----------
    csv_filepath : str
        Path to the CSV file
    n : int, optional
        Number of random rows to select (default: 10)
    random_seed : int, optional
        Seed for reproducibility (default: None)
    
    Returns:
    --------
    pandas.DataFrame
        DataFrame containing n randomly selected rows
    """
    # Set random seed if provided for reproducibility
    if random_seed is not None:
        random.seed(random_seed)
    
    # Read the CSV file
    df = pd.read_csv(csv_filepath)
    
    # Ensure n doesn't exceed the number of rows
    n = min(n, len(df))
    
    # Sample n random rows
    random_rows = df.sample(n=n, random_state=random_seed)
    
    return random_rows


# Example usage:
if __name__ == "__main__":
    # Example: Get 10 random rows with a seed for reproducibility
    result_with_seed = get_random_rows('/Users/jiayulim/Documents/GitHub/dual_flood_gnn/data/Model4/raw/test.csv', n=8, random_seed=42)
    print("\nWith seed=42:")
    print(result_with_seed)
    
    # Save to a new CSV if needed
    result_with_seed.to_csv('random_subset.csv', index=False)