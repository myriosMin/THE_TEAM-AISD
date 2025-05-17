"""Write ETL functions here like data loading, cleaning, and transformation."""

import pandas as pd

def load_csv(file_path):
    """
    Load a CSV file into a pandas DataFrame.
    
    Args:
        file_path (str): Path to the CSV file.
        
    Returns:
        pd.DataFrame: Loaded DataFrame.
    """
    if not file_path.exists():
        raise FileNotFoundError(f"{file_path} does not exist. Please check the path.")
    return pd.read_csv(file_path) 