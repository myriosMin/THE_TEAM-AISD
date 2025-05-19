"""Write ETL functions here like data loading, cleaning, and transformation."""

import kagglehub
from kagglehub import KaggleDatasetAdapter
import pandas as pd
from pathlib import Path

def load_csv(file_path: Path) -> pd.DataFrame:
    """
    Load a CSV file into a pandas DataFrame. Downloads the file if it does not exist locally.
    
    Args:
        file_path (str): Path to the CSV file.
        
    Returns:
        pd.DataFrame: Loaded DataFrame.
    """
    # Download the file if it does not exist (fixed to Olist dataset from Kaggle for demonstration)
    if not file_path.exists():
        print(f"File {file_path.name} not found. Downloading from Kaggle...")
        
        df = kagglehub.load_dataset(
        KaggleDatasetAdapter.PANDAS,
        "olistbr/brazilian-ecommerce",
        file_path.name,
        )

        df.to_csv(file_path, index=False)
        print(f"File {file_path.name} downloaded successfully and loaded into {file_path.parent}.")
        
    # Returns the pandas DataFrame    
    return pd.read_csv(file_path) 