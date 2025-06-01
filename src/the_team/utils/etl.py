"""Write ETL functions here like data loading, cleaning, and transformation."""

import kagglehub
from kagglehub import KaggleDatasetAdapter
import pandas as pd
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from typing import Optional, Union
from unidecode import unidecode
import logging
# Set up logging
logging.basicConfig(level=logging.INFO, format='%(levelname)s - %(message)s')
    
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

def null_duplicate_check(df: pd.DataFrame, col: Optional[list[str]] = None, verbose: Optional[bool] = True) -> None:
    """
    Check for null values and duplicates in the DataFrame.
    
    Args:
        df (pd.DataFrame): DataFrame to check.
        col (list of str, optional): Specific column(s) to check for duplicated values. Defaults to None.
        verbose (bool, optional): If True, print detailed information. Defaults to True.
        
    Returns:
        None
    """
    # Check for null values
    if df.isnull().values.any():
        logging.warning("Null values found.")
        # Count the number of rows with any null values
        # and calculate the percentage of such rows
        rows_with_any_null = df.isnull().any(axis=1).sum()
        null_rows_percent = (rows_with_any_null / len(df)) * 100
        logging.info(f"{null_rows_percent:.2f}% or {rows_with_any_null} rows have 1 or more null values.")
        if verbose:
            logging.info("Null value counts per column:")
            logging.info(df.isnull().sum())
    else:
        print("No null values found.")
    
    # Check for duplicates
    if df.duplicated().any():
        logging.warning("Duplicates found.")
        # Count the number of complete duplicate rows
        # and calculate the percentage of such rows
        total_dupli = df.duplicated(subset=col).sum()
        dupli_percentage = (total_dupli / len(df)) * 100
        logging.info(f"{dupli_percentage:.2f}% or {total_dupli} rows are complete duplicates.")
        if verbose:
            logging.info(df[df.duplicated(keep=False)])  # Show all duplicates
    else:
        logging.info("No duplicates found.")
    

def cap_outliers(col: pd.Series,  
                 min_cap: Union[float, bool, None] = None,
                 max_cap: Union[float, bool, None] = None) -> pd.Series:
    """
    Cap outliers in a specified column of the DataFrame.
    
    Parameters:
        col (pd.Series): Series to apply capping.
        min_cap (float or bool): 
            - If float: use this as the lower cap value.
            - If True: use 1st percentile as lower cap.
            - If None or False: no lower capping.
        max_cap (float or bool): 
            - If float: use this as the upper cap value.
            - If True: use 99th percentile as upper cap.
            - If None or False: no upper capping.
    
    Returns:
        pd.Series: The capped column as a new Series.
    """

    # Determine cap values
    if isinstance(min_cap, bool) and min_cap:
        min_val = col.quantile(0.01)
    elif isinstance(min_cap, (int, float)):
        min_val = min_cap
    else:
        min_val = -np.inf

    if isinstance(max_cap, bool) and max_cap:
        max_val = col.quantile(0.99)
    elif isinstance(max_cap, (int, float)):
        max_val = max_cap
    else:
        max_val = np.inf

    # Apply capping
    capped_series = col.clip(lower=min_val, upper=max_val)

    return capped_series

def clean_location_columns(df: pd.DataFrame, zip_col: str, city_col: str, state_col: str) -> pd.DataFrame:  # gonna use when cleaning
    """
    Standardize zip code, city, and state columns for consistency in merging, grouping, and filtering.

    This function is reusable across datasets that share location-based structure, such as:
    - customers
    - geolocation
    - sellers

    Args:
        df (pd.DataFrame): The DataFrame to clean.
        zip_col (str): Name of the zip prefix column.
        city_col (str): Name of the city column.
        state_col (str): Name of the state column.

    Returns:
        pd.DataFrame: Cleaned dataframe with standardized location fields.
    """
    # 1. Convert zip prefix to string and remove whitespace
    df[zip_col] = df[zip_col].astype(str).str.strip()
    assert df[zip_col].dtype == "object", f"{zip_col} should be string"

    # 2. Normalize city names: lowercase, stripped, and accent-removed
    df[city_col] = df[city_col].str.lower().str.strip().apply(unidecode)

    # 3. Standardize state codes: uppercase and stripped
    df[state_col] = df[state_col].str.upper().str.strip()

    return df

def format_customers(customers: pd.DataFrame) -> pd.DataFrame:  # gonna use when cleaning
    """
    Format and standardize the customers dataset.

    This builds on location cleaning by also:
    - Ensuring string consistency in hashed customer ID fields
    - Supporting reliable merges and indexing later

    Args:
        customers (pd.DataFrame): Raw customer data.

    Returns:
        pd.DataFrame: Fully cleaned customer data ready for EDA.
    """

    # 1. Apply shared location cleaning logic (zip, city, state)
    customers = clean_location_columns(customers, "customer_zip_code_prefix", "customer_city", "customer_state")

    # 2. Clean hashed ID fields: convert to string and remove hidden whitespace
    customers["customer_id"] = customers["customer_id"].astype(str).str.strip()
    customers["customer_unique_id"] = customers["customer_unique_id"].astype(str).str.strip()

    return customers

def format_geolocation(geolocation: pd.DataFrame) -> pd.DataFrame:  # gonna use when cleaning
    """
    Format, standardize, and aggregate the geolocation dataset.

    This includes:
    - Standardizing zip prefix, city, and state for consistent merging
    - Aggregating to one average latitude/longitude per zip prefix

    Args:
        geolocation (pd.DataFrame): Raw geolocation data.

    Returns:
        pd.DataFrame: Aggregated dataset with one row per zip prefix and averaged coordinates.
    """

    # Step 1: Standardize text formatting for zip, city, and state
    geolocation = clean_location_columns(
        geolocation,
        zip_col="geolocation_zip_code_prefix",
        city_col="geolocation_city",
        state_col="geolocation_state"
    )

    # Step 2: Drop fully duplicated rows to avoid skewing averages
    geolocation = geolocation.drop_duplicates()

    # Step 3: Aggregate to average lat/lng per zip code prefix
    geo_prefix = (
        geolocation.groupby("geolocation_zip_code_prefix")[["geolocation_lat", "geolocation_lng"]]
        .mean()
        .reset_index()
    )

    return geo_prefix

def format_sellers(sellers: pd.DataFrame) -> pd.DataFrame:  # gonna use when cleaning
    """
    Format and standardize the sellers dataset.

    This function applies general location standardization and ensures
    the seller ID is string-cleaned for reliable merging or grouping.

    Args:
        sellers (pd.DataFrame): Raw seller dataset.

    Returns:
        pd.DataFrame: Cleaned sellers dataset.
    """

    # Step 1: Apply shared location cleaning (zip, city, state)
    sellers = clean_location_columns(
        sellers,
        zip_col="seller_zip_code_prefix",
        city_col="seller_city",
        state_col="seller_state"
    )

    # Step 2: Clean seller_id (convert to string, remove whitespace)
    sellers["seller_id"] = sellers["seller_id"].astype(str).str.strip()

    return sellers
def to_datetime(col: pd.Series, format:str="%Y-%m-%d %H:%M:%S") -> pd.Series:
    """
    Convert specified columns in the DataFrame to datetime format.
    
    Args:
        col (pd.Series): Column to convert.
        format (str): Format string for datetime conversion. Default is "%Y-%m-%d %H:%M:%S".
        
    Returns:
        pd.Series: Series with converted columns.
    """
    return pd.to_datetime(col, format=format, errors='coerce')

def clip_datetime(col: pd.Series, start_year:int = 2016, end_year:int = 2018) -> pd.Series:
    """
    Filter a datetime Series to only include rows where the year is within the given range (inclusive).

    Args:
        col (pd.Series): Series of datetime values.
        start_year (int): Start year (inclusive).
        end_year (int): End year (inclusive).

    Returns:
        pd.Series: Filtered Series with only the specified year range.
    """
    if not pd.api.types.is_datetime64_any_dtype(col):
        col = to_datetime(col)
    mask = (col.dt.year >= start_year) & (col.dt.year <= end_year)
    return col[mask]
