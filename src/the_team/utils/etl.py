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

def null_duplicate_check(df: pd.DataFrame, col: Optional[list[str]] = None) -> None:
    """
    Check for null values and duplicates in the DataFrame.
    
    Args:
        df (pd.DataFrame): DataFrame to check.
        col (list of str, optional): Specific column(s) to check for duplicated values. Defaults to None.
        
    Returns:
        None
    """
    # Check for null values
    if df.isnull().values.any():
        print("Null values found.")
        # Count the number of rows with any null values
        # and calculate the percentage of such rows
        rows_with_any_null = df.isnull().any(axis=1).sum()
        null_rows_percent = (rows_with_any_null / len(df)) * 100
        print(f"{null_rows_percent:.2f}% or {rows_with_any_null} rows have 1 or more null values.")
        print("Null value counts per column:")
        print(df.isnull().sum())
    else:
        print("No null values found.")
    
    # Check for duplicates
    if df.duplicated().any():
        print("Duplicates found.")
        # Count the number of complete duplicate rows
        # and calculate the percentage of such rows
        total_dupli = df.duplicated(subset=col).sum()
        dupli_percentage = (total_dupli / len(df)) * 100
        print(f"{dupli_percentage:.2f}% or {total_dupli} rows are complete duplicates.")
        print(df[df.duplicated(keep=False)])  # Show all duplicates
    else:
        print("No duplicates found.")
        
def set_plot_style() -> None:
    """
    Set the plot style for consistent visualizations.
    
    Args:
        None
    Returns:
        None
    """
    plot_style_dict = {
        'font.family': ['Arial', 'Helvetica', 'sans-serif'],
        'font.sans-serif': ['Arial', 'Helvetica', 'sans-serif'],
        'axes.facecolor': '#f2f0e8',
        'axes.edgecolor': 'black',
        'axes.labelcolor': '#011547',
        'axes.labelsize': 12,
        'axes.labelweight': 'bold',
        'axes.titlesize': 14,
        'axes.titleweight': 'bold',
        'axes.titlepad': 15,
        'text.color': '#011547',
        'xtick.color': '#011547',
        'ytick.color': '#011547',
        'figure.figsize': (10, 6),
    }
    sns.set_theme(palette="husl", rc=plot_style_dict)
    plt.rcParams.update(plot_style_dict)
    print("Custom plot style set.")
    
def plot_numeric_distribution(df: pd.DataFrame) -> None:
    """
    Plot the distribution of all numeric columns in the DataFrame.
    
    Args:
        df (pd.DataFrame): DataFrame containing the data.
        column (str): Column name to plot.
        
    Returns:
        None
    """
    # Select numeric columns only
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    n = len(numeric_cols)

    # Create subplots: one column, multiple rows
    fig, axes = plt.subplots(n, 1, figsize=(12, 2.5 * n), sharey=True)

    # Ensure axes is iterable
    if n == 1:
        axes = [axes]

    # Plot with shared y-axis but individual x-axis for readability in different scales
    for i, col in enumerate(numeric_cols):
        sns.boxplot(x=df[col], ax=axes[i], orient='h')
        # Add vertical lines for 1st and 99th percentiles
        p1 = df[col].quantile(0.01)
        p99 = df[col].quantile(0.99)
        axes[i].axvline(p1, color='red', linestyle='--', label='1st percentile')
        axes[i].axvline(p99, color='green', linestyle='--', label='99th percentile')
        
        axes[i].set_title(f"{col}", loc='left', fontsize=12, fontweight='bold', color='#011547')
        axes[i].set_xlabel("")  
        axes[i].set_ylabel("")  
        axes[i].legend(loc='lower right', ncol=2, fontsize=10, frameon=False)

    # Shared xlabel and title
    fig.suptitle("Boxplot of Numeric Columns", fontsize=14, fontweight='bold', color='#011547')
    fig.supxlabel("Value", fontsize=12, fontweight='bold', color='#011547')

    plt.tight_layout()
    plt.show()

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
