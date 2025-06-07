"""
This is a boilerplate pipeline 'data_engineering'
generated using Kedro 0.19.12
"""
import pandas as pd
from datetime import datetime
import numpy as np
from unidecode import unidecode # type: ignore
import logging
logger = logging.getLogger(__name__)

# Make utils importable in Kedro
import sys
import os
src_path = os.path.abspath("../../../src")
if src_path not in sys.path:
    sys.path.insert(0, src_path)
from the_team.utils import etl

# Suppress warnings for cleaner output
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
warnings.simplefilter(action='ignore', category=pd.errors.SettingWithCopyWarning)

def clean_orders_dataset(orders: pd.DataFrame, date_cols: list, to_drop: list) -> pd.DataFrame:
    """
    Clean the olist_orders_dataset with the following steps:
    1. Convert date columns to datetime, and filter to 2016–2018
    2. Drop 'order_delivered_carrier_date'
    3. Drop rows with unwanted order statuses; unpaid
    4. Impute missing 'order_delivered_customer_date' using median duration between expected and actual delivery
    5. Drop rows with null 'order_approved_at'

    Args:
        orders (pd.DataFrame): Raw orders dataset
        date_cols (list): List of date columns to convert
        to_drop (list): List of columns to drop
        
    Returns:
        pd.DataFrame: Cleaned and filtered orders
    """
    logger.info("Starting orders dataset cleaning...")
    # Step 1: Convert to datetime & filter to 2016–2018
    for col in orders[date_cols]:
        orders[col] = etl.clip_datetime(orders[col])
    logger.info(f"Orders date range: {orders['order_purchase_timestamp'].min()} to {orders['order_purchase_timestamp'].max()}")

    # Step 2: Drop 'order_delivered_carrier_date'
    orders = orders.drop(columns=to_drop)
    logger.info(f"Orders shape after dropping columns: {orders.shape}")

    # Step 3: Remove unwanted order statuses
    unwanted_statuses = ["canceled", "created", "invoiced", "unavailable"]
    orders = orders[~orders["order_status"].isin(unwanted_statuses)]
    logger.info(f"Orders shape after removing unwanted statuses: {orders.shape}")

    # Step 4: Fill nulls in 'order_delivered_customer_date'
    # Calculate median difference (actual - estimated)
    delivery_diff = (
        orders["order_estimated_delivery_date"] - orders["order_delivered_customer_date"]
    )
    median_offset = delivery_diff.dropna().median()
    orders["order_delivered_customer_date"] = orders["order_delivered_customer_date"].fillna(
        orders["order_estimated_delivery_date"] - median_offset
    )
    logger.info(f"Orders shape after imputing delivery dates: {orders.shape}")
    
    # Step 5: Drop nulls in 'order_approved_at'
    orders = orders.dropna(subset=["order_approved_at"])
    logger.info(f"Orders shape after dropping nulls in 'order_approved_at': {orders.shape}")

    logger.info("Orders dataset cleaning completed.")
    
    return orders

def clean_items_dataset(items: pd.DataFrame, to_drop: list, to_cap: list, upper_cap_value: float) -> pd.DataFrame:
    """
    Clean the olist_order_items dataset with the following steps:
    1. Drop 'shipping_limit_date' column
    3. Cap outliers in price and freight_value (99.9th percentile)

    Args:
        items (pd.DataFrame): Raw items data
        to_drop (list): List of columns to drop
        to_cap (list): List of columns to cap outliers
        upper_cap_value (float): Value to cap outliers at (e.g., 99.9th percentile)

    Returns:
        pd.DataFrame: Cleaned items data
    """
    logger.info("Starting items dataset cleaning...")
    # Step 1: Drop 'shipping_limit_date' column
    items = items.drop(columns=to_drop)
    logger.info(f"Items shape after dropping columns: {items.shape}")

    # Step 2: Cap outliers at 99.9th percentile
    for col in to_cap:
        upper = np.percentile(items[col], upper_cap_value)
        items[col] = np.where(items[col] > upper, upper, items[col])
    logger.info(f"Items shape after capping outliers in {to_cap}: {items.shape}")

    logger.info("Items dataset cleaning completed.")
    
    return items

def clean_location_columns(df: pd.DataFrame, zip_col: str, city_col: str, state_col: str) -> pd.DataFrame:
    """
    Standardize zip code, city, and state columns for consistency in merging, grouping, and filtering.

    Args:
        df (pd.DataFrame): The DataFrame to clean.
        zip_col (str): Name of the zip prefix column.
        city_col (str): Name of the city column.
        state_col (str): Name of the state column.

    Returns:
        pd.DataFrame: Cleaned dataframe with standardized location fields.
    """
    logger.info("Starting location columns cleaning...")
    # Step 1:  Convert zip prefix to string and remove whitespace
    # df[zip_col] = df[zip_col].astype(str).str.strip()
    df[zip_col] = df[zip_col].astype(str).str.strip().str.zfill(5)
    assert df[zip_col].dtype == "object", f"{zip_col} should be string"
    logger.info(f"Zip column '{zip_col}' standardized to string with leading zeros.")
    
    # Step 2:  Normalize city names
    df[city_col] = df[city_col].str.lower().str.strip().apply(unidecode)
    logger.info(f"City column '{city_col}' normalized to lowercase, stripped, and accent-removed.")
    
    # Step 3:  Standardize state codes
    df[state_col] = df[state_col].str.upper().str.strip()
    logger.info(f"State column '{state_col}' standardized to uppercase and stripped.")
    
    logger.info("Location columns cleaning completed.")
    return df


def clean_customers_dataset(customers: pd.DataFrame) -> pd.DataFrame:
    """
    Clean the olist_customers dataset with the following steps:
    This dataset is already clean, so we will just return it as is.

    Args:
        customers (pd.DataFrame): Raw customers data

    Returns:
        pd.DataFrame: Cleaned customers data
    """
    logger.info("Starting customers dataset cleaning...")
    logger.info("Customers dataset cleaning completed.")
    return clean_location_columns(customers, "customer_zip_code_prefix", "customer_city", "customer_state")

def clean_geolocation_dataset(geolocation: pd.DataFrame) -> pd.DataFrame:
    """
    Clean and format the olist_geolocation dataset:
    1. Remove lat/lng outliers
    2. Standardize zip, city, state
    3. Drop duplicates
    4. Identify zip prefixes with large lat/lng range and drop them
    5. Aggregate to average lat/lng per zip prefix

    Args:
        geolocation (pd.DataFrame): Raw geolocation data

    Returns:
        pd.DataFrame: Aggregated geolocation data (1 row per zip prefix)
    """
    logger.info("Starting geolocation dataset cleaning...")
    # Step 1: Remove invalid lat/lng rows (outside Brazil’s bounding box)
    valid_lat_range = (-33.75116944, 5.27438888)
    valid_lng_range = (-73.98283055, -34.79314722)
    logger.info(f"Filtering geolocation data to valid lat range: {valid_lat_range} and lng range: {valid_lng_range}")
    geolocation = geolocation[
        (geolocation["geolocation_lat"].between(*valid_lat_range)) &
        (geolocation["geolocation_lng"].between(*valid_lng_range))
    ]
    logger.info(f"Geolocation shape after removing invalid lat/lng: {geolocation.shape}")
    
    # Step 2: Standardize text formatting (zip, city, state)
    geolocation = clean_location_columns(
        geolocation,
        zip_col="geolocation_zip_code_prefix",
        city_col="geolocation_city",
        state_col="geolocation_state"
    )
    logger.info("Geolocation columns standardized: zip, city, state.")

    # Step 3: Remove exact duplicate rows
    geolocation = geolocation.drop_duplicates()
    logger.info(f"Geolocation shape after dropping duplicates: {geolocation.shape}")
    
    # Step 4: Compute per-zip min/max/std and drop zip prefixes with large spread
    #   4a. Group by zip prefix, compute lat_min, lat_max, lng_min, lng_max, etc.
    geo_range = (
        geolocation.groupby("geolocation_zip_code_prefix")
        .agg({
            "geolocation_lat": ["min", "max"],
            "geolocation_lng": ["min", "max"]
        })
    )
    # Flatten column names
    geo_range.columns = ["lat_min", "lat_max", "lng_min", "lng_max"]
    geo_range = geo_range.reset_index()
    # Compute actual numeric range for lat and lng
    geo_range["lat_range"] = geo_range["lat_max"] - geo_range["lat_min"]
    geo_range["lng_range"] = geo_range["lng_max"] - geo_range["lng_min"]

    #   4b. Identify zip prefixes where either lat_range or lng_range > 0.5 degrees
    noisy_zips = geo_range[
        (geo_range["lat_range"] > 0.5) | (geo_range["lng_range"] > 0.5)
    ]["geolocation_zip_code_prefix"].unique()

    #   4c. Drop all rows from geolocation whose zip prefix is in noisy_zips
    geolocation = geolocation[
        ~geolocation["geolocation_zip_code_prefix"].isin(noisy_zips)
    ].reset_index(drop=True)
    logger.info(f"Geolocation shape after dropping noisy zip prefixes: {geolocation.shape}")

    # Step 5: Aggregate to average lat/lng per zip prefix
    geolocation = (
        geolocation
        .groupby("geolocation_zip_code_prefix")[["geolocation_lat", "geolocation_lng"]]
        .mean()
        .reset_index()
    )
    logger.info(f"Geolocation shape after aggregating to average lat/lng: {geolocation.shape}")

    logger.info("Geolocation dataset cleaning completed.")
    
    return geolocation

def clean_payments_dataset(payments: pd.DataFrame, drop_payment_type: str, to_drop: list) -> pd.DataFrame:
    """
    Clean the olist_payments dataset with the following steps:
    1. Drop rows with 'not_defined' payment_type.
    2. Clip 'payment_installments' at 99th percentile to handle outliers.
    3. Drop 'payment_sequential' and 'payment_value' as they're not relevant.

    Args:
        payments (pd.DataFrame): Raw payments data

    Returns:
        pd.DataFrame: Cleaned payments data
    """
    logger.info("Starting payments dataset cleaning...")
    
    # Step 1: Drop 'not_defined' payment types
    payments = payments[payments["payment_type"] != drop_payment_type]
    logger.info(f"Payments shape after dropping '{drop_payment_type}' payment types: {payments.shape}")

    # Step 2: Clip 'payment_installments' to 99th percentile
    upper_clip = payments["payment_installments"].quantile(0.99)
    payments["payment_installments"] = payments["payment_installments"].clip(upper=upper_clip)
    logger.info(f"Payments shape after clipping 'payment_installments' at {upper_clip}: {payments.shape}")
    
    # Step 3: Drop irrelevant columns
    payments.drop(columns=to_drop, inplace=True)
    logger.info(f"Payments shape after dropping columns {to_drop}: {payments.shape}")
    
    logger.info("Payments dataset cleaning completed.")
    
    return payments

def clean_reviews_dataset(reviews: pd.DataFrame) -> pd.DataFrame:
    """
    Clean the olist_reviews dataset with the following steps:
    1.  Drops all non essential columns

    Args:
        reviews (pd.DataFrame): Raw reviews data

    Returns:
        pd.DataFrame: Cleaned reviews data
    """
    logger.info("Starting reviews dataset cleaning...")
    logger.info("Reviews dataset cleaning completed.")
    # Step 1: 
    reviews.drop(columns=["review_id", "review_comment_title","review_creation_date", "review_answer_timestamp"], inplace=True)

    return reviews

def clean_products_dataset(products: pd.DataFrame) -> pd.DataFrame:
    """
    Clean the olist_products dataset with the following steps:
    1. Drops NaN cells
    2. Rename wrongly spelled column: lenght to length
    3. Change datatype to int

    Args:
        products (pd.DataFrame): Raw products data

    Returns:
        pd.DataFrame: Cleaned products data
    """
    logger.info("Starting products dataset cleaning...")
    # Step 1: 
    products["product_category_name"] = products["product_category_name"].fillna("unknown")
    products["product_name_lenght"] = products["product_name_lenght"].fillna(0).astype(float)
    products["product_description_lenght"] = products["product_description_lenght"].fillna(0).astype(float)
    products["product_photos_qty"] = products["product_photos_qty"].fillna(0).astype(float)
    logger.info(f"Products shape after filling NaNs: {products.shape}")

    # Step 2:
    products = products.dropna(subset=["product_weight_g", "product_length_cm", "product_height_cm", "product_width_cm"]).reset_index(drop=True)
    logger.info(f"Products shape after dropping NaNs in essential columns: {products.shape}")
    
    # Step 3: 
    products.rename(columns={"product_name_lenght": "product_name_length"}, inplace=True)
    products.rename(columns={"product_description_lenght": "product_description_length"}, inplace=True)
    logger.info("Renamed columns: 'product_name_lenght' to 'product_name_length' and 'product_description_lenght' to 'product_description_length'.")
    
    # Step 4:
    products = products.astype({
    "product_name_length": int,
    "product_description_length": int,
    "product_photos_qty": int
    })
    logger.info("Changed data types for product_name_length, product_description_length, and product_photos_qty to int.")
    logger.info("Products dataset cleaning completed.")

    return products

def clean_sellers_dataset(sellers: pd.DataFrame) -> pd.DataFrame:
    """
    Clean the olist_sellers dataset with the following steps:
    1. Convert zip code prefix to string and remove whitespace
    2. Normalize city names: lowercase, stripped, and accent-removed
    3. Standardize state codes: uppercase and stripped

    Args:
        sellers (pd.DataFrame): Raw sellers data

    Returns:
        pd.DataFrame: Cleaned sellers data
    """
    logger.info("Starting sellers dataset cleaning...")
    logger.info("Sellers dataset cleaning completed.")
    return clean_location_columns(sellers, "seller_zip_code_prefix", "seller_city", "seller_state")

def generate_mega_id_labels(
    clean_orders: pd.DataFrame,
    clean_items: pd.DataFrame,
    clean_customers: pd.DataFrame,
) -> pd.DataFrame:
    """
    Creates a centralized dataset keyed by order_id with essential columns and repeat buyer labels.

    Args:
        clean_orders (pd.DataFrame): Cleaned orders dataset.
        clean_items (pd.DataFrame): Cleaned order items dataset.
        clean_customers (pd.DataFrame): Cleaned customers dataset.

    Returns:
        pd.DataFrame: Dataset with one row per order_id and repeat buyer labels.
    """
    logger.info("Starting mega ID labels generation...")
    # Step 1: Merge orders with customer info
    orders_customers = pd.merge(
        clean_orders[["order_id", "customer_id"]],
        clean_customers[["customer_id", "customer_unique_id"]],
        on="customer_id",
        how="left"
    )
    logger.info(f"Merged orders with customers: {orders_customers.shape}")

    # Step 2: Count number of unique orders per customer_unique_id
    repeat_flags = (
        orders_customers.groupby("customer_unique_id")
        .agg(num_orders=("order_id", "nunique"))
        .reset_index()
    )
    repeat_flags["is_repeat_buyer"] = repeat_flags["num_orders"] > 1
    logger.info(f"Calculated repeat flags: {repeat_flags.shape}")
    
    # Step 3: Merge repeat flag back to each order row
    labeled_orders = pd.merge(
        orders_customers,
        repeat_flags,
        on="customer_unique_id",
        how="left"
    )
    logger.info(f"Merged repeat flags back to orders: {labeled_orders.shape}")

    # Step 4: Add product_id and seller_id from items (each order_id appears once, use first item)
    item_refs = clean_items[["order_id", "product_id", "seller_id"]].drop_duplicates("order_id")
    logger.info(f"Extracted item references: {item_refs.shape}")
    
    # Final merge
    mega = pd.merge(
        labeled_orders,
        item_refs,
        on="order_id",
        how="left"
    )
    # Drop nulls 
    mega = mega.dropna()
    logger.info(f"Final mega dataset shape: {mega.shape}")
    logger.info("Mega ID labels generation completed.")

    return mega[
        [
            "order_id",
            "customer_id",
            "customer_unique_id",
            "product_id",
            "seller_id",
            "num_orders",
            "is_repeat_buyer"
        ]
    ]
