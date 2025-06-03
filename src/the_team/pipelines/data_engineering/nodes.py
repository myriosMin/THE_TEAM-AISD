"""
This is a boilerplate pipeline 'data_engineering'
generated using Kedro 0.19.12
"""
import pandas as pd
from datetime import datetime
import numpy as np
from unidecode import unidecode # type: ignore

def clean_orders_dataset(orders: pd.DataFrame) -> pd.DataFrame:
    """
    Clean the olist_orders_dataset with the following steps:
    1. Convert date columns to datetime, and filter to 2016–2018
    2. Drop 'order_delivered_carrier_date'
    3. Drop rows with unwanted order statuses
    4. Impute missing 'order_delivered_customer_date' using median duration between expected and actual delivery
    5. Drop rows with null 'order_approved_at'

    Args:
        orders (pd.DataFrame): Raw orders dataset

    Returns:
        pd.DataFrame: Cleaned and filtered orders
    """
    # Step 1: Convert to datetime
    date_cols = [
        "order_purchase_timestamp",
        "order_approved_at",
        "order_delivered_carrier_date",
        "order_delivered_customer_date",
        "order_estimated_delivery_date"
    ]
    for col in date_cols:
        orders[col] = pd.to_datetime(orders[col], errors="coerce")

    # Filter to orders between 2016-01-01 and 2018-12-31
    orders = orders[
        (orders["order_purchase_timestamp"] >= "2016-01-01") &
        (orders["order_purchase_timestamp"] <= "2018-12-31")
    ]

    # Step 2: Drop 'order_delivered_carrier_date'
    orders = orders.drop(columns=["order_delivered_carrier_date"])

    # Step 3: Remove unwanted order statuses
    unwanted_statuses = ["canceled", "created", "invoiced", "unavailable"]
    orders = orders[~orders["order_status"].isin(unwanted_statuses)]

    # Step 4: Fill nulls in 'order_delivered_customer_date'
    # Calculate median difference (actual - estimated)
    delivery_diff = (
        orders["order_estimated_delivery_date"] - orders["order_delivered_customer_date"]
    )
    median_offset = delivery_diff.dropna().median()
    orders["order_delivered_customer_date"] = orders["order_delivered_customer_date"].fillna(
        orders["order_estimated_delivery_date"] - median_offset
    )

    # Step 5: Drop nulls in 'order_approved_at'
    orders = orders.dropna(subset=["order_approved_at"])

    return orders

def clean_items_dataset(items: pd.DataFrame) -> pd.DataFrame:
    """
    Clean the olist_order_items dataset with the following steps:
    1. Convert shipping_limit_date to datetime
    2. Filter to records between 2016–2018
    3. Cap outliers in price and freight_value (99.9th percentile)

    Args:
        items (pd.DataFrame): Raw items data

    Returns:
        pd.DataFrame: Cleaned items data
    """
    # Step 1: Convert shipping_limit_date to datetime
    items["shipping_limit_date"] = pd.to_datetime(items["shipping_limit_date"], errors="coerce")

    # Step 2: Filter by year from 2016 to 2018
    items = items[
        (items["shipping_limit_date"] >= "2016-01-01") &
        (items["shipping_limit_date"] <= "2018-12-31")
    ]

    # Step 3: Cap outliers at 99.9th percentile
    for col in ["price", "freight_value"]:
        upper = np.percentile(items[col], 99.9)
        items[col] = np.where(items[col] > upper, upper, items[col])

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
     # Step 1:  Convert zip prefix to string and remove whitespace
    df[zip_col] = df[zip_col].astype(str).str.strip()
    assert df[zip_col].dtype == "object", f"{zip_col} should be string"
    
    # Step 2:  Normalize city names
    df[city_col] = df[city_col].str.lower().str.strip().apply(unidecode)
    
    # Step 3:  Standardize state codes
    df[state_col] = df[state_col].str.upper().str.strip()
    
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
    return customers

def clean_geolocation_dataset(geolocation: pd.DataFrame) -> pd.DataFrame:
    """
    Clean the olist_geolocation dataset with the following steps:
    1. Removing outliers in latitude and longitude
    2. Convert zip code prefix to string and remove whitespace
    3. Normalize city names: lowercase, stripped, and accent-removed
    4. Standardize state codes: uppercase and stripped

    Args:
        geolocation (pd.DataFrame): Raw geolocation data

    Returns:
        pd.DataFrame: Cleaned geolocation data
    """
    # Step 1: Remove outliers in latitude and longitude
    valid_lat_range = (-33.75116944, 5.27438888)
    valid_lng_range = (-73.98283055, -34.79314722)

    geolocation = geolocation[
    (geolocation["geolocation_lat"].between(*valid_lat_range)) &
    (geolocation["geolocation_lng"].between(*valid_lng_range))
]

    return clean_location_columns(geolocation, "geolocation_zip_code_prefix", "geolocation_city", "geolocation_state")

def clean_payments_dataset(payments: pd.DataFrame) -> pd.DataFrame:
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

    # Step 1: Drop 'not_defined' payment types
    payments = payments[payments["payment_type"] != "not_defined"]

    # Step 2: Clip 'payment_installments' to 99th percentile
    upper_clip = payments["payment_installments"].quantile(0.99)
    payments["payment_installments"] = payments["payment_installments"].clip(upper=upper_clip)

    # Step 3: Drop irrelevant columns
    payments.drop(columns=["payment_sequential"], inplace=True)

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
    # Step 1: 
    reviews.drop(columns=["review_id", "review_comment_title","review_creation_date", "review_answer_timestamp"], inplace=True)

    return reviews

def clean_products_dataset(products: pd.DataFrame) -> pd.DataFrame:
    """
    Clean the olist_products dataset with the following steps:
    1. Drops NaN cells
    2. Change datatype to int

    Args:
        products (pd.DataFrame): Raw products data

    Returns:
        pd.DataFrame: Cleaned products data
    """
    # Step 1: 
    products.dropna(inplace=True)

    # Step 2: 
    products = products.astype({
    "product_name_length": int,
    "product_description_length": int,
    "product_photos_qty": int
    })

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
    # Step 1: 

    # Step 2: 
    
    # Step 3: 
   

    return sellers

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

    # Step 1: Merge orders with customer info
    orders_customers = pd.merge(
        clean_orders[["order_id", "customer_id"]],
        clean_customers[["customer_id", "customer_unique_id"]],
        on="customer_id",
        how="left"
    )

    # Step 2: Count number of unique orders per customer_unique_id
    repeat_flags = (
        orders_customers.groupby("customer_unique_id")
        .agg(num_orders=("order_id", "nunique"))
        .reset_index()
    )
    repeat_flags["is_repeat_buyer"] = repeat_flags["num_orders"] > 1

    # Step 3: Merge repeat flag back to each order row
    labeled_orders = pd.merge(
        orders_customers,
        repeat_flags,
        on="customer_unique_id",
        how="left"
    )

    # Step 4: Add product_id and seller_id from items (each order_id appears once, use first item)
    item_refs = clean_items[["order_id", "product_id", "seller_id"]].drop_duplicates("order_id")

    # Final merge
    mega = pd.merge(
        labeled_orders,
        item_refs,
        on="order_id",
        how="left"
    )
    # Drop nulls 
    mega = mega.dropna()

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