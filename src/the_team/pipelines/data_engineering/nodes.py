"""
This is a boilerplate pipeline 'data_engineering'
generated using Kedro 0.19.12
"""
import pandas as pd
from datetime import datetime
import numpy as np

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

def clean_customers_dataset(customers: pd.DataFrame) -> pd.DataFrame:
    """
    Clean the olist_customers dataset with the following steps:
    1. 
    2. 
    3. 

    Args:
        customers (pd.DataFrame): Raw customers data

    Returns:
        pd.DataFrame: Cleaned customers data
    """
    # Step 1: 

    # Step 2: 
    
    # Step 3: 
   

    return customers

def clean_geolocation_dataset(geolocation: pd.DataFrame) -> pd.DataFrame:
    """
    Clean the olist_geolocation dataset with the following steps:
    1. 
    2. 
    3. 

    Args:
        geolocation (pd.DataFrame): Raw geolocation data

    Returns:
        pd.DataFrame: Cleaned geolocation data
    """
    # Step 1: 

    # Step 2: 
    
    # Step 3: 
   

    return geolocation

def clean_payments_dataset(payments: pd.DataFrame) -> pd.DataFrame:
    """
    Clean the olist_payments dataset with the following steps:
    1. 
    2. 
    3. 

    Args:
        payments (pd.DataFrame): Raw payments data

    Returns:
        pd.DataFrame: Cleaned payments data
    """
    # Step 1: 

    # Step 2: 
    
    # Step 3: 
   

    return payments

def clean_reviews_dataset(reviews: pd.DataFrame) -> pd.DataFrame:
    """
    Clean the olist_reviews dataset with the following steps:
    1. 
    2. 
    3. 

    Args:
        reviews (pd.DataFrame): Raw reviews data

    Returns:
        pd.DataFrame: Cleaned reviews data
    """
    # Step 1: 

    # Step 2: 
    
    # Step 3: 
   

    return reviews

def clean_products_dataset(products: pd.DataFrame) -> pd.DataFrame:
    """
    Clean the olist_products dataset with the following steps:
    1. 
    2. 
    3. 

    Args:
        products (pd.DataFrame): Raw products data

    Returns:
        pd.DataFrame: Cleaned products data
    """
    # Step 1: 

    # Step 2: 
    
    # Step 3: 
   

    return products

def clean_sellers_dataset(sellers: pd.DataFrame) -> pd.DataFrame:
    """
    Clean the olist_sellers dataset with the following steps:
    1. 
    2. 
    3. 

    Args:
        sellers (pd.DataFrame): Raw sellers data

    Returns:
        pd.DataFrame: Cleaned sellers data
    """
    # Step 1: 

    # Step 2: 
    
    # Step 3: 
   

    return sellers