"""
This is a boilerplate pipeline 'feature_engineering'
generated using Kedro 0.19.12
"""
from haversine import haversine, Unit
import pandas as pd

def high_density_customer_flag(clean_customers: pd.DataFrame) -> pd.DataFrame:
    """
    Flags customers from top 5 most frequent cities as high-density areas.
    """
    top_cities = clean_customers["customer_city"].value_counts().nlargest(5).index.tolist()
    clean_customers["high_density_customer_area"] = clean_customers["customer_city"].isin(top_cities).astype(int)
    return clean_customers

def compute_seller_buyer_distance(
    clean_items: pd.DataFrame,
    clean_orders: pd.DataFrame,
    clean_customers: pd.DataFrame,
    clean_sellers: pd.DataFrame,
    clean_geolocation: pd.DataFrame,
    mega_id_labels: pd.DataFrame,
) -> pd.DataFrame:
    """Compute distance between customer and seller for each order and merge into the main dataset."""

    # Step 1: Get customer coordinates
    customers_loc = pd.merge(
        clean_customers,
        clean_geolocation,
        left_on="customer_zip_code_prefix",
        right_on="geolocation_zip_code_prefix",
        how="left"
    ).rename(columns={
        "geolocation_lat": "customer_lat",
        "geolocation_lng": "customer_lng"
    })[["customer_id", "customer_lat", "customer_lng", "high_density_customer_area"]].dropna()

    # Step 2: Get seller coordinates
    sellers_loc = pd.merge(
        clean_sellers,
        clean_geolocation,
        left_on="seller_zip_code_prefix",
        right_on="geolocation_zip_code_prefix",
        how="left"
    ).rename(columns={
        "geolocation_lat": "seller_lat",
        "geolocation_lng": "seller_lng"
    })[["seller_id", "seller_lat", "seller_lng"]].dropna()

    # Step 3: Merge orders and items
    order_details = pd.merge(
        clean_items,
        clean_orders,
        on="order_id",
        how="left"
    )

    # Step 4: Merge customer and seller coordinates
    order_details = pd.merge(order_details, customers_loc, on="customer_id", how="left")
    order_details = pd.merge(order_details, sellers_loc, on="seller_id", how="left")

    # Step 5: Drop rows with missing coordinates
    order_details = order_details.dropna(
        subset=["customer_lat", "customer_lng", "seller_lat", "seller_lng"]
    )

    # Step 6: Compute distance using Haversine in a vectorized manner
    customer_coords = order_details[["customer_lat", "customer_lng"]].to_numpy()
    seller_coords = order_details[["seller_lat", "seller_lng"]].to_numpy()
    order_details["distance_km"] = [
        haversine(cust, sell, unit=Unit.KILOMETERS)
        for cust, sell in zip(customer_coords, seller_coords)
    ]
    # Step 7: Reduce to 1 row per order
    order_distances = order_details.drop_duplicates("order_id")[["order_id", "distance_km", "high_density_customer_area"]]

    # Step 8: Merge with base dataset (mega_id_labels)
    df = pd.merge(
        mega_id_labels,
        order_distances,
        on="order_id",
        how="left"
    )
    df = df.dropna(subset=["distance_km"])

    return df


def calculate_seller_repeat_buyer_rate(order_distances: pd.DataFrame) -> pd.DataFrame:
    """
    Calculates and adds a seller-level repeat buyer rate feature.

    Args:
        df (pd.DataFrame): DataFrame containing 'seller_id', 'customer_unique_id', and 'is_repeat_buyer'

    Returns:
        pd.DataFrame: Original DataFrame with added 'seller_repeat_buyer_rate' column.
    """
    # Step 1: Count repeat buyers per seller
    repeat_buyers = (
        order_distances[order_distances["is_repeat_buyer"] == True]
        .groupby("seller_id")["customer_unique_id"]
        .nunique()
        .reset_index()
        .rename(columns={"customer_unique_id": "num_repeat_buyers"})
    )

    # Step 2: Count total unique buyers per seller
    total_buyers = (
        order_distances.groupby("seller_id")["customer_unique_id"]
        .nunique()
        .reset_index()
        .rename(columns={"customer_unique_id": "num_unique_buyers"})
    )

    # Step 3: Merge & calculate repeat buyer rate
    seller_repeat_stats = pd.merge(total_buyers, repeat_buyers, on="seller_id", how="left")
    seller_repeat_stats["num_repeat_buyers"] = seller_repeat_stats["num_repeat_buyers"].fillna(0)
    seller_repeat_stats["seller_repeat_buyer_rate"] = (
        seller_repeat_stats["num_repeat_buyers"] / seller_repeat_stats["num_unique_buyers"]
    )

    # Step 4: Merge this back into main dataset
    distance_seller_stats = pd.merge(order_distances, seller_repeat_stats[["seller_id", "seller_repeat_buyer_rate"]], on="seller_id", how="left")

    return distance_seller_stats