"""
This is a boilerplate pipeline 'feature_engineering'
generated using Kedro 0.19.12
"""
import pandas as pd
from haversine import haversine, Unit
import subprocess
import sys
import logging
logger = logging.getLogger(__name__)

def create_transaction_features(
    clean_orders: pd.DataFrame,
    clean_items: pd.DataFrame,
    clean_payments: pd.DataFrame,
    mega_id_labels: pd.DataFrame
) -> pd.DataFrame:
    """
    Create transaction-level features at the order-item level.
    Each row in the generated df corresponds to a single item in an order.

    Returns:
        pd.DataFrame: Transaction-level features per item.
    """
    logger.info("Creating transaction features at order-item level...")
    # Step 1: Start with item-level
    df = clean_items.copy()

    # Step 2: Join with orders
    df = df.merge(
        clean_orders[
            ["order_id", "order_approved_at", "order_delivered_customer_date", "order_estimated_delivery_date"]
        ],
        on="order_id",
        how="left"
    )

    # Step 3: Add payments (grouped by order, multi-hot coded + percentage breakdown)
    payment_encoded = (
        clean_payments
        .groupby(["order_id", "payment_type"])["payment_value"]
        .sum()
        .unstack(fill_value=0)
        .reset_index()
    )

    # Step 4: Merge repeat buyer info
    df = df.merge(
        mega_id_labels[["order_id", "customer_id", "customer_unique_id", "is_repeat_buyer"]],
        on="order_id",
        how="left"
    )

    # Feature 1: Delivery duration (expected vs actual)
    df["deli_duration_exp"] = (df["order_delivered_customer_date"] - df["order_estimated_delivery_date"]).dt.days

    # Feature 2: Delivery duration from approval
    df["deli_duration_paid"] = (df["order_delivered_customer_date"] - df["order_approved_at"]).dt.days

    # Feature 3: Delivery cost
    df["deli_cost"] = df["freight_value"]

    # Feature 4: Free delivery flag
    df["free_delivery"] = df["freight_value"] == 0.0

    # Feature 5: Bulk order flag (same product repeated in an order)
    product_counts = (
        clean_items.groupby(["order_id", "product_id"])
        .size()
        .reset_index(name="product_qty")
    )
    product_counts["is_bulk"] = product_counts["product_qty"] > 1
    df = df.merge(product_counts[["order_id", "product_id", "is_bulk"]], on=["order_id", "product_id"], how="left")
    df["is_bulk"] = df["is_bulk"].fillna(False)

    # Feature 6: Item price
    df["item_price"] = df["price"]

    # Feature 7: High price flag using IQR
    q1 = clean_items["price"].quantile(0.25)
    q3 = clean_items["price"].quantile(0.75)
    iqr = q3 - q1
    high_price_threshold = q3 + 1.5 * iqr
    df["high_price"] = df["price"] > high_price_threshold

    # Feature 8: Discount flag (item price < avg product price)
    avg_product_price = (
        clean_items
        .groupby("product_id")["price"]
        .mean()
        .reset_index()
        .rename(columns={"price": "avg_price"})
    )

    # Merge with main df
    df = df.merge(avg_product_price, on="product_id", how="left")
    df["discount"] = df["price"] < df["avg_price"]
    df.drop(columns=["avg_price"], inplace=True)

    # Feature 9: Calculate total spent per order
    payment_encoded["total_spent"] = payment_encoded.drop(columns="order_id").sum(axis=1)

    # Feature 10: Generate payment types and price ratios per order
    payment_types = [col for col in payment_encoded.columns if col not in ["order_id", "total_spent"]]
    payment_encoded[payment_types] = payment_encoded[payment_types].div(payment_encoded["total_spent"], axis=0).fillna(0)
    df = df.merge(payment_encoded, on="order_id", how="left")
        
    # Feature 11: Installment count; sum installments per order
    installment_info = (
        clean_payments
        .groupby("order_id")["payment_installments"]
        .sum()
        .reset_index()
    )

    df = df.merge(installment_info, on="order_id", how="left")
    df["installment"] = df["payment_installments"].fillna(0).astype(int)
    df.drop(columns=["payment_installments"], inplace=True)

    # Final selection
    base_cols = [
        "order_id", "product_id", "seller_id", "customer_id", "customer_unique_id",
        "deli_duration_exp", "deli_duration_paid", "deli_cost", "free_delivery",
        "item_price", "high_price", "discount", "is_bulk", "is_repeat_buyer", "installment"
    ]
    
    # order_id, product_id, customer_id, customer_unique_id will be dropped later before training
    # we keep them here for reference and potential future use in joining with other datasets

    # Automatically include all normalized payment ratio columns
    ratio_cols = [col for col in payment_encoded.columns if col not in ["order_id"]]

    logger.info(f"Transaction features created with {len(df)} rows and {len(base_cols + ratio_cols)} columns.")
    
    # Return only non-null rows
    return df[base_cols + ratio_cols].dropna()

#  Product/Review Related Nodes

# Cache dictionary to avoid redundant predictions
sentiment_cache = {}

def map_sentiment(sentiment_label: str) -> str:
    return {
        "POS": "positive",
        "NEU": "neutral",
        "NEG": "negative"
    }.get(sentiment_label, "neutral")

# For falsely identified positives as neutral
def heuristic_sentiment(text: str, model_sentiment: str) -> str:
    text_lower = text.strip().lower()
    words = text_lower.split()

    positive_keywords = {
        "ótimo", "excelente", "bom", "recomendo", "confiável",
        "satisfeito", "tranquilo", "correto", "tudo certo", "tudo ok"
    }

    if model_sentiment == "neutral":
        for word in positive_keywords:
            if word in text_lower:
                return "positive"
        if len(words) <= 5 and any(p in text_lower for p in ["tudo certo", "tudo ok", "site confiável", "cumpriu", "conforme"]):
            return "positive"

    return model_sentiment

def analyze_sentiment(text: str) -> str:
    global analyzer
    if not text.strip():
        return "No comment"
    if text in sentiment_cache:
        return sentiment_cache[text]

    result = analyzer.predict(text)
    sentiment = heuristic_sentiment(text, map_sentiment(result.output))  # type: ignore
    sentiment_cache[text] = sentiment
    return sentiment

def add_verified_rating(reviews: pd.DataFrame, run_sentiment: bool = True) -> pd.DataFrame:
    """
    Adds 'sentiment' and 'is_verified' features to the reviews DataFrame using a Portuguese BERT model.

    If run_sentiment=False, skips analysis and removes review_comment_message.

    Args:
        reviews (pd.DataFrame): Input with 'review_comment_message' and 'review_score'
        run_sentiment (bool): Whether to run sentiment analysis

    Returns:
        pd.DataFrame: Processed DataFrame with 'sentiment', 'is_verified' added, and comment column dropped
    """
    logger.info("Adding verified rating to reviews...")
    reviews = reviews.copy()
    reviews["review_comment_message"] = reviews["review_comment_message"].fillna("").astype(str)
    global analyzer # Ensure analyzer is defined globally
    
    if not run_sentiment:
        # Skip sentiment analysis entirely
        reviews.drop(columns=["review_comment_message"], inplace=True)
        return reviews
    
    # Try importing pysentimiento, install it if missing
    try:
        from pysentimiento import create_analyzer
    except ImportError:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "pysentimiento>=0.3.0"])
        from pysentimiento import create_analyzer  # re-import after install

    analyzer = create_analyzer(task="sentiment", lang="pt")  # using a Portuguese model to directly analyze Portuguese
    
    # Running sentiment analysis with caching
    reviews["sentiment"] = reviews["review_comment_message"].apply(analyze_sentiment)

    # Apply matching logic
    def is_verified(row):
        sentiment = row["sentiment"]
        score = row["review_score"]

        if sentiment == "No comment":
            return None

        return (
            (sentiment in ["positive", "neutral"] and score in [4, 5]) or
            (sentiment == "neutral" and score == 3) or
            (sentiment == "negative" and score in [1, 2])
        )

    reviews["is_verified"] = reviews.apply(is_verified, axis=1) # type: ignore

    # Dropping comments column
    reviews.drop(columns=["review_comment_message"], inplace=True)
    
    # Drop rows where 'is_verified' is None
    reviews = reviews[reviews["is_verified"].notna()]
    
    logger.info(f"Processed {len(reviews)} reviews with sentiment and verification status.")
    return reviews

def translate_product_categories(products: pd.DataFrame, translation: pd.DataFrame) -> pd.DataFrame:
    """
    Replaces 'product_category_name' with its English translation using a lookup table (translation dataset).

    Parameters:
        products (pd.DataFrame): Products data with 'product_category_name'.
        translation (pd.DataFrame): Mapping table with Portuguese and English names.

    Returns:
        pd.DataFrame: Same DataFrame with 'product_category_name' replaced in English.
    """
    logger.info("Translating product categories to English...")
    # Create the mapping
    translation_map = translation.set_index("product_category_name")["product_category_name_english"].to_dict()

    # Replace in place
    products["product_category_name"] = products["product_category_name"].map(translation_map)
    logger.info(f"Translated {len(products)} product categories.")
    return products

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
    logger.info("Computing seller-buyer distances...")
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
    logger.info(f"Computed distances for {len(df)} orders.")
    return df


def calculate_seller_repeat_buyer_rate(df: pd.DataFrame) -> pd.DataFrame:
    """
    Calculates and adds a seller-level repeat buyer rate feature.

    Args:
        df (pd.DataFrame): DataFrame containing 'seller_id', 'customer_unique_id', and 'is_repeat_buyer'

    Returns:
        pd.DataFrame: Original DataFrame with added 'seller_repeat_buyer_rate' column.
    """
    logger.info("Calculating seller repeat buyer rates...")
    # Step 1: Count repeat buyers per seller
    repeat_buyers = (
        df[df["is_repeat_buyer"] == True]
        .groupby("seller_id")["customer_unique_id"]
        .nunique()
        .reset_index()
        .rename(columns={"customer_unique_id": "num_repeat_buyers"})
    )

    # Step 2: Count total unique buyers per seller
    total_buyers = (
        df.groupby("seller_id")["customer_unique_id"]
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
    distance_seller_stats = pd.merge(df, seller_repeat_stats[["seller_id", "seller_repeat_buyer_rate"]], on="seller_id", how="left")
    logger.info(f"Calculated repeat buyer rates for {len(distance_seller_stats)} sellers.")
    return distance_seller_stats

def merge_model_inputs(
    transaction_features: pd.DataFrame,
    product_features: pd.DataFrame,
    review_features: pd.DataFrame,
    distance_seller_stats: pd.DataFrame
) -> pd.DataFrame:
    """
    Merge all feature datasets into a single modeling input DataFrame.

    Args:
        transaction_features (pd.DataFrame): Order-item level transaction features
        product_features (pd.DataFrame): Product-level attributes
        review_features (pd.DataFrame): Review scores per order
        distance_seller_stats (pd.DataFrame): Seller-buyer geographic and repeat info

    Returns:
        pd.DataFrame: Merged modeling inputs with selected features
    """
    logger.info("Merging all feature datasets into model inputs...")
    # 1. Merge seller-level distance and repeat rate features
    merged = transaction_features.merge(
        distance_seller_stats.drop(columns=["num_orders", "is_repeat_buyer"]),
        on=["order_id", "product_id", "seller_id", "customer_id", "customer_unique_id"],
        how="left"
    )

    # 2. Merge product-level features
    merged = merged.merge(
        product_features,
        on="product_id",
        how="left"
    )

    # 3. Merge review scores
    merged = merged.merge(
        review_features,
        on="order_id",
        how="left"
    )

    # 4. Final feature selection
    final_cols = [
        "deli_duration_exp", "deli_duration_paid", "deli_cost", "free_delivery",
        "is_bulk", "discount", "item_price", "high_price",
        "credit_card", "voucher", "debit_card", "boleto",
        "total_spent", "installment",
        "distance_km", "high_density_customer_area", "seller_repeat_buyer_rate",
        "review_score",
        "product_category_name", "product_name_length", "product_description_length",
        "product_photos_qty", "product_weight_g", "product_length_cm",
        "product_height_cm", "product_width_cm",
        "is_repeat_buyer",
    ]
    
    # 5. Check if is_verfied exists in review_features; doing so will drop a lot of rows since many reviews do not have any comment to verify
    if "is_verified" in review_features.columns:
        final_cols.append("is_verified")

    # Drop missing and duplicate rows
    model_inputs = merged[final_cols].dropna().drop_duplicates()
    logger.info(f"Final model inputs created with {len(model_inputs)} rows and {len(model_inputs.columns)} columns.")
    return model_inputs
