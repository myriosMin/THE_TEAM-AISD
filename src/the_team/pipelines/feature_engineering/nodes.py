"""
This is a boilerplate pipeline 'feature_engineering'
generated using Kedro 0.19.12
"""
import pandas as pd
from pysentimiento import create_analyzer

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

    # Step 3: Add payments (grouped by order, multi-hot + breakdown)
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

    # Feature 9: Calculate total spent
    payment_encoded["total_spent"] = payment_encoded.drop(columns="order_id").sum(axis=1)

    # Feature 10: Generate payment types and price ratios per order
    payment_types = [col for col in payment_encoded.columns if col not in ["order_id", "total_spent"]]
    payment_encoded[payment_types] = payment_encoded[payment_types].div(payment_encoded["total_spent"], axis=0).fillna(0)
    df = df.merge(payment_encoded, on="order_id", how="left")
        
    # Feature 11: Installment count
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

    # Return only non-null rows
    return df[base_cols + ratio_cols].dropna()

#  Product/Review Related Nodes

# calling model at module level to ensure it is called once per run
analyzer = create_analyzer(task="sentiment", lang="pt") # using a portugese model to directly analyze portugese

# Load the model once
analyzer = create_analyzer(task="sentiment", lang="pt")

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

def analyze_sentiment(text: str) -> str | None:
    if not text.strip():
        return None
    if text in sentiment_cache:
        return sentiment_cache[text]

    result = analyzer.predict(text)
    sentiment = heuristic_sentiment(text, map_sentiment(result.output)) # type: ignore
    sentiment_cache[text] = sentiment
    return sentiment

def add_verified_rating(reviews: pd.DataFrame) -> pd.DataFrame:
    """
    Adds 'sentiment' and 'is_verified' features using a Portuguese BERT model.
    """

    reviews = reviews.copy()
    reviews["review_comment_message"] = reviews["review_comment_message"].fillna("").astype(str)

    # Running sentiment analysis with caching
    reviews["sentiment"] = reviews["review_comment_message"].apply(analyze_sentiment)

    # Apply matching logic
    def is_verified(row):
        sentiment = row["sentiment"]
        score = row["review_score"]
        if sentiment is None:
            return False
        return (
            (sentiment == "positive" and score in [4, 5]) or
            (sentiment == "neutral" and score == 3) or
            (sentiment == "negative" and score in [1, 2])
        )

    reviews["is_verified"] = reviews.apply(is_verified, axis=1)
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
    # Create the mapping
    translation_map = translation.set_index("product_category_name")["product_category_name_english"].to_dict()

    # Replace in place
    products["product_category_name"] = products["product_category_name"].map(translation_map)

    return products
