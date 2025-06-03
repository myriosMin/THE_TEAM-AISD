"""
This is a boilerplate pipeline 'feature_engineering'
generated using Kedro 0.19.12
"""
import pandas as pd

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
    df["instalment"] = df["payment_installments"].fillna(0).astype(int)
    df.drop(columns=["payment_installments"], inplace=True)

    # Final selection
    base_cols = [
        "order_id", "product_id", "seller_id", "customer_id", "customer_unique_id",
        "deli_duration_exp", "deli_duration_paid", "deli_cost", "free_delivery",
        "item_price", "high_price", "discount", "is_bulk", "is_repeat_buyer", "instalment"
    ]
    
    # order_id, product_id, customer_id, customer_unique_id will be dropped later before training
    # we keep them here for reference and potential future use in joining with other datasets

    # Automatically include all normalized payment ratio columns
    ratio_cols = [col for col in payment_encoded.columns if col not in ["order_id"]]

    # Return only non-null rows
    return df[base_cols + ratio_cols].dropna()