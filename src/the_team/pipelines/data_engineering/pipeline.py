"""
This is a boilerplate pipeline 'data_engineering'
generated using Kedro 0.19.12
"""

from kedro.pipeline import node, Pipeline, pipeline  # noqa
from .nodes import *

def create_pipeline(**kwargs) -> Pipeline:
    return pipeline([
        node(
            func=clean_orders_dataset,
            inputs=["orders", "params:orders.date_cols", "params:orders.to_drop"],
            outputs="clean_orders",
            name="clean_orders_node"
        ), 
        node(
            func=clean_items_dataset,
            inputs=["items", "params:items.to_drop", "params:items.to_cap", "params:items.upper_cap_value"],
            outputs="clean_items",
            name="clean_items_node"
        ),
        node(
            func=clean_payments_dataset,
            inputs=["payments", "params:payments.drop_payment_type", "params:payments.to_drop"],
            outputs="clean_payments",
            name="clean_payments_node"
        ),
        node(
            func=clean_customers_dataset,
            inputs="customers",
            outputs="clean_customers",
            name="clean_customers_node"
        ),
        node(
            func=clean_geolocation_dataset,
            inputs="geolocation",
            outputs="clean_geolocation",
            name="clean_geolocation_node"
        ),
        node(
            func=clean_sellers_dataset,
            inputs="sellers",
            outputs="clean_sellers",
            name="clean_sellers_node"
        ),
        node(
            func=clean_reviews_dataset,
            inputs="reviews",
            outputs="clean_reviews",
            name="clean_reviews_node"
        ),
        node(
            func=clean_products_dataset,
            inputs="products",
            outputs="clean_products",
            name="clean_products_node"
        ),
        node(
            func=generate_mega_id_labels,
            inputs=["clean_orders", "clean_items", "clean_customers"],
            outputs="mega_id_labels",
            name="generate_mega_id_labels_node"
        ),
    ])