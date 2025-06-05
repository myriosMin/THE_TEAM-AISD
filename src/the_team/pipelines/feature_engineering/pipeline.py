"""
This is a boilerplate pipeline 'feature_engineering'
generated using Kedro 0.19.12
"""

from kedro.pipeline import node, Pipeline, pipeline  # noqa
from .nodes import *

def create_pipeline(**kwargs) -> Pipeline:
    return pipeline([
        node(
            func=high_density_customer_flag,
            inputs="clean_customers",
            outputs="tagged_customers",
            name="add_high_density_customer_flag_node"
        ),
        node(
            func=compute_seller_buyer_distance,
            inputs=["clean_items", "clean_orders", "tagged_customers", "clean_sellers", "clean_geolocation", "mega_id_labels"],
            outputs="order_distances",
            name="compute_distance_node"
        ),
        node(
            func=calculate_seller_repeat_buyer_rate,
            inputs="order_distances",
            outputs="distance_seller_stats",
            name="calculate_seller_repeat_buyer_rate_node"
        ),
        node(
            func=add_verified_rating,
            inputs=["clean_reviews","params:feature_engineering.run_sentiment"],
            outputs="review_features",
            name="create_review_features_node"
        ),
        node(
            func=translate_product_categories,
            inputs=["clean_products", "translation"],
            outputs="product_features",
            name="create_product_features_node"
        ),
        node(
            func=create_transaction_features,
            inputs=["clean_orders","clean_items","clean_payments","mega_id_labels"],
            outputs="transaction_features",
            name="create_transaction_features_node"
        ),
        node(
            func=merge_model_inputs,
            inputs=["transaction_features", "product_features", "review_features", "distance_seller_stats"],
            outputs="model_inputs",
            name="merge_model_inputs_node"
        ),
    ])
