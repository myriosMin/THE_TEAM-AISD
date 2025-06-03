"""
This is a boilerplate pipeline 'feature_engineering'
generated using Kedro 0.19.12
"""

from kedro.pipeline import node, Pipeline, pipeline  # noqa
from .nodes import *

def create_pipeline(**kwargs) -> Pipeline:
    return Pipeline([
        node(
            func=add_verified_rating,
            inputs="clean_reviews",
            outputs="review_features",
            name="create_review_features_node"
        ),
        node(
            func=translate_product_categories,
            inputs="clean_products",
            outputs="product_features",
            name="create_product_features_node"
        ),
        node(
            func=create_transaction_features,
            inputs=[
                "clean_orders",
                "clean_items",
                "clean_payments",
                "mega_id_labels"
            ],
            outputs="transaction_features",
            name="create_transaction_features_node"
        ),
    ])
