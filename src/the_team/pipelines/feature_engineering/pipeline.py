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
    ])
