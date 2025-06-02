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
            outputs="reviews_with_verified_rating",
            name="verified_rating_node"
        )
    ])
