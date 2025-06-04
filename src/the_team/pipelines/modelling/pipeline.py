"""
This is a boilerplate pipeline 'modelling'
generated using Kedro 0.19.12
"""

from kedro.pipeline import node, Pipeline, pipeline  # noqa
from .nodes import *

def create_pipeline(**kwargs) -> Pipeline:
    return pipeline([])
def create_pipeline(**kwargs) -> Pipeline:
    return pipeline([
        node(
            func=train_logistic_model,
            inputs=[
                "model_inputs",
                "params:split.test_size",
                "params:split.stratify",
                "params:split.random_state",
                "params:logistic_regression.C",
                "params:logistic_regression.max_iter",
                "params:logistic_regression.solver",
                "params:logistic_regression.class_weight",
                "params:skip.skip_logistic"
            ],
            outputs=[
                "trained_logistic_model",
                "logistic_model_metrics",
                "logistic_predictions_test",
                "logistic_top_10_predictions"
            ],
            name="train_logistic_model_node"
        ),
        node(
            func=train_lightgbm_model,
            inputs=[
                "model_inputs",
                "params:split.test_size",
                "params:split.stratify",
                "params:split.random_state",
                "params:lightgbm.n_estimators",
                "params:lightgbm.learning_rate",
                "params:lightgbm.max_depth",
                "params:lightgbm.scale_pos_weight",
                "params:skip.skip_lightgbm"
            ],
            outputs=[
                "trained_lightgbm_model",
                "lightgbm_model_metrics",
                "lightgbm_predictions_test",
                "lightgbm_top_10_predictions"
            ],
            name="train_lightgbm_model_node"
        )
    ])