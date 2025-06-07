"""
This is a boilerplate pipeline 'modelling'
generated using Kedro 0.19.12
"""

from kedro.pipeline import node, Pipeline, pipeline  # noqa
from .nodes import *

def create_pipeline(**kwargs) -> Pipeline:
    return pipeline([
            node(
            func=train_random_forest_model,
            inputs=[
                "X_train_transformed",
                "X_test_transformed",
                "y_train",
                "y_test",
                "params:general.random_state",
                "params:random_forest.n_estimators",
                "params:random_forest.max_depth",
                "params:random_forest.class_weight",
                "params:skip.skip_random_forest"
            ],
            outputs=[
                "trained_random_forest_model",
                "random_forest_model_metrics",
                "random_forest_predictions_test",
                "random_forest_top_10_predictions"
            ],
            name="train_random_forest_model_node"
        ),
        node(
            func=train_logistic_model,
            inputs=[
                "X_train_transformed",
                "X_test_transformed",
                "y_train",
                "y_test",
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
                "X_train_transformed",
                "X_test_transformed",
                "y_train",
                "y_test",
                "params:general.random_state",
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
        ),
        node(
            func=train_xgboost_model,
            inputs=[
                "X_train_transformed",
                "X_test_transformed",
                "y_train",
                "y_test",
                "params:general.random_state",
                "params:xgboost.n_estimators",
                "params:xgboost.learning_rate",
                "params:xgboost.max_depth",
                "params:xgboost.scale_pos_weight",
                "params:skip.skip_xgboost"
            ],
            outputs=[
                "trained_xgboost_model",
                "xgboost_model_metrics",
                "xgboost_model_predictions_test",
                "xgboost_model_top_10_predictions"
            ],
            name="train_xgboost"
        ),
    ])

