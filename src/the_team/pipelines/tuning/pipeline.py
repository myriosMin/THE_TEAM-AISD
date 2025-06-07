"""
This is a boilerplate pipeline 'tuning'
generated using Kedro 0.19.12
"""

from kedro.pipeline import node, Pipeline, pipeline  # noqa
from .nodes import tune_logistic_model

def create_pipeline(**kwargs) -> Pipeline:
    return pipeline([
        node(
            func=tune_logistic_model,
            inputs=[
                "X_train_transformed",
                "X_test_transformed",
                "y_train",
                "y_test",
                "params:logistic_tuning",
            ],
            outputs=[
                "best_logistic_model",
                "logistic_model_tuning_metrics",
                "logistic_predictions_test_tuning",
                "logistic_top_10_predictions_tuning"
            ],
            name="tune_logistic_model_node"
        ),
    ])
