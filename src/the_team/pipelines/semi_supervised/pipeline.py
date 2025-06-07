"""
This is a boilerplate pipeline 'semi_supervised'
generated using Kedro 0.19.12
"""

from kedro.pipeline import node, Pipeline, pipeline  # noqa
from .nodes import define_weak_positive_criteria, generate_pseudo_labels, retrain_model_with_pseudo_labels, analyze_ssl_bias

def create_pipeline(**kwargs) -> Pipeline:
    return Pipeline([
        node(
            func=define_weak_positive_criteria,
            inputs="model_inputs",
            outputs="weak_positive_mask",
            name="define_weak_positive_criteria_node"
        ),
        node(
            func=generate_pseudo_labels,
            inputs=[
                "best_logistic_model",
                "model_inputs",
                "weak_positive_mask",
                "train_columns"
            ],
            outputs="pseudo_labeled_data",
            name="generate_pseudo_labels_node"
        ),
        node(
            func=retrain_model_with_pseudo_labels,
            inputs="pseudo_labeled_data",
            outputs=[
                "trained_ssl_logistic_model",
                "ssl_logistic_model_metrics",
                "ssl_logistic_predictions_test",
                "ssl_logistic_top_10_predictions"
            ],
            name="retrain_model_with_pseudo_labels_node",
        ),
        node(
            func=analyze_ssl_bias,
            inputs="pseudo_labeled_data",
            outputs="ssl_bias_report",
            name="analyze_ssl_bias_node",
        ),
    ])
