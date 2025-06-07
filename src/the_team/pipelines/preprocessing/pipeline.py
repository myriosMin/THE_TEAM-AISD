"""
This is a boilerplate pipeline 'preprocessing'
generated using Kedro 0.19.12
"""

from kedro.pipeline import node, Pipeline
from .nodes import split_data, build_preprocessor, apply_preprocessor_train, apply_preprocessor_test, extract_feature_names

def create_pipeline(**kwargs) -> Pipeline:
    return Pipeline([
        node(
            func=split_data,
            inputs=["model_inputs", "params:test_size", "params:stratify", "params:random_state"],
            outputs=["X_train", "X_test", "y_train", "y_test"],
            name="split_data_node",
        ),
        node(
            func=build_preprocessor,
            inputs=[
                "params:numeric_features",
                "params:payment_features",
                "params:cat_features"
            ],
            outputs="preprocessor_unfitted",
            name="build_preprocessor_node",
        ),
        node(
            func=apply_preprocessor_train,
            inputs=["preprocessor_unfitted", "X_train"],
            outputs=["preprocessor", "X_train_transformed"],
            name="apply_preprocessor_train_node",
        ),
        node(
            func=apply_preprocessor_test,
            inputs=["preprocessor", "X_test"],
            outputs="X_test_transformed",
            name="apply_preprocessor_test_node",
        ),
        node(
            func=extract_feature_names,
            inputs="preprocessor",
            outputs="train_columns",
            name="extract_train_columns_node",
        ),
    ])