"""Project pipelines."""

from kedro.pipeline import Pipeline
from the_team.pipelines import data_engineering, feature_engineering, preprocessing, modelling, tuning, semi_supervised

def register_pipelines() -> dict[str, Pipeline]:
    """Manually register the project's pipelines."""
    de_pipeline = data_engineering.create_pipeline()
    fe_pipeline = feature_engineering.create_pipeline()
    pre_pipeline = preprocessing.create_pipeline()
    m_pipeline = modelling.create_pipeline()
    tu_pipeline = tuning.create_pipeline()
    ss_pipeline = semi_supervised.create_pipeline()

    return {
        "__default__": de_pipeline + fe_pipeline + pre_pipeline + m_pipeline + tu_pipeline + ss_pipeline,
        "data_engineering": de_pipeline,
        "feature_engineering": fe_pipeline,
        "preprocessing": pre_pipeline,
        "modelling": m_pipeline,
        "tuning": tu_pipeline,
        "semi_supervised": ss_pipeline,
        "__preprocess_train__": pre_pipeline + m_pipeline,
    }