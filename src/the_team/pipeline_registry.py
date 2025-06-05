"""Project pipelines."""

from kedro.pipeline import Pipeline
from the_team.pipelines import data_engineering, feature_engineering, modelling, tuning

def register_pipelines() -> dict[str, Pipeline]:
    """Manually register the project's pipelines."""
    de_pipeline = data_engineering.create_pipeline()
    fe_pipeline = feature_engineering.create_pipeline()
    m_pipeline = modelling.create_pipeline()
    tu_pipeline = tuning.create_pipeline()

    return {
        "__default__": de_pipeline + fe_pipeline + m_pipeline,
        "data_engineering": de_pipeline,
        "feature_engineering": fe_pipeline,
        "modelling": m_pipeline,
        "tuning": tu_pipeline,
    }