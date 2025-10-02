"""Feature pipeline definition."""

from kedro.pipeline import Pipeline, node
from .nodes import (
    build_features,
    select_features,
)


def create_pipeline(**kwargs) -> Pipeline:
    """Create the feature engineering pipeline."""
    return Pipeline(
        [
            node(
                func=build_features,
                inputs=["validated_data", "params:feature.engineering"],
                outputs="feature_data",
                name="build_features",
                tags="feature",
            ),
            node(
                func=select_features,
                inputs=["feature_data", "params:feature.selection"],
                outputs="base_table",
                name="select_features",
                tags="feature",
            ),
        ]
    )
