"""Preprocessing pipeline definition."""

from kedro.pipeline import Pipeline, node
from .nodes import (
    clean_data,
    filter_data,
    normalize_value,
    validate_data,
    normalize_missing_values,
)


def create_pipeline(**kwargs) -> Pipeline:
    """Create the preprocessing pipeline."""
    return Pipeline(
        [
            node(
                func=clean_data,
                inputs=["standardized_data", "params:preprocess.clean"],
                outputs="cleaned_data",
                name="clean_data",
                tags="preprocess",
            ),
            node(
                func=filter_data,
                inputs=["cleaned_data", "params:preprocess.filter"],
                outputs="filtered_data",
                name="filter_data",
                tags="preprocess",
            ),
            node(
                func=normalize_value,
                inputs=["filtered_data", "params:preprocess.code_mappings"],
                outputs="normalized_data",
                name="normalize_value",
                tags="preprocess",
            ),
            node(
                func=validate_data,
                inputs=["normalized_data", "parameters"],
                outputs="validated_data_raw",
                name="validate_data",
                tags="preprocess",
            ),
            node(
                func=normalize_missing_values,
                inputs=["validated_data_raw", "params:preprocess.missing_values"],
                outputs="validated_data",
                name="normalize_missing_values",
                tags="preprocess",
            ),
        ]
    )
