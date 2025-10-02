"""Pipeline for data ingestion."""

from kedro.pipeline import Pipeline, node
from .nodes import (
    load_data,
    collect_missing_stats,
    standardize_columns,
    transform_dtype,
    extract_metadata
)


def create_pipeline(**kwargs) -> Pipeline:
    """Create the data ingestion pipeline.

    Returns:
        Pipeline for ingesting and standardizing accounting data
    """
    return Pipeline(
        [
            node(
                func=load_data,
                inputs="raw_account_data",
                outputs="validated_raw_data",
                name="load_data",
                tags="ingestion",
            ),
            node(
                func=collect_missing_stats,
                inputs="validated_raw_data",
                outputs="raw_missing_stats",
                name="collect_raw_missing_stats",
                tags="ingestion",
            ),
            node(
                func=standardize_columns,
                inputs="validated_raw_data",
                outputs="standardized_data_raw",
                name="standardize_columns",
                tags="ingestion",
            ),
            node(
                func=transform_dtype,
                inputs="standardized_data_raw",
                outputs="standardized_data",
                name="transform_dtype",
                tags="ingestion",
            ),
            node(
                func=extract_metadata,
                inputs="standardized_data",
                outputs="ingestion_metadata",
                name="extract_ingestion_metadata",
                tags="ingestion",
            )
        ]
    )
