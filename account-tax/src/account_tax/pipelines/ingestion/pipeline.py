"""Pipeline for data ingestion."""

from kedro.pipeline import Pipeline, node
from .nodes import (
    load_data,
    standardize_columns,
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
                func=standardize_columns,
                inputs="validated_raw_data",
                outputs="standardized_data",
                name="standardize_columns",
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
