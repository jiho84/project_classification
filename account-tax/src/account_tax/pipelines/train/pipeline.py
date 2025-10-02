"""Training pipeline definition."""

from kedro.pipeline import Pipeline, node
from .nodes import (
    tokenize_datasets,
)


def create_pipeline(**kwargs) -> Pipeline:
    """Create the training pipeline for NLP models.

    Pipeline: tokenize_datasets (with integrated token length analysis)
    Input: serialized_datasets (from Split pipeline)
    Output: tokenized_datasets (MLflow Artifact), token_length_report
    """
    return Pipeline([
        node(
            func=tokenize_datasets,
            inputs=["serialized_datasets", "params:train.tokenization"],
            outputs=["tokenized_datasets", "token_length_report"],
            name="tokenize_datasets",
            tags="train",
        ),
    ])