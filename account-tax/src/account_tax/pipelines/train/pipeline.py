"""Training pipeline definition."""

from kedro.pipeline import Pipeline, node
from .nodes import tokenize_datasets


def create_pipeline(**kwargs) -> Pipeline:
    """Create the training pipeline for NLP models."""
    return Pipeline(
        [
            node(
                func=tokenize_datasets,
                inputs=["text_datasets", "params:train.tokenization"],
                outputs="tokenized_datasets",
                name="tokenize_datasets",
                tags="train",
            ),
        ]
    )
