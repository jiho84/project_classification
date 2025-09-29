"""Pipeline for data splitting using HuggingFace datasets."""

from kedro.pipeline import Pipeline, node
from .nodes import (
    create_dataset,
    to_hf_and_split,
    labelize_and_cast,
    serialize_for_nlp,
    export_prepared_partitions,
    export_text_partitions,
    analyze_token_lengths,
)


def create_pipeline(**kwargs) -> Pipeline:
    """Create the data splitting pipeline with Dataset operations."""
    return Pipeline([
        node(
            func=create_dataset,
            inputs=["base_table", "params:split"],
            outputs=["hf_dataset", "label_names"],
            name="create_dataset",
            tags="split",
        ),
        node(
            func=to_hf_and_split,
            inputs=[
                "hf_dataset",
                "params:split.label_column",
                "params:split.seed",
                "params:split.test_size",
                "params:split.val_size",
            ],
            outputs="split_datasets",
            name="to_hf_and_split",
            tags="split",
        ),
        node(
            func=labelize_and_cast,
            inputs=[
                "split_datasets",
                "label_names",
                "params:split.label_column",
                "params:split.dummy_label",
                "params:split.labelize_num_proc",
            ],
            outputs="prepared_datasets",
            name="labelize_and_cast",
            tags="split",
        ),
        node(
            func=export_prepared_partitions,
            inputs="prepared_datasets",
            outputs="prepared_datasets_mlflow",
            name="export_prepared_partitions",
            tags="split",
        ),
        node(
            func=serialize_for_nlp,
            inputs=["prepared_datasets", "params:train.serialization"],
            outputs="text_datasets",
            name="serialize_for_nlp",
            tags="split",
        ),
        node(
            func=analyze_token_lengths,
            inputs=[
                "text_datasets",
                "params:train.tokenization",
                "params:train.tokenization.diagnostics",
            ],
            outputs=["text_datasets_with_stats", "token_length_report"],
            name="analyze_token_lengths",
            tags="split",
        ),
        node(
            func=export_text_partitions,
            inputs="text_datasets_with_stats",
            outputs="text_datasets_mlflow",
            name="export_text_partitions",
            tags="split",
        ),
    ])
