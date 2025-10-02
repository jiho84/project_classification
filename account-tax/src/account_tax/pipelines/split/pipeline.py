"""Pipeline for data splitting using HuggingFace datasets."""

from kedro.pipeline import Pipeline, node
from .nodes import (
    create_dataset,
    to_hf_and_split,
    labelize_and_cast,
    serialize_to_text,
)


def create_pipeline(**kwargs) -> Pipeline:
    """Create the data splitting pipeline with Dataset operations.

    Output: serialized_datasets (text + labels) saved as PickleDataset
    """
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
            outputs="split_datasets_raw",
            name="to_hf_and_split",
            tags="split",
        ),
        node(
            func=labelize_and_cast,
            inputs=[
                "split_datasets_raw",
                "label_names",
                "params:split.label_column",
                "params:split.labelize_num_proc",
            ],
            outputs="split_datasets",
            name="labelize_and_cast",
            tags="split",
        ),
        node(
            func=serialize_to_text,
            inputs=["split_datasets", "params:train.serialization"],
            outputs="serialized_datasets",
            name="serialize_to_text",
            tags="split",
        ),
    ])
