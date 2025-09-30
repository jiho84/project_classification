"""Training pipeline definition."""

from kedro.pipeline import Pipeline, node
from .nodes import (
    tokenize_datasets,
    load_model,
    apply_optimization,
    prepare_trainer,
    train_model,
    evaluate_model
)


def create_pipeline(**kwargs) -> Pipeline:
    """Create the training pipeline for NLP models.

    This pipeline implements a complete training workflow:
    1. Tokenization of text data
    2. Model loading and configuration
    3. Optimization techniques (LoRA, compilation)
    4. Trainer preparation with DeepSpeed support
    5. Model training
    6. Evaluation on validation and test sets

    The pipeline is designed to work with HuggingFace transformers
    and supports advanced features like mixed precision training,
    gradient checkpointing, and distributed training.
    """
    return Pipeline(
        [
            node(
                func=tokenize_datasets,
                inputs=["text_datasets", "params:train.tokenization"],
                outputs=["tokenized_datasets", "tokenization_metadata"],
                name="tokenize_datasets",
                tags=["train", "tokenization"],
            ),
            node(
                func=load_model,
                inputs=["tokenized_datasets", "params:train.model"],
                outputs=["base_model", "model_metadata"],
                name="load_model",
                tags=["train", "model_loading"],
            ),
            node(
                func=apply_optimization,
                inputs=["base_model", "params:train.optimization"],
                outputs=["optimized_model", "optimization_metadata"],
                name="apply_optimization",
                tags=["train", "optimization"],
            ),
            node(
                func=prepare_trainer,
                inputs=["optimized_model", "tokenized_datasets", "params:train"],
                outputs=["trainer_components", "trainer_metadata"],
                name="prepare_trainer",
                tags=["train", "trainer_setup"],
            ),
            node(
                func=train_model,
                inputs=["trainer_components", "params:train"],
                outputs=["trained_model", "training_metrics", "training_artifacts"],
                name="train_model",
                tags=["train", "training"],
            ),
            node(
                func=evaluate_model,
                inputs=["trainer_components", "params:train.evaluation"],
                outputs=["evaluation_metrics", "evaluation_artifacts"],
                name="evaluate_model",
                tags=["train", "evaluation"],
            ),
        ]
    )