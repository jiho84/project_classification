"""Account-Tax Project Pipeline Registry."""

from kedro.pipeline import Pipeline
from .pipelines import (
    ingestion,
    preprocess,
    feature,
    split,
    train
    # evaluation  # Temporarily disabled
)


def register_pipelines() -> dict[str, Pipeline]:
    """Register account-tax project pipelines.

    Returns:
        Mapping from pipeline names to Pipeline objects.
    """

    # Individual pipelines
    ingestion_pipeline = ingestion.create_pipeline()
    preprocess_pipeline = preprocess.create_pipeline()
    feature_pipeline = feature.create_pipeline()
    split_pipeline = split.create_pipeline()
    train_pipeline = train.create_pipeline()  # NLP training pipeline
    # evaluation_pipeline = evaluation.create_pipeline()  # Temporarily disabled

    # Data preparation pipeline (before split)
    data_prep_pipeline = (
        ingestion_pipeline +
        preprocess_pipeline +
        feature_pipeline
    )

    # Full preprocessing pipeline (includes split)
    full_preprocess = (
        data_prep_pipeline +
        split_pipeline
    )

    # Full NLP pipeline (preprocess + train)
    full_pipeline = (
        full_preprocess +
        train_pipeline
    )

    # Data pipeline (only data preparation, no split)
    data_pipeline = data_prep_pipeline  # ingestion + preprocess + feature

    # Training pipeline (split + train)
    training_pipeline = split_pipeline + train_pipeline

    # End-to-end pipeline (data + training)
    e2e_pipeline = data_pipeline + training_pipeline

    return {
        "__default__": data_pipeline,
        "data": data_pipeline,
        "training": training_pipeline,
        "e2e": e2e_pipeline,
        # Legacy names for backward compatibility
        "full_preprocess": full_preprocess,
        "full": full_pipeline,
        "data_prep": data_prep_pipeline,
        # Individual pipelines
        "ingestion": ingestion_pipeline,
        "preprocess": preprocess_pipeline,
        "feature": feature_pipeline,
        "split": split_pipeline,
        "train": train_pipeline,
        # "evaluation": evaluation_pipeline,  # Temporarily disabled
    }
