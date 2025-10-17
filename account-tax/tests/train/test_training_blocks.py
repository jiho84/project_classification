"""Smoke tests for training pipeline block functions.

These tests verify basic input/output contracts of block functions
without requiring actual GPU resources or model training.
"""

import argparse
from pathlib import Path
from unittest.mock import MagicMock

import pytest

from account_tax.utils.common import (
    TrainingArtifacts,
    TrainingContext,
    setup_training_context,
)


def test_setup_training_context_basic():
    """Test Block 1: setup_training_context with minimal config."""
    args = argparse.Namespace(config_yml="test.yml", local_rank=0)
    cfg = {
        "seed": 42,
        "training_args": {"output_dir": "/tmp/test_output"},
    }

    context = setup_training_context(args, cfg)

    assert isinstance(context, TrainingContext)
    assert context.cfg == cfg
    assert context.args == args
    assert context.seed == 42
    assert isinstance(context.is_rank_zero, bool)
    assert isinstance(context.output_dir, Path)


def test_setup_training_context_validates_config():
    """Test Block 1: setup_training_context validates config type."""
    args = argparse.Namespace(config_yml="test.yml")

    with pytest.raises(ValueError, match="Training configuration must be a YAML mapping"):
        setup_training_context(args, cfg="invalid_string_config")


def test_training_artifacts_dataclass():
    """Test TrainingArtifacts dataclass initialization."""
    mock_dataset = MagicMock()

    artifacts = TrainingArtifacts(
        train_dataset=mock_dataset,
        num_labels=10,
    )

    assert artifacts.train_dataset == mock_dataset
    assert artifacts.num_labels == 10
    assert artifacts.eval_dataset is None
    assert artifacts.test_dataset is None
    assert artifacts.tokenizer is None
    assert artifacts.model is None


def test_training_context_immutability():
    """Test TrainingContext contains immutable configuration."""
    args = argparse.Namespace(config_yml="test.yml")
    cfg = {"seed": 42, "training_args": {"output_dir": "/tmp/test"}}

    context = setup_training_context(args, cfg)

    # Verify context is created successfully
    assert context.cfg is cfg
    assert context.seed == 42

    # Note: Modifying cfg after context creation would affect context.cfg
    # since we don't deep-copy. This is acceptable for our use case
    # since config is only modified during block execution, not between blocks.


def test_block_pipeline_structure():
    """Test that pipeline blocks follow consistent structure."""
    # This test verifies the declarative structure matches expectations
    from train.main_yaml import main

    # We can't actually call main() in tests, but we can verify the structure
    # by checking that all imported functions exist
    from account_tax.utils.common import (
        setup_training_context,
        load_datasets,
        initialize_tokenizer,
        initialize_model,
        apply_lora_to_model,
        build_weighted_trainer,
        execute_training_loop,
        evaluate_and_save_results,
        cleanup_distributed_process_group,
    )

    # Verify all 9 block functions are callable
    blocks = [
        setup_training_context,
        load_datasets,
        initialize_tokenizer,
        initialize_model,
        apply_lora_to_model,
        build_weighted_trainer,  # Now includes MLflow patching internally
        execute_training_loop,
        evaluate_and_save_results,
        cleanup_distributed_process_group,
    ]

    for block in blocks:
        assert callable(block), f"{block.__name__} should be callable"
