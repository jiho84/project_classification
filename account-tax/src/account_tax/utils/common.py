"""Shared helper utilities for project-wide tasks."""

from __future__ import annotations

import argparse
import json
import logging
import os
from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, Iterable

import numpy as np
import torch
import torch.nn as nn
from transformers import Trainer, TrainerCallback, TrainingArguments

try:
    from peft import LoraConfig, TaskType, get_peft_model
except ImportError:
    LoraConfig = None
    TaskType = None
    get_peft_model = None


def find_project_root() -> Path:
    """Locate the Kedro project root by searching for ``pyproject.toml``."""

    current = Path(__file__).resolve()
    for parent in (current, *current.parents):
        if (parent / "pyproject.toml").exists():
            return parent
    return Path.cwd()


def ensure_dir(path: Path) -> Path:
    """Create parent directories for the given path and return the path."""

    path.parent.mkdir(parents=True, exist_ok=True)
    return path


def ensure_dirname(path: Path) -> Path:
    """Create the directory itself (not just parent) and return the path."""

    path.mkdir(parents=True, exist_ok=True)
    return path


def build_training_arguments(
    training_args_cfg: Dict[str, Any],
    deepspeed_cfg: Dict[str, Any],
    num_gpus: int,
    rank_zero_fn: Callable[[], bool]
):
    """Build TrainingArguments with integrated DeepSpeed config.

    Combines DeepSpeed configuration with TrainingArguments and returns
    a ready-to-use TrainingArguments object for Trainer.

    Args:
        training_args_cfg: Training arguments configuration dict
        deepspeed_cfg: DeepSpeed configuration dict (required)
        num_gpus: Number of GPUs to use
        rank_zero_fn: Function to check if current process is rank 0

    Returns:
        TrainingArguments object with DeepSpeed config integrated
    """
    from transformers import TrainingArguments

    # Step 1: Compose DeepSpeed config
    if "config" in deepspeed_cfg:
        base_config = deepspeed_cfg.get("config", {}) or {}
    else:
        base_config = {k: v for k, v in deepspeed_cfg.items() if k != "num_gpus"}
    ds_config = deepcopy(base_config)

    per_device = int(training_args_cfg.get("per_device_train_batch_size", 1))
    grad_accum = int(training_args_cfg.get("gradient_accumulation_steps", 1))
    total_gpus = max(1, int(num_gpus))

    # Set batch size parameters
    ds_config.setdefault("train_micro_batch_size_per_gpu", per_device)
    ds_config.setdefault("gradient_accumulation_steps", grad_accum)
    ds_config.setdefault("train_batch_size", per_device * grad_accum * total_gpus)

    # Sync optimizer parameters
    max_grad_norm = training_args_cfg.get("max_grad_norm")
    if max_grad_norm is not None:
        ds_config.setdefault("gradient_clipping", max_grad_norm)

    optimizer_cfg = ds_config.setdefault("optimizer", {})
    optimizer_params = optimizer_cfg.setdefault("params", {})
    learning_rate = training_args_cfg.get("learning_rate")
    if learning_rate is not None:
        optimizer_params.setdefault("lr", learning_rate)
    weight_decay = training_args_cfg.get("weight_decay")
    if weight_decay is not None:
        optimizer_params.setdefault("weight_decay", weight_decay)

    # Sync precision settings
    bf16_flag = training_args_cfg.get("bf16")
    if bf16_flag is not None:
        ds_config.setdefault("bf16", {})
        ds_config["bf16"]["enabled"] = bool(bf16_flag)

    fp16_flag = training_args_cfg.get("fp16")
    if fp16_flag is not None:
        ds_config.setdefault("fp16", {})
        ds_config["fp16"]["enabled"] = bool(fp16_flag)

    # Step 2: Build TrainingArguments
    cfg = dict(training_args_cfg)
    cfg["deepspeed"] = ds_config
    cfg.pop("num_gpus", None)

    # Override with DeepSpeed values to prevent mismatch
    cfg["per_device_train_batch_size"] = ds_config["train_micro_batch_size_per_gpu"]
    cfg["gradient_accumulation_steps"] = ds_config["gradient_accumulation_steps"]

    if rank_zero_fn():
        logger = logging.getLogger(__name__)
        logger.info("DeepSpeed override: per_device_train_batch_size = %s", cfg["per_device_train_batch_size"])
        logger.info("DeepSpeed override: gradient_accumulation_steps = %s", cfg["gradient_accumulation_steps"])

    # Trainer compatibility
    if "report_to" in cfg and isinstance(cfg["report_to"], str):
        cfg["report_to"] = [cfg["report_to"]]

    output_dir = Path(cfg.get("output_dir", "./outputs"))
    output_dir.mkdir(parents=True, exist_ok=True)

    return TrainingArguments(**cfg)




def build_class_weight_tensor(
    labels: Iterable[int],
    num_labels: int,
    alpha: float,
    weight_min: float,
    weight_max: float,
    enabled: bool = True,
) -> torch.Tensor:
    """Return a torch tensor of class weights with mean value 1.0."""

    if num_labels <= 0:
        raise ValueError("num_labels must be positive")
    if weight_min > weight_max:
        raise ValueError("weight_min must be <= weight_max")

    if not enabled:
        weights = np.ones(num_labels, dtype=np.float32)
        return torch.tensor(weights, dtype=torch.float32)

    label_array = np.asarray(labels, dtype=np.int64)
    class_counts = np.bincount(label_array, minlength=num_labels).astype(np.float64)

    zero_mask = class_counts == 0
    if zero_mask.any():
        class_counts[zero_mask] = 1.0

    weights = np.power(1.0 / class_counts, alpha, dtype=np.float64)
    weights = np.clip(weights, weight_min, weight_max)
    weights /= weights.mean()

    return torch.tensor(weights.astype(np.float32), dtype=torch.float32)


class RankZeroLogger:
    """Wrapper that only logs on rank zero."""

    def __init__(self, logger: logging.Logger, rank_zero_fn: Callable[[], bool]):
        self._logger = logger
        self._rank_zero_fn = rank_zero_fn

    def __getattr__(self, name: str):
        attr = getattr(self._logger, name)
        if not callable(attr):
            return attr

        def wrapper(*args, **kwargs):
            if self._rank_zero_fn():
                return attr(*args, **kwargs)
            return None

        return wrapper


def save_metrics(metrics: Dict[str, Any], path: str | None) -> None:
    """Save metrics dictionary to JSON file."""
    if not path:
        return
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    with open(destination, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2, ensure_ascii=False)


def compute_classification_metrics(eval_pred):
    """Compute classification metrics (accuracy, f1_weighted, f1_macro).

    Args:
        eval_pred: Tuple of (predictions, labels) from Trainer

    Returns:
        Dict with accuracy, f1_weighted, f1_macro
    """
    from sklearn.metrics import f1_score

    predictions, labels = eval_pred
    preds = predictions.argmax(axis=-1) if predictions.ndim > 1 else predictions

    return {
        "accuracy": float((preds == labels).mean()),
        "f1_weighted": float(f1_score(labels, preds, average='weighted', zero_division=0)),
        "f1_macro": float(f1_score(labels, preds, average='macro', zero_division=0)),
    }


def infer_num_labels(dataset) -> int:
    """Infer number of labels from dataset features."""
    if "labels" in dataset.features:
        feature = dataset.features["labels"]
        # SequenceClassification: ClassLabel
        if hasattr(feature, "num_classes"):
            return feature.num_classes
    return len(set(dataset["labels"]))


def maybe_apply_lora(
    model,
    lora_section: Dict[str, Any],
    logger: logging.Logger | None = None,
    rank_zero_fn: Callable[[], bool] | None = None,
) -> torch.nn.Module:
    """Apply LoRA to model if enabled in config."""
    if not lora_section.get("enable", False):
        return model
    if LoraConfig is None or get_peft_model is None:
        raise ImportError("peft package is required for LoRA but is not installed")

    cfg = dict(lora_section.get("config", {}))
    task_type = cfg.pop("task_type", "SEQ_CLS")
    if TaskType is not None and isinstance(task_type, str):
        task_type_enum = TaskType[task_type]
        cfg["task_type"] = task_type_enum

    lora_cfg = LoraConfig(**cfg)
    effective_logger = logging.getLogger(__name__) if logger is None else logger
    if rank_zero_fn is None or rank_zero_fn():
        effective_logger.info("Applying LoRA with config: %s", cfg)
    return get_peft_model(model, lora_cfg)


class GPUMemoryCallback(TrainerCallback):
    """Callback to monitor and log GPU memory usage during training."""

    def __init__(self, rank_zero_fn: Callable[[], bool]):
        """Initialize GPU memory callback.

        Args:
            rank_zero_fn: Function that returns True if current process is rank 0.
        """
        self.is_rank_zero = rank_zero_fn
        self.enabled = self.is_rank_zero() and torch.cuda.is_available()
        self.gpu_count = torch.cuda.device_count() if self.enabled else 0
        self.device_props = []
        if self.enabled and self.gpu_count > 0:
            props = torch.cuda.get_device_properties(0)
            self.device_props.append(props)

    def on_log(self, args, state, control, logs=None, **kwargs):
        """Add GPU memory usage to logs."""
        if not self.enabled or logs is None or self.gpu_count == 0:
            return

        allocated = torch.cuda.memory_allocated(0) / 1024**3
        reserved = torch.cuda.memory_reserved(0) / 1024**3
        total = self.device_props[0].total_memory / 1024**3
        percent = (allocated / total) * 100 if total > 0 else 0

        logs["gpu_mem"] = f"{allocated:.1f}GB/{total:.0f}GB ({percent:.0f}%)"
        logs["gpu_reserved"] = f"{reserved:.1f}GB"


class WeightedTrainer(Trainer):
    """Trainer with custom weighted loss for imbalanced classification."""

    def __init__(self, *args, class_weights: torch.Tensor | None = None, **kwargs):
        """Initialize weighted trainer.

        Args:
            class_weights: Tensor of class weights for loss calculation.
            *args, **kwargs: Arguments passed to parent Trainer.
        """
        super().__init__(*args, **kwargs)
        self.class_weights = class_weights
        # Fix gradient accumulation loss scaling: Let Trainer handle loss averaging
        self.model_accepts_loss_kwargs = False

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        """Compute weighted cross-entropy loss."""
        labels = inputs.get("labels")

        if labels is None or self.class_weights is None:
            # Fallback to default loss
            return super().compute_loss(model, inputs, return_outputs=return_outputs, num_items_in_batch=num_items_in_batch)

        # Remove labels to prevent model from computing loss internally
        inputs = dict(inputs)
        labels = inputs.pop("labels")

        # Forward pass without labels
        outputs = model(**inputs)
        logits = outputs.logits

        # Custom weighted loss with ignore_index=-100
        loss_fct = nn.CrossEntropyLoss(
            weight=self.class_weights.to(logits.device, dtype=logits.dtype),
            ignore_index=-100  # Important: ignore padding tokens!
        )
        loss = loss_fct(logits.view(-1, self.model.config.num_labels), labels.view(-1))

        return (loss, outputs) if return_outputs else loss


# ==============================================================================
# Training Pipeline Block Functions (for main_yaml.py subprocess refactoring)
# ==============================================================================


@dataclass
class TrainingContext:
    """Immutable training environment and configuration.

    This object is created once at the beginning of the training subprocess
    and passed through all block functions. It contains all configuration
    and environment state that doesn't change during training.

    Attributes:
        cfg: Complete training configuration (from YAML)
        args: Command-line arguments (from argparse)
        seed: Random seed for reproducibility
        is_rank_zero: Whether current process is rank 0
        output_dir: Directory for saving outputs
    """
    cfg: Dict[str, Any]
    args: argparse.Namespace
    seed: int
    is_rank_zero: bool
    output_dir: Path


@dataclass
class TrainingArtifacts:
    """Mutable training artifacts passed between block functions.

    This object carries the state that gets built up progressively
    through the training pipeline. Each block function receives this
    object and may modify it or return a new version.

    Attributes:
        train_dataset: Training dataset
        eval_dataset: Evaluation dataset (optional)
        test_dataset: Test dataset (optional)
        num_labels: Number of classification labels
        tokenizer: HuggingFace tokenizer
        model: Model instance (with optional LoRA)
        collator: Data collator for batching
        training_args: TrainingArguments object
        class_weights: Class weight tensor for loss
        trainer: Configured Trainer instance (optional)
        training_result: Result from trainer.train() (optional)
    """
    train_dataset: Any  # datasets.Dataset
    eval_dataset: Any | None = None
    test_dataset: Any | None = None
    num_labels: int = 0
    tokenizer: Any = None  # transformers.PreTrainedTokenizer
    model: torch.nn.Module | None = None
    collator: Any = None  # transformers.DataCollator
    training_args: TrainingArguments | None = None
    class_weights: torch.Tensor | None = None
    trainer: Trainer | None = None
    training_result: Any = None


def setup_training_context(args: argparse.Namespace, cfg: Dict[str, Any]) -> TrainingContext:
    """Block 1: Initialize training context from args and config.

    This is the first block in the training pipeline. It validates the
    configuration, sets the random seed, and creates the immutable context
    object that will be passed to all subsequent blocks.

    Args:
        args: Parsed command-line arguments
        cfg: Loaded YAML configuration

    Returns:
        TrainingContext with initialized environment

    Raises:
        ValueError: If configuration validation fails
    """
    from transformers import set_seed

    if not isinstance(cfg, dict):
        raise ValueError("Training configuration must be a YAML mapping")

    seed = int(cfg.get("seed", 42))
    set_seed(seed)

    is_rank_zero = os.environ.get("LOCAL_RANK", "0") == "0"
    output_dir = Path(cfg.get("training_args", {}).get("output_dir", "./outputs"))
    output_dir.mkdir(parents=True, exist_ok=True)

    return TrainingContext(
        cfg=cfg,
        args=args,
        seed=seed,
        is_rank_zero=is_rank_zero,
        output_dir=output_dir,
    )


def load_datasets(
    context: TrainingContext,
    logger: logging.Logger,
) -> TrainingArtifacts:
    """Block 2: Load tokenized datasets from disk.

    Args:
        context: Training context with data config
        logger: Logger instance

    Returns:
        TrainingArtifacts with train/eval/test datasets

    Raises:
        FileNotFoundError: If dataset path doesn't exist
    """
    from datasets import load_from_disk

    data_cfg = context.cfg.get("data", {})
    tokenized_path = data_cfg["tokenized_path"]

    logger.info("Loading tokenized datasets from %s", tokenized_path)
    dataset_dict = load_from_disk(tokenized_path)

    train_split = data_cfg.get("train_split", "train")
    eval_split = data_cfg.get("eval_split")
    test_split = data_cfg.get("test_split")

    return TrainingArtifacts(
        train_dataset=dataset_dict[train_split],
        eval_dataset=dataset_dict[eval_split] if eval_split and eval_split in dataset_dict else None,
        test_dataset=dataset_dict[test_split] if test_split and test_split in dataset_dict else None,
    )


def initialize_tokenizer(
    context: TrainingContext,
    artifacts: TrainingArtifacts,
    logger: logging.Logger,
) -> TrainingArtifacts:
    """Block 3: Initialize and configure tokenizer.

    Args:
        context: Training context with model config
        artifacts: Artifacts with datasets
        logger: Logger instance

    Returns:
        Updated artifacts with tokenizer and num_labels

    Raises:
        ValueError: If tokenizer configuration fails
    """
    from transformers import AutoTokenizer

    model_cfg = context.cfg.get("model", {})
    model_name = model_cfg["name_or_path"]

    tokenizer = AutoTokenizer.from_pretrained(
        model_name,
        trust_remote_code=model_cfg.get("trust_remote_code", False)
    )

    # Handle missing pad_token
    if tokenizer.pad_token_id is None:
        eos_token = tokenizer.eos_token
        if eos_token is None:
            raise ValueError("Tokenizer is missing both pad and eos tokens; cannot proceed.")
        pad_token_id = tokenizer.convert_tokens_to_ids(eos_token)
        if pad_token_id is None:
            raise ValueError("Unable to determine token id for eos token; cannot reuse for padding.")
        tokenizer.pad_token = eos_token
        tokenizer.pad_token_id = pad_token_id

    # Infer num_labels
    num_labels = model_cfg.get("num_labels")
    if num_labels is None:
        num_labels = infer_num_labels(artifacts.train_dataset)
        logger.info("Inferred num_labels=%s", num_labels)

    artifacts.tokenizer = tokenizer
    artifacts.num_labels = num_labels
    return artifacts


def initialize_model(
    context: TrainingContext,
    artifacts: TrainingArtifacts,
    logger: logging.Logger,
) -> TrainingArtifacts:
    """Block 4: Initialize base model without optimization.

    Loads the pretrained model and configures basic settings.
    Does not apply any optimization (LoRA, etc).

    Args:
        context: Training context with model config
        artifacts: Artifacts with tokenizer and num_labels
        logger: Logger instance

    Returns:
        Updated TrainingArtifacts with base model
    """
    from transformers import AutoModelForSequenceClassification

    model_cfg = context.cfg.get("model", {})

    # Determine dtype
    torch_dtype_str = model_cfg.get("torch_dtype", "bfloat16")
    dtype_map = {
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
        "float32": torch.float32,
    }
    torch_dtype = dtype_map.get(torch_dtype_str, torch.bfloat16)

    # Load model
    model_name = model_cfg["name_or_path"]
    logger.info("Loading model: %s (dtype=%s)", model_name, torch_dtype_str)

    model = AutoModelForSequenceClassification.from_pretrained(
        model_name,
        num_labels=artifacts.num_labels,
        torch_dtype=torch_dtype,
        trust_remote_code=model_cfg.get("trust_remote_code", False),
    )

    # Configure pad_token
    if getattr(model.config, "pad_token_id", None) is None:
        model.config.pad_token_id = artifacts.tokenizer.pad_token_id

    # Gradient checkpointing
    if model_cfg.get("gradient_checkpointing", False):
        model.gradient_checkpointing_enable()
        logger.info("Gradient checkpointing enabled")

    artifacts.model = model
    return artifacts


def apply_lora_to_model(
    context: TrainingContext,
    artifacts: TrainingArtifacts,
    logger: logging.Logger,
) -> TrainingArtifacts:
    """Block 5: Apply LoRA optimization to model if configured.

    Args:
        context: Training context with LoRA config
        artifacts: Artifacts with base model
        logger: Logger instance

    Returns:
        Updated artifacts with LoRA-optimized model

    Raises:
        ImportError: If LoRA is enabled but peft not installed
    """
    lora_cfg = context.cfg.get("lora", {})

    artifacts.model = maybe_apply_lora(
        artifacts.model,
        lora_cfg,
        logger=logger,
        rank_zero_fn=lambda: context.is_rank_zero
    )

    return artifacts


def build_weighted_trainer(
    context: TrainingContext,
    artifacts: TrainingArtifacts,
    logger: logging.Logger,
    logger_zero: Any,  # RankZeroLogger
) -> TrainingArtifacts:
    """Block 6: Build WeightedTrainer with all components and patch MLflow callback.

    Builds TrainingArguments, data collator, class weights, creates the
    WeightedTrainer instance with all necessary callbacks, and patches
    MLflowCallback to prevent subprocess hang.

    Args:
        context: Training context
        artifacts: Training artifacts with model, tokenizer, datasets
        logger: Logger instance
        logger_zero: Rank-zero logger

    Returns:
        Updated TrainingArtifacts with configured trainer
    """
    from transformers import DataCollatorWithPadding

    training_args_cfg = context.cfg.get("training_args", {})
    deepspeed_cfg = context.cfg.get("deepspeed", {})

    training_args = build_training_arguments(
        training_args_cfg,
        deepspeed_cfg,
        num_gpus=deepspeed_cfg.get("num_gpus", 4),
        rank_zero_fn=lambda: context.is_rank_zero,
    )

    collator = DataCollatorWithPadding(
        tokenizer=artifacts.tokenizer,
        pad_to_multiple_of=8 if training_args.bf16 or training_args.fp16 else None,
    )

    # Build class weights
    loss_cfg = context.cfg.get("loss", {})
    use_class_weights = bool(loss_cfg.get("use_class_weights", False))
    class_weight_alpha = float(loss_cfg.get("class_weight_alpha", 1.0))
    class_weight_min = float(loss_cfg.get("class_weight_min", 1.0))
    class_weight_max = float(loss_cfg.get("class_weight_max", 1.0))

    train_labels = np.array(artifacts.train_dataset["labels"])
    class_weights_tensor = build_class_weight_tensor(
        labels=train_labels,
        num_labels=artifacts.num_labels,
        alpha=class_weight_alpha,
        weight_min=class_weight_min,
        weight_max=class_weight_max,
        enabled=use_class_weights,
    )

    logger_zero.info(
        "Class weights configured (alpha=%.3f, min=%.3f, max=%.3f, mean=%.4f)",
        class_weight_alpha,
        class_weight_min,
        class_weight_max,
        float(class_weights_tensor.mean().item()),
    )

    # Create callbacks
    callbacks = []
    if torch.cuda.is_available():
        callbacks.append(GPUMemoryCallback(rank_zero_fn=lambda: context.is_rank_zero))

    trainer = WeightedTrainer(
        model=artifacts.model,
        args=training_args,
        train_dataset=artifacts.train_dataset,
        eval_dataset=artifacts.eval_dataset,
        processing_class=artifacts.tokenizer,
        data_collator=collator,
        class_weights=class_weights_tensor,
        compute_metrics=compute_classification_metrics if artifacts.eval_dataset is not None else None,
        callbacks=callbacks,
    )

    # Patch MLflow callback to prevent subprocess hang
    _patch_mlflow_callback_internal(trainer, logger_zero)

    artifacts.collator = collator
    artifacts.training_args = training_args
    artifacts.class_weights = class_weights_tensor
    artifacts.trainer = trainer
    return artifacts


def _patch_mlflow_callback_internal(trainer: Trainer, logger_zero: Any) -> None:
    """Internal: Patch MLflowCallback to prevent subprocess hang.

    This is called automatically by build_weighted_trainer().

    The hang occurs when nested MLflow run tries to finalize and sync with
    parent process. We skip mlflow.end_run() in subprocess; parent Kedro
    process will handle run finalization.

    Args:
        trainer: Trainer instance to patch
        logger_zero: Rank-zero logger
    """
    try:
        from transformers.integrations import MLflowCallback
    except ImportError:
        return  # MLflow not available, nothing to patch

    if MLflowCallback is None:
        return

    for callback in trainer.callback_handler.callbacks:
        if isinstance(callback, MLflowCallback):
            def safe_on_train_end(self, args, state, control, **kwargs):  # noqa: ARG001
                """Skip mlflow.end_run() that causes hang in subprocess.

                Root cause: mlflow.end_run() tries to:
                1. Flush metrics to tracking server
                2. Sync artifacts to storage
                3. Update run status to FINISHED
                4. Notify parent run (for nested runs)

                Step 4 causes hang because subprocess cannot communicate
                with parent process's ThreadLocal MLflow context.

                Solution: Skip end_run() here. Parent Kedro process will
                handle run finalization after subprocess completes.
                """
                logger_zero.info("MLflow metrics logged successfully during training")
                logger_zero.info("Skipping mlflow.end_run() in subprocess to prevent hang")
                logger_zero.info("Parent Kedro process will finalize MLflow run")
                return control

            # Replace the method (binding to instance)
            callback.on_train_end = safe_on_train_end.__get__(callback, MLflowCallback)
            logger_zero.info("MLflowCallback.on_train_end() overridden for hang prevention")
            break


def execute_training_loop(
    context: TrainingContext,
    artifacts: TrainingArtifacts,
    logger: logging.Logger,
    logger_zero: Any,  # RankZeroLogger
) -> TrainingArtifacts:
    """Block 8: Execute training loop with checkpoint resume support.

    Runs trainer.train() with optional checkpoint resume. Handles interrupts
    and errors gracefully.

    Args:
        context: Training context
        artifacts: Training artifacts with configured trainer
        logger: Logger instance
        logger_zero: Rank-zero logger

    Returns:
        Updated TrainingArtifacts with training_result

    Raises:
        Exception: If training fails (propagates after logging)
    """
    resume_cfg = context.cfg.get("resume", {})
    resume_enabled = resume_cfg.get("enabled", False)
    checkpoint_path = resume_cfg.get("checkpoint_path") if resume_enabled else None

    if checkpoint_path:
        logger_zero.info("Resuming training from checkpoint: %s", checkpoint_path)
    else:
        logger_zero.info("Starting training from scratch (no checkpoint resume)")

    training_completed = False
    try:
        artifacts.trainer.train(resume_from_checkpoint=checkpoint_path)
        training_completed = True
        logger.info("trainer.train() returned successfully")
    except KeyboardInterrupt:
        logger.warning("Training interrupted by user")
    except Exception as e:
        logger.error("Training failed with error: %s", e)
        raise

    artifacts.training_result = {"completed": training_completed}
    return artifacts


def evaluate_and_save_results(
    context: TrainingContext,
    artifacts: TrainingArtifacts,
    logger_zero: Any,  # RankZeroLogger
) -> None:
    """Block 9: Run test evaluation, save model, and save metrics.

    Evaluates on test set if available, saves the final model (with DeepSpeed
    all-rank participation), and writes metrics to JSON files.

    Args:
        context: Training context
        artifacts: Training artifacts with trained model
        logger_zero: Rank-zero logger

    Returns:
        None (saves to disk)

    Raises:
        RuntimeError: If DeepSpeed is not initialized
    """
    training_completed = artifacts.training_result.get("completed", False)

    # Test evaluation
    eval_metrics: Dict[str, Any] | None = None
    test_metrics: Dict[str, Any] | None = None
    if artifacts.test_dataset is not None and training_completed:
        try:
            test_metrics = artifacts.trainer.evaluate(
                eval_dataset=artifacts.test_dataset,
                metric_key_prefix="test"
            )
            logger_zero.info(
                "Test evaluation completed: %s",
                {k: float(v) if isinstance(v, (int, float)) else v for k, v in test_metrics.items()}
            )
        except Exception as e:
            logger_zero.warning("Test evaluation failed: %s", e)

    # Save model with DeepSpeed all-rank participation
    if not torch.distributed.is_initialized():
        raise RuntimeError("DeepSpeed expected but torch.distributed is not initialized")

    logger_zero.info("DeepSpeed detected: saving with all-rank participation")
    torch.distributed.barrier()

    unwrapped_model = artifacts.trainer.model
    if hasattr(unwrapped_model, 'module'):
        unwrapped_model = unwrapped_model.module

    unwrapped_model.save_pretrained(
        context.output_dir,
        safe_serialization=True,
    )
    artifacts.tokenizer.save_pretrained(context.output_dir)
    torch.distributed.barrier()

    logger_zero.info("PEFT adapter saved to %s for inference (all-rank save)", context.output_dir)

    # Save metrics (rank 0 only)
    if artifacts.trainer.is_world_process_zero():
        metrics_cfg = context.cfg.get("metrics", {})
        metrics: Dict[str, Any] = {"global_step": artifacts.trainer.state.global_step}
        if eval_metrics is not None:
            metrics.update(eval_metrics)
            save_metrics(metrics, metrics_cfg.get("path"))
        if test_metrics is not None:
            test_payload: Dict[str, Any] = {"global_step": artifacts.trainer.state.global_step}
            test_payload.update(test_metrics)
            save_metrics(test_payload, metrics_cfg.get("test_path"))

        logger_zero.info("Training can be resumed from checkpoint-%d/", artifacts.trainer.state.global_step)
        logger_zero.info("MLflow artifacts will be logged by Kedro node.")


def cleanup_distributed_process_group(logger_zero: Any) -> None:
    """Block 10: Clean up DeepSpeed distributed process group.

    Destroys the process group to release resources. Handles errors gracefully
    since cleanup may fail in some edge cases.

    Args:
        logger_zero: Rank-zero logger

    Returns:
        None
    """
    try:
        torch.distributed.barrier()
        torch.distributed.destroy_process_group()
        logger_zero.info("DeepSpeed process group destroyed successfully")
    except Exception as e:
        logger_zero.warning("Failed to destroy process group: %s", e)

    logger_zero.info("Training script exiting normally")
