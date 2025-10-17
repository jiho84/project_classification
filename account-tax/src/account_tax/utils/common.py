"""Shared helper utilities for project-wide tasks."""

from __future__ import annotations

import json
import logging
import os
from copy import deepcopy
from pathlib import Path
from typing import Any, Callable, Dict, Iterable

import numpy as np
import torch
import torch.nn as nn
from transformers import Trainer, TrainerCallback

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


def compose_deepspeed_config(
    training_args_cfg: Dict[str, Any],
    deepspeed_cfg: Dict[str, Any],
    num_gpus: int,
) -> Dict[str, Any] | None:
    """Merge DeepSpeed config with TrainingArguments values to avoid duplication.

    DEPRECATED: Use build_training_arguments() instead for direct TrainingArguments creation.
    This function is kept for backward compatibility.
    """

    base_config = deepspeed_cfg.get("config")
    if not base_config:
        return None

    ds_config = deepcopy(base_config)

    per_device = int(training_args_cfg.get("per_device_train_batch_size", 1))
    grad_accum = int(training_args_cfg.get("gradient_accumulation_steps", 1))
    total_gpus = max(1, int(num_gpus))

    ds_config.setdefault("train_micro_batch_size_per_gpu", per_device)
    ds_config.setdefault("gradient_accumulation_steps", grad_accum)
    ds_config.setdefault("train_batch_size", per_device * grad_accum * total_gpus)

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

    bf16_flag = training_args_cfg.get("bf16")
    if bf16_flag is not None:
        ds_config.setdefault("bf16", {})
        ds_config["bf16"]["enabled"] = bool(bf16_flag)

    fp16_flag = training_args_cfg.get("fp16")
    if fp16_flag is not None:
        ds_config.setdefault("fp16", {})
        ds_config["fp16"]["enabled"] = bool(fp16_flag)

    return ds_config


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
