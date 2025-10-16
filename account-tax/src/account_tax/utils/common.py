"""Shared helper utilities for project-wide tasks."""

from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from typing import Any, Dict, Iterable

import numpy as np
import torch


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


def compose_deepspeed_config(
    training_args_cfg: Dict[str, Any],
    deepspeed_cfg: Dict[str, Any],
    num_gpus: int,
) -> Dict[str, Any] | None:
    """Merge DeepSpeed config with TrainingArguments values to avoid duplication."""

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
