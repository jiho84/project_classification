"""Helpers for deriving class-weight tensors used during training."""

from __future__ import annotations

from typing import Iterable

import numpy as np
import torch


def build_class_weight_tensor(
    labels: Iterable[int],
    num_labels: int,
    alpha: float,
    weight_min: float,
    weight_max: float,
) -> torch.Tensor:
    """Return a torch tensor of class weights with mean value 1.0.

    Args:
        labels: Iterable of class indices for the training split.
        num_labels: Total number of classes expected in ``labels``.
        alpha: Exponent applied to inverse class frequency.
        weight_min: Lower bound applied after exponentiation.
        weight_max: Upper bound applied after exponentiation.

    Raises:
        ValueError: If ``num_labels`` is non-positive or min/max bounds are invalid.

    """
    if num_labels <= 0:
        raise ValueError("num_labels must be positive")
    if weight_min > weight_max:
        raise ValueError("weight_min must be <= weight_max")

    label_array = np.asarray(labels, dtype=np.int64)
    class_counts = np.bincount(label_array, minlength=num_labels).astype(np.float64)

    zero_mask = class_counts == 0
    if zero_mask.any():
        class_counts[zero_mask] = 1.0

    weights = np.power(1.0 / class_counts, alpha, dtype=np.float64)
    weights = np.clip(weights, weight_min, weight_max)
    weights /= weights.mean()

    return torch.tensor(weights.astype(np.float32), dtype=torch.float32)
