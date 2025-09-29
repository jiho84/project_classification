"""Simplified dataset preparation nodes."""

from __future__ import annotations

import logging
import random
from typing import Dict, List, Tuple, Any

import pandas as pd
import numpy as np
from datasets import Dataset, DatasetDict, ClassLabel
from transformers import AutoTokenizer
import mlflow

logger = logging.getLogger(__name__)


def _initialize_label_slots(max_classes: int, dummy_prefix: str) -> List[str]:
    """Create deterministic dummy label slots."""
    return [f"{dummy_prefix}{i + 1}" for i in range(max_classes)]


def _upsert_labels_into_slots(
    names: List[str],
    new_labels: List[str],
    dummy_prefix: str,
) -> List[str]:
    """Insert real labels into dummy slots while preserving order."""
    existing = set(names)
    candidates = [label for label in new_labels if label not in existing]

    dummy_positions = (idx for idx, name in enumerate(names) if name.startswith(dummy_prefix))
    for label in candidates:
        position = next(dummy_positions, None)
        if position is None:
            raise RuntimeError(
                f"Not enough dummy slots to accommodate label '{label}'. Increase max_classes."
            )
        names[position] = label
        existing.add(label)

    return names


def make_label2id(names: List[str]) -> Dict[str, int]:
    """Create label → id mapping preserving index positions."""
    return {name: idx for idx, name in enumerate(names)}


def make_id2label(names: List[str]) -> Dict[int, str]:
    """Create id → label mapping preserving index positions."""
    return {idx: name for idx, name in enumerate(names)}


def create_dataset(
    base_table: pd.DataFrame,
    params: Dict[str, Any],
) -> Tuple[Dataset, List[str]]:
    """Create a HuggingFace Dataset and deterministic label slot list."""

    label_column = params.get("label_column", "acct_code")
    max_classes = params.get("max_classes", 100)
    dummy_prefix = params.get("dummy_prefix", "dummy")

    if label_column not in base_table.columns:
        raise ValueError(f"Label column '{label_column}' not present in base table")

    cleaned = base_table.reset_index(drop=True)

    dataset = Dataset.from_pandas(cleaned, preserve_index=False)

    unique_labels = cleaned[label_column].astype(str).drop_duplicates().tolist()
    names = _initialize_label_slots(max_classes, dummy_prefix)
    names = _upsert_labels_into_slots(names, unique_labels, dummy_prefix)

    logger.info("Dataset created with %s rows and %s columns", dataset.num_rows, len(dataset.column_names))
    logger.info("Real labels registered: %s", len(unique_labels))

    return dataset, names


def to_hf_and_split(
    dataset: Dataset,
    label_col: str,
    seed: int,
    test_size: float,
    val_size: float,
) -> DatasetDict:
    """Split a HuggingFace Dataset into train/valid/test with stratification when possible."""

    if label_col not in dataset.column_names:
        raise ValueError(f"Label column '{label_col}' not present in dataset")

    if not 0 < test_size < 1:
        raise ValueError("test_size must be between 0 and 1")

    if not 0 < val_size < 1:
        raise ValueError("val_size must be between 0 and 1")

    if val_size >= test_size:
        raise ValueError("val_size must be smaller than test_size to allow stratification")

    logger.info(
        "Splitting dataset with %s rows (label column: %s): test_size=%s, val_size=%s",
        dataset.num_rows,
        label_col,
        test_size,
        val_size,
    )

    try:
        tmp = dataset.train_test_split(
            test_size=test_size,
            stratify_by_column=label_col,
            seed=seed,
        )
    except ValueError:
        logger.warning("Stratified split failed; falling back to random split for train/test.")
        tmp = dataset.train_test_split(test_size=test_size, seed=seed)

    remain = tmp["test"]
    val_ratio = val_size / test_size

    try:
        remain_split = remain.train_test_split(
            test_size=val_ratio,
            stratify_by_column=label_col,
            seed=seed,
        )
    except ValueError:
        logger.warning("Stratified split failed; falling back to random split for valid/test.")
        remain_split = remain.train_test_split(test_size=val_ratio, seed=seed)

    splits = DatasetDict(
        train=tmp["train"],
        valid=remain_split["train"],
        test=remain_split["test"],
    )

    for split_name in ("train", "valid", "test"):
        logger.info("Split '%s' size: %s", split_name, len(splits[split_name]))

    return splits


def labelize_and_cast(
    splits: DatasetDict,
    names: List[str],
    label_col: str,
    dummy_label: str = "dummy1",
    num_proc: int = 1,
) -> DatasetDict:
    """Attach integer labels and ClassLabel schema to DatasetDict splits."""
    if label_col not in splits["train"].column_names:
        raise ValueError(f"Label column '{label_col}' not present in dataset")

    label2id = make_label2id(names)
    fallback = label2id.get(dummy_label, 0)

    def encode(batch: Dict[str, Any]) -> Dict[str, Any]:
        values = [label2id.get(str(val), fallback) for val in batch[label_col]]
        batch["labels"] = values
        return batch

    encoded = splits.map(encode, batched=True, num_proc=num_proc)

    class_label = ClassLabel(names=names)
    try:
        encoded = encoded.cast_column("labels", class_label)
    except (NotImplementedError, ValueError):
        features = encoded["train"].features.copy()
        features["labels"] = class_label
        encoded = encoded.cast(features)

    encoded.label_metadata = {
        "label2id": label2id,
        "id2label": make_id2label(names),
        "names": names,
        "dummy_label": dummy_label,
    }

    return encoded


def serialize_for_nlp(
    dataset_dict: DatasetDict,
    params: Dict[str, Any]
) -> DatasetDict:
    """Serialize structured splits into text sequences for downstream NLP."""

    text_columns = params.get("text_columns", [])
    separator = params.get("separator", ", ")
    include_column_names = params.get("include_column_names", True)
    num_proc = params.get("num_proc", 1)
    keep_columns = params.get("retain_columns", ["text", "labels"])
    label_column = params.get("label_column")
    if label_column and label_column not in keep_columns:
        keep_columns.append(label_column)

    if not text_columns:
        text_columns = [
            col for col in dataset_dict["train"].column_names
            if col not in ["labels", "label_id"]
        ]

    def create_text(batch: Dict[str, Any]) -> Dict[str, Any]:
        size = len(batch.get(text_columns[0], [])) if text_columns else len(batch.get("labels", []))
        texts: List[str] = []
        for idx in range(size):
            parts: List[str] = []
            for col in text_columns:
                column_values = batch.get(col)
                if column_values is None:
                    value = "missing"
                else:
                    value = column_values[idx]
                    if value is None:
                        value = "missing"
                value_str = str(value)
                parts.append(f"{col}: {value_str}" if include_column_names else value_str)

            texts.append(separator.join(parts))

        batch["text"] = texts
        return batch

    dataset_dict = dataset_dict.map(
        create_text,
        desc="Creating text sequences",
        batched=True,
        num_proc=num_proc,
    )

    drop_candidates = [
        col
        for col in dataset_dict["train"].column_names
        if col not in keep_columns
    ]
    if drop_candidates:
        dataset_dict = dataset_dict.remove_columns(drop_candidates)

    logger.info("Created text sequences from %s columns", len(text_columns))
    if len(dataset_dict["train"]) > 0:
        sample_text = dataset_dict["train"][0]["text"]
        logger.info("Sample text length: %s characters", len(sample_text))
        logger.debug("Sample text preview: %s...", sample_text[:200])

    return dataset_dict


def analyze_token_lengths(
    dataset_dict: DatasetDict,
    tokenization_params: Dict[str, Any],
    diagnostics_params: Dict[str, Any],
) -> Tuple[DatasetDict, Dict[str, Any]]:
    """Generate token length statistics and representative samples."""

    model_name = tokenization_params.get("model_name")
    if not model_name:
        raise ValueError("tokenization_params.model_name must be provided for token diagnostics")

    diagnostics_params = diagnostics_params or {}
    sample_size = int(diagnostics_params.get("sample_size", 5))
    percentiles = diagnostics_params.get("percentiles", [50, 75, 90, 95, 99])
    seed = diagnostics_params.get("seed", 42)

    tokenizer = AutoTokenizer.from_pretrained(model_name)

    rng = random.Random(seed)
    summary_per_split: Dict[str, Dict[str, float]] = {}
    samples: List[Dict[str, Any]] = []
    all_lengths: List[int] = []

    for split_name, split_ds in dataset_dict.items():
        texts = split_ds["text"]
        if not texts:
            summary_per_split[split_name] = {
                "count": 0,
                "mean": 0.0,
                "max": 0,
            }
            continue

        batch_encoding = tokenizer(
            texts,
            add_special_tokens=True,
            truncation=False,
        )
        lengths = [len(ids) for ids in batch_encoding["input_ids"]]

        all_lengths.extend(lengths)
        summary_per_split[split_name] = {
            "count": len(lengths),
            "mean": float(np.mean(lengths)),
            "max": int(np.max(lengths)),
        }

        sample_count = min(sample_size, len(texts))
        for idx in rng.sample(range(len(texts)), sample_count):
            token_ids = batch_encoding["input_ids"][idx]
            samples.append(
                {
                    "split": split_name,
                    "index": int(idx),
                    "text": texts[idx],
                    "token_length": int(lengths[idx]),
                    "token_ids": token_ids,
                    "tokens": tokenizer.convert_ids_to_tokens(token_ids),
                }
            )

    if all_lengths:
        overall_summary = {
            "count": len(all_lengths),
            "mean": float(np.mean(all_lengths)),
            "max": int(np.max(all_lengths)),
            "min": int(np.min(all_lengths)),
        }
        percentile_values = {
            f"p{int(p)}": float(np.percentile(all_lengths, p)) for p in percentiles
        }
        overall_summary.update(percentile_values)

        if mlflow.active_run():
            mlflow.log_metric("token_length_mean", overall_summary["mean"])
            mlflow.log_metric("token_length_max", overall_summary["max"])
            for key, value in percentile_values.items():
                metric_name = f"token_length_{key}"
                mlflow.log_metric(metric_name, value)
    else:
        overall_summary = {"count": 0, "mean": 0.0, "max": 0, "min": 0}
        for p in percentiles:
            overall_summary[f"p{int(p)}"] = 0.0

    report = {
        "overall": overall_summary,
        "per_split": summary_per_split,
        "samples": samples,
        "model_name": model_name,
    }

    return dataset_dict, report


def _datasetdict_to_partitions(
    dataset_dict: DatasetDict,
    split_column: str = "split",
) -> Dict[str, pd.DataFrame]:
    """Convert DatasetDict splits into pandas DataFrames keyed by split name."""

    partitions: Dict[str, pd.DataFrame] = {}
    for split_name, split_dataset in dataset_dict.items():
        frame = split_dataset.to_pandas()
        frame[split_column] = split_name
        partitions[split_name] = frame
    return partitions


def export_prepared_partitions(dataset_dict: DatasetDict) -> Dict[str, pd.DataFrame]:
    """Convert prepared DatasetDict splits to partitioned pandas DataFrames."""

    return _datasetdict_to_partitions(dataset_dict, split_column="split")


def export_text_partitions(dataset_dict: DatasetDict) -> Dict[str, pd.DataFrame]:
    """Convert serialized text DatasetDict splits to partitioned pandas DataFrames."""

    return _datasetdict_to_partitions(dataset_dict, split_column="split")
