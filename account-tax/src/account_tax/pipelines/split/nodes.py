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

MISSING_TOKEN = "NULL"

logger = logging.getLogger(__name__)


def _initialize_label_slots(max_classes: int) -> List[str]:
    """Create deterministic dummy label slots.

    Dummy labels are hardcoded as 'dummy_1', 'dummy_2', ... 'dummy_N'.
    """
    return [f"dummy_{i + 1}" for i in range(max_classes)]


def _upsert_labels_into_slots(
    names: List[str],
    new_labels: List[str],
) -> List[str]:
    """Insert real labels into dummy slots while preserving order.

    Real labels fill slots from left to right, remaining slots stay as dummy_N.
    """
    existing = set(names)
    candidates = [label for label in new_labels if label not in existing]

    dummy_positions = (idx for idx, name in enumerate(names) if name.startswith("dummy_"))
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
    """Create a HuggingFace Dataset and deterministic label slot list.

    Supports optional data extraction (sampling) before dataset creation.

    Args:
        base_table: Full preprocessed DataFrame
        params: Configuration dict with:
            - label_column: Label column name
            - max_classes: Maximum number of label slots
            - extract_ratio: Optional fraction of data to extract (0 < ratio < 1)
            - extract_seed: Random seed for extraction
            - stratify_extract: Whether to stratify sampling by label

    Returns:
        - HuggingFace Dataset
        - List of label names (real labels + dummy placeholders)
    """
    label_column = params.get("label_column", "acct_code")
    max_classes = params.get("max_classes", 100)
    extract_ratio = params.get("extract_ratio")
    extract_seed = params.get("extract_seed", 42)
    stratify_extract = params.get("stratify_extract", True)

    if label_column not in base_table.columns:
        raise ValueError(f"Label column '{label_column}' not present in base table")

    # Optional: Extract subset of data
    if extract_ratio and 0 < extract_ratio < 1:
        original_size = len(base_table)

        if stratify_extract:
            # Stratified sampling to maintain label distribution
            base_table = base_table.groupby(label_column, group_keys=False).apply(
                lambda x: x.sample(frac=extract_ratio, random_state=extract_seed)
            ).reset_index(drop=True)
        else:
            # Random sampling
            sample_size = int(original_size * extract_ratio)
            base_table = base_table.sample(n=sample_size, random_state=extract_seed).reset_index(drop=True)

        logger.info(
            "Extracted %s rows (%.1f%%) from %s total rows",
            len(base_table), extract_ratio * 100, original_size
        )

    cleaned = base_table.reset_index(drop=True)
    dataset = Dataset.from_pandas(cleaned, preserve_index=False)

    unique_labels = cleaned[label_column].astype(str).drop_duplicates().tolist()
    names = _initialize_label_slots(max_classes)
    names = _upsert_labels_into_slots(names, unique_labels)

    logger.info("Dataset created with %s rows and %s columns", dataset.num_rows, len(dataset.column_names))
    logger.info("Real labels registered: %s (max slots: %s)", len(unique_labels), max_classes)

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
    num_proc: int = 1,
) -> DatasetDict:
    """Attach integer labels and ClassLabel schema to DatasetDict splits.

    Labels not found in `names` fall back to the first label (index 0).
    The first label is typically the first dummy label (e.g., 'dummy_1').
    """
    if label_col not in splits["train"].column_names:
        raise ValueError(f"Label column '{label_col}' not present in dataset")

    label2id = make_label2id(names)
    fallback = 0  # First label in names (typically dummy_1)

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

    # Store label metadata (names list already contains all label information)
    encoded.label_metadata = {
        "label2id": label2id,
        "id2label": make_id2label(names),
        "names": names,
    }

    return encoded


def serialize_to_text(
    split_datasets: DatasetDict,
    serialization_params: Dict[str, Any]
) -> DatasetDict:
    """
    Serialize feature columns to text for NLP models.

    Args:
        split_datasets: DatasetDict with feature columns + labels
        serialization_params: Configuration dict with:
            - text_columns: List of columns to serialize ([] = all except labels, acct_code)
            - separator: String separator between items (default ", ")
            - key_value_separator: String separator between key and value (default ": ")
            - include_column_names: Whether to include "col: value" format
            - num_proc: Number of processes for parallel processing

    Returns:
        DatasetDict with 'text' and 'labels' columns

    Example:
        >>> serialized = serialize_to_text(
        ...     split_datasets=split_data,
        ...     serialization_params={"text_columns": [], "separator": "|", "key_value_separator": "="}
        ... )
        >>> print(serialized["train"][0]["text"][:100])
        'col1=value1|col2=value2|...'
    """
    text_columns = serialization_params.get("text_columns", [])
    separator = serialization_params.get("separator", ", ")
    key_value_separator = serialization_params.get("key_value_separator", ": ")
    include_column_names = serialization_params.get("include_column_names", True)
    num_proc = serialization_params.get("num_proc", 4)

    if not text_columns:
        text_columns = [
            col for col in split_datasets["train"].column_names
            if col not in ["labels", "acct_code"]
        ]

    def serialize_function(examples):
        # Convert batch dict to DataFrame for vectorized operations
        df = pd.DataFrame({col: examples[col] for col in text_columns})

        # Pattern-based column formatting
        date_cols = [col for col in df.columns if 'date' in col.lower()]
        amount_cols = [col for col in df.columns if 'amount' in col.lower()]

        # 1. Format date columns: Int64 → "YYYY-MM-DD"
        for col in date_cols:
            try:
                df[col] = (
                    pd.to_datetime(df[col].astype("Int64"), format="%Y%m%d", errors="coerce")
                      .dt.strftime("%Y-%m-%d")
                )
            except Exception:
                pass  # Keep original if conversion fails

        # 2. Format amount columns: float → int → string (no ".0")
        for col in amount_cols:
            try:
                numeric = pd.to_numeric(df[col], errors="coerce")
                df[col] = numeric.round(0).astype("Int64")
            except Exception:
                pass  # Keep original if conversion fails

        # 3. Normalize missing placeholders across all columns
        df = df.astype("string")
        df = df.fillna(MISSING_TOKEN)
        df = df.replace(
            {
                "": MISSING_TOKEN,
                "nan": MISSING_TOKEN,
                "NaN": MISSING_TOKEN,
                "None": MISSING_TOKEN,
                "<NA>": MISSING_TOKEN,
            }
        )

        # 4. Build text strings
        if include_column_names:
            # Vectorized string formatting with column names
            texts = df.apply(
                lambda row: separator.join([
                    f"{col}{key_value_separator}{str(row[col])}" for col in text_columns
                ]),
                axis=1
            ).tolist()
        else:
            # Optimized vectorized join without column names
            texts = df.astype(str).agg(separator.join, axis=1).tolist()

        return {"text": texts}

    logger.info("Serializing feature columns to text...")
    serialized = split_datasets.map(
        serialize_function,
        batched=True,
        num_proc=num_proc,
        remove_columns=[c for c in split_datasets["train"].column_names if c not in ["labels"]],
        desc="Serializing to text"
    )

    # Log statistics
    for split_name in serialized.keys():
        split_data = serialized[split_name]
        text_lengths = [len(text) for text in split_data["text"]]
        avg_len = np.mean(text_lengths)
        max_len = max(text_lengths)
        logger.info(f"{split_name}: {len(split_data)} samples, avg text chars: {avg_len:.1f}, max: {max_len}")
        if len(split_data) > 0:
            logger.debug(f"Sample text: {split_data[0]['text'][:200]}...")

    return serialized
