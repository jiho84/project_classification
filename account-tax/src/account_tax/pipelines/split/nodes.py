"""Simplified dataset preparation nodes."""

from __future__ import annotations

import logging
import random
from typing import Dict, List, Tuple, Any

import numpy as np
import pandas as pd
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


def prepare_text_fields(
    split_datasets: DatasetDict,
    serialization_params: Dict[str, Any],
) -> DatasetDict:
    """Format numeric/date columns and normalize missing values as strings.

    Args:
        split_datasets: DatasetDict prior to text serialization
        serialization_params: Same configuration dict used for serialization

    Returns:
        DatasetDict with feature columns converted to normalized strings
    """
    text_columns = serialization_params.get("text_columns", [])
    num_proc = serialization_params.get("num_proc", 4)

    if not text_columns:
        text_columns = [
            col for col in split_datasets["train"].column_names
            if col not in ["labels", "acct_code"]
        ]

    date_cols = [col for col in text_columns if "date" in col.lower()]
    amount_cols = [col for col in text_columns if "amount" in col.lower()]

    def format_amount_series(series: pd.Series) -> pd.Series:
        numeric = pd.to_numeric(series, errors="coerce")
        sign = np.sign(numeric.fillna(0)).astype(int)
        exponent = np.zeros(len(numeric), dtype=int)
        mant = np.zeros(len(numeric), dtype=float)

        nonzero_mask = numeric.notna() & (numeric != 0)
        abs_vals = numeric[nonzero_mask].abs()
        if not abs_vals.empty:
            exponents = np.floor(np.log10(abs_vals)).astype(int)
            exponent[nonzero_mask] = exponents
            mant[nonzero_mask] = abs_vals / np.power(10.0, exponents)

        mant = np.round(mant, 4)

        result = np.full(len(numeric), None, dtype=object)
        valid_idx = np.where(numeric.notna())[0]
        for idx in valid_idx:
            result[idx] = (
                f"sign={sign[idx]}| "
                f"mant={mant[idx]:.4f}| "
                f"exponent={exponent[idx]}"
            )

        return pd.Series(result, index=series.index)

    def format_batch(examples: Dict[str, Any]) -> Dict[str, Any]:
        formatted = dict(examples)
        df = pd.DataFrame({col: examples[col] for col in text_columns})

        for col in date_cols:
            try:
                df[col] = (
                    pd.to_datetime(df[col].astype("Int64"), format="%Y%m%d", errors="coerce")
                      .dt.strftime("%Y-%m-%d")
                )
            except Exception:
                pass  # Keep original if conversion fails

        for col in amount_cols:
            try:
                df[col] = format_amount_series(df[col])
            except Exception:
                pass

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

        for col in text_columns:
            formatted[col] = df[col].tolist()

        return formatted

    logger.info("Formatting text fields prior to serialization...")
    formatted_datasets = split_datasets.map(
        format_batch,
        batched=True,
        num_proc=num_proc,
        desc="Formatting text fields",
    )

    return formatted_datasets


def serialize_to_text(
    split_datasets: DatasetDict,
    serialization_params: Dict[str, Any]
) -> DatasetDict:
    """Join prepared feature columns into text strings."""

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

    def join_batch(examples: Dict[str, Any]) -> Dict[str, Any]:
        if include_column_names:
            texts = [
                separator.join(
                    f"{col}{key_value_separator}{examples[col][i]}" for col in text_columns
                )
                for i in range(len(examples[text_columns[0]]))
            ]
        else:
            texts = [
                separator.join(examples[col][i] for col in text_columns)
                for i in range(len(examples[text_columns[0]]))
            ]
        result = dict(examples)
        result["text"] = texts
        return result

    logger.info("Serializing formatted feature columns to text...")
    serialized = split_datasets.map(
        join_batch,
        batched=True,
        num_proc=num_proc,
        remove_columns=[c for c in split_datasets["train"].column_names if c not in ["labels"]],
        desc="Serializing to text",
    )

    for split_name in serialized.keys():
        split_data = serialized[split_name]
        text_lengths = [len(text) for text in split_data["text"]]
        avg_len = np.mean(text_lengths)
        max_len = max(text_lengths)
        logger.info(
            "%s: %s samples, avg text chars: %.1f, max: %s",
            split_name,
            len(split_data),
            avg_len,
            max_len,
        )
        if len(split_data) > 0:
            logger.debug("Sample text: %s...", split_data[0]["text"][:200])

    return serialized
