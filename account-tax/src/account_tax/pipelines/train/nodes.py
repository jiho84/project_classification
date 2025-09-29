"""Training pipeline nodes for NLP models."""

import logging
from typing import Dict, Any
from datasets import DatasetDict, ClassLabel
from transformers import AutoTokenizer, DataCollatorWithPadding

logger = logging.getLogger(__name__)


def tokenize_datasets(
    dataset_dict: DatasetDict,
    params: Dict[str, Any]
) -> DatasetDict:
    """
    Tokenize text data for transformer models.

    Pipeline Order: Second step in training pipeline
    Role: Convert text to token IDs

    Args:
        dataset_dict: DatasetDict with 'text' column
        params: Tokenization parameters:
            - model_name: HuggingFace model name
            - max_length: Maximum sequence length
            - truncation: Whether to truncate
            - num_proc: Number of processes for parallel tokenization

    Returns:
        DatasetDict with input_ids, attention_mask columns
    """
    model_name = params["model_name"]
    max_length = params.get("max_length", 512)
    truncation = params.get("truncation", True)
    num_proc = params.get("num_proc", 4)

    # Load tokenizer
    logger.info(f"Loading tokenizer: {model_name}")
    tokenizer = AutoTokenizer.from_pretrained(
        model_name,
        trust_remote_code=True
    )

    # Ensure pad token is set
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    def tokenize_function(examples):
        """Batch tokenization."""
        outputs = tokenizer(
            examples["text"],
            truncation=truncation,
            max_length=max_length,
            # Don't pad here, use DataCollator for dynamic padding
        )
        # Preserve labels
        outputs["labels"] = examples["labels"]
        return outputs

    # Remove text column after tokenization to save memory
    tokenized_datasets = dataset_dict.map(
        tokenize_function,
        batched=True,
        num_proc=num_proc,
        remove_columns=["text"],
        desc="Tokenizing"
    )

    # Log statistics
    for split_name in ["train", "valid", "test"]:
        split_data = tokenized_datasets[split_name]
        logger.info(f"{split_name}: {len(split_data)} samples tokenized")

    # Store tokenizer info for later use
    tokenized_datasets.tokenizer_name = model_name
    tokenized_datasets.max_length = max_length

    return tokenized_datasets


def prepare_for_trainer(
    dataset_dict: DatasetDict,
    params: Dict[str, Any]
) -> Dict[str, Any]:
    """
    Final preparation for HuggingFace Trainer.

    Pipeline Order: Final step in training pipeline
    Role: Format data and create collator for Trainer

    Args:
        dataset_dict: Tokenized DatasetDict
        params: Trainer preparation parameters

    Returns:
        Dictionary with:
            - datasets: DatasetDict with torch format
            - collator: DataCollatorWithPadding
            - model_config: Configuration for model initialization
    """
    model_name = params.get("model_name", dataset_dict.tokenizer_name)

    # Load tokenizer for collator
    tokenizer = AutoTokenizer.from_pretrained(
        model_name,
        trust_remote_code=True
    )

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Set format to PyTorch tensors
    dataset_dict = dataset_dict.with_format(
        "torch",
        columns=["input_ids", "attention_mask", "labels"]
    )

    # Create data collator for dynamic padding
    collator = DataCollatorWithPadding(
        tokenizer=tokenizer,
        pad_to_multiple_of=8  # For tensor cores efficiency
    )

    # Prepare model configuration
    # Prefer deriving from Dataset features (ClassLabel)
    names = None
    try:
        labels_feature = dataset_dict["train"].features.get("labels")
        if isinstance(labels_feature, ClassLabel):
            names = list(labels_feature.names)
    except Exception:
        names = None

    if names:
        num_labels = len(names)
        id2label = {i: n for i, n in enumerate(names)}
        label2id = {n: i for i, n in enumerate(names)}
    elif hasattr(dataset_dict, 'label_metadata'):
        label_metadata = dataset_dict.label_metadata
        id2label = label_metadata.get("id2label")
        label2id = label_metadata.get("label2id")
        num_labels = label_metadata.get("max_classes", len(id2label) if id2label else 0)
    else:
        # Fallback if metadata not available
        num_labels = len(set(dataset_dict["train"]["labels"].tolist()))
        id2label = {i: str(i) for i in range(num_labels)}
        label2id = {str(i): i for i in range(num_labels)}

    model_config = {
        "model_name": model_name,
        "num_labels": num_labels,
        "id2label": id2label,
        "label2id": label2id
    }

    logger.info(f"Prepared data for Trainer:")
    logger.info(f"  - Model: {model_name}")
    logger.info(f"  - Num labels: {num_labels}")
    logger.info(f"  - Train samples: {len(dataset_dict['train'])}")
    logger.info(f"  - Valid samples: {len(dataset_dict['valid'])}")
    logger.info(f"  - Test samples: {len(dataset_dict['test'])}")

    return {
        "datasets": dataset_dict,
        "collator": collator,
        "model_config": model_config
    }
