"""
Training pipeline nodes implementing BlueScreen design specification.

This module provides a clean, streamlined implementation for DeepSpeed Stage 2 + LoRA
training using HuggingFace Trainer. The design follows the account-tax project philosophy:
대칭화 (Pattern), 모듈화 (Modularity), 순서화 (Ordering).

Architecture:
    tokenize_datasets → load_model → apply_lora_adapters → build_training_arguments
    → train_model → save_training_metrics

Key Features:
    - DeepSpeed Stage 2 support (dict-based config)
    - LoRA optimization via PEFT
    - MLflow auto-logging via Trainer (report_to=["mlflow"])
    - Custom callbacks for speed and memory tracking
    - Rank 0 logging for distributed training
"""

import logging
import os
import time
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import numpy as np
import torch
from datasets import ClassLabel, DatasetDict
from evaluate import load as load_metric
from peft import LoraConfig, TaskType, get_peft_model
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    DataCollatorWithPadding,
    EvalPrediction,
    Trainer,
    TrainerCallback,
    TrainerControl,
    TrainerState,
    TrainingArguments,
)
from transformers.integrations import MLflowCallback

logger = logging.getLogger(__name__)


# =============================================================================
# Helper Functions
# =============================================================================


def is_rank0() -> bool:
    """
    Check if current process is rank 0 in distributed training.

    Returns:
        True if rank 0 or single-process, False otherwise.
    """
    rank = int(os.environ.get("RANK", os.environ.get("LOCAL_RANK", 0)))
    return rank == 0


def compute_metrics(eval_pred: EvalPrediction) -> Dict[str, float]:
    """
    Compute evaluation metrics for classification.

    Uses HuggingFace evaluate library to compute accuracy, F1, precision, recall.

    Args:
        eval_pred: EvalPrediction object containing predictions and labels.

    Returns:
        Dictionary with metrics: accuracy, f1, precision, recall.

    Example:
        >>> metrics = compute_metrics(eval_pred)
        >>> print(metrics)
        {'accuracy': 0.95, 'f1': 0.94, 'precision': 0.93, 'recall': 0.95}
    """
    predictions, labels = eval_pred.predictions, eval_pred.label_ids

    # Handle logits (take argmax)
    if predictions.ndim > 1:
        predictions = np.argmax(predictions, axis=1)

    # Load metrics
    accuracy_metric = load_metric("accuracy")
    f1_metric = load_metric("f1")
    precision_metric = load_metric("precision")
    recall_metric = load_metric("recall")

    # Compute metrics
    accuracy = accuracy_metric.compute(predictions=predictions, references=labels)
    f1 = f1_metric.compute(predictions=predictions, references=labels, average="weighted")
    precision = precision_metric.compute(predictions=predictions, references=labels, average="weighted")
    recall = recall_metric.compute(predictions=predictions, references=labels, average="weighted")

    return {
        "accuracy": accuracy["accuracy"],
        "f1": f1["f1"],
        "precision": precision["precision"],
        "recall": recall["recall"],
    }


# =============================================================================
# Callback Classes
# =============================================================================


class SpeedCallback(TrainerCallback):
    """
    Callback to measure and log training speed (tokens/sec).

    Logs to MLflow at each step from rank 0 only.
    """

    def __init__(self):
        self.start_time = None
        self.total_tokens = 0

    def on_step_begin(self, args, state, control, **kwargs):
        """Record step start time."""
        if is_rank0():
            self.start_time = time.time()

    def on_step_end(self, args, state, control, **kwargs):
        """Calculate and log tokens/sec."""
        if is_rank0() and self.start_time is not None:
            elapsed = time.time() - self.start_time

            # Estimate tokens per step (batch_size * seq_len)
            # Using max_length as approximation
            batch_size = args.per_device_train_batch_size * args.gradient_accumulation_steps
            # Default sequence length estimate (will be overridden by actual data)
            seq_len = 512  # Conservative estimate
            tokens_per_step = batch_size * seq_len
            tokens_per_sec = tokens_per_step / elapsed if elapsed > 0 else 0

            # Log to MLflow via Trainer's internal logger
            if hasattr(state, 'log_history'):
                state.log_history.append({
                    "speed/tokens_per_sec": tokens_per_sec,
                    "step": state.global_step
                })

            self.start_time = None


class TorchMemoryCallback(TrainerCallback):
    """
    Callback to monitor and log GPU memory usage.

    Logs allocated and max allocated memory to MLflow from rank 0 only.
    """

    def on_step_end(self, args, state, control, **kwargs):
        """Log GPU memory after each step."""
        if is_rank0() and torch.cuda.is_available():
            allocated = torch.cuda.memory_allocated(0) / 1024**2  # MB
            max_allocated = torch.cuda.max_memory_allocated(0) / 1024**2  # MB

            if hasattr(state, 'log_history'):
                state.log_history.append({
                    "gpu/alloc_mb": allocated,
                    "gpu/max_alloc_mb": max_allocated,
                    "step": state.global_step
                })

    def on_evaluate(self, args, state, control, **kwargs):
        """Log GPU memory after evaluation."""
        if is_rank0() and torch.cuda.is_available():
            allocated = torch.cuda.memory_allocated(0) / 1024**2  # MB
            max_allocated = torch.cuda.max_memory_allocated(0) / 1024**2  # MB

            if hasattr(state, 'log_history'):
                state.log_history.append({
                    "gpu/alloc_mb_eval": allocated,
                    "gpu/max_alloc_mb_eval": max_allocated,
                    "step": state.global_step
                })


# =============================================================================
# Core Training Nodes
# =============================================================================


def tokenize_datasets(
    serialized_datasets: DatasetDict,
    tokenization_params: Dict[str, Any]
) -> Tuple[DatasetDict, Dict[str, Any]]:
    """
    Tokenize text datasets and analyze token lengths with sampling.

    Args:
        serialized_datasets: DatasetDict with 'text' and 'labels' columns
        tokenization_params: Configuration dict with:
            - model_name: HuggingFace model name for tokenizer
            - max_length: Maximum sequence length
            - truncation: Whether to truncate sequences
            - padding: Whether to pad (False for dynamic padding)
            - num_proc: Number of processes for parallel tokenization
            - sample_size: Number of samples to extract per split (default: 0)

    Returns:
        Tuple of (tokenized_datasets, token_length_report) where:
            - tokenized_datasets: DatasetDict with input_ids, attention_mask, labels, length
            - token_length_report: Dict with token length statistics and optional samples

    Example:
        >>> tokenized, report = tokenize_datasets(
        ...     serialized_datasets=serialized_data,
        ...     tokenization_params={"model_name": "bert-base-uncased", "max_length": 512, "sample_size": 10}
        ... )
        >>> print(report["overall"]["mean"])
        245.3
        >>> print(report["samples"]["train"][0]["tokens"])
        ['hello', 'world']
    """
    model_name = tokenization_params["model_name"]
    max_length = tokenization_params["max_length"]
    truncation = tokenization_params["truncation"]
    padding = tokenization_params["padding"]
    num_proc = tokenization_params.get("num_proc", 4)
    sample_size = tokenization_params.get("sample_size", 0)

    logger.info(f"Loading tokenizer: {model_name}")
    tokenizer = AutoTokenizer.from_pretrained(
        model_name,
        trust_remote_code=True
    )

    # Ensure pad token exists
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id
        logger.info(f"Set pad_token to eos_token: {tokenizer.pad_token}")

    # Extract samples BEFORE tokenization (if requested)
    samples = {}
    if sample_size > 0:
        logger.info(f"Extracting {sample_size} samples per split for inspection...")
        for split_name in serialized_datasets.keys():
            n_samples = min(sample_size, len(serialized_datasets[split_name]))
            sample_ds = serialized_datasets[split_name].shuffle(seed=42).select(range(n_samples))

            samples[split_name] = []
            for ex in sample_ds:
                # Use tokenizer.tokenize() to get token strings directly
                tokens = tokenizer.tokenize(ex["text"], truncation=truncation, max_length=max_length)
                samples[split_name].append({
                    "text": ex["text"],
                    "labels": ex["labels"],
                    "tokens": tokens,
                    "token_count": len(tokens),
                })

            logger.info(f"Extracted {len(samples[split_name])} samples from {split_name}")

    def tokenize_function(examples):
        """Batch tokenization with length tracking."""
        return tokenizer(
            examples["text"],
            truncation=truncation,
            max_length=max_length,
            padding=padding,
            return_length=True,  # Automatically compute token lengths
        )

    logger.info("Tokenizing text datasets...")
    tokenized_datasets = serialized_datasets.map(
        tokenize_function,
        batched=True,
        num_proc=num_proc,
        remove_columns=["text"],  # Remove text to save memory
        desc="Tokenizing"
    )

    # Analyze token lengths using Arrow -> NumPy (zero-copy)
    logger.info("Analyzing token lengths...")
    all_lengths = []
    per_split_stats = {}

    for split_name in tokenized_datasets.keys():
        split_data = tokenized_datasets[split_name]
        # Use the 'length' column from return_length=True
        lengths = np.array(split_data["length"])
        all_lengths.append(lengths)

        per_split_stats[split_name] = {
            "count": int(lengths.size),
            "mean": float(lengths.mean()),
            "max": int(lengths.max()),
            "min": int(lengths.min()),
            "p95": float(np.percentile(lengths, 95)),
        }

        logger.info(
            f"{split_name}: {len(split_data)} samples, "
            f"avg tokens: {per_split_stats[split_name]['mean']:.1f}, "
            f"max: {per_split_stats[split_name]['max']}"
        )

    # Concatenate all lengths efficiently (NumPy arrays)
    all_lengths = np.concatenate(all_lengths)

    # Overall statistics with percentiles
    percentiles = [50, 75, 90, 95, 99]
    overall_stats = {
        "count": int(all_lengths.size),
        "mean": float(all_lengths.mean()),
        "max": int(all_lengths.max()),
        "min": int(all_lengths.min()),
    }

    for p in percentiles:
        overall_stats[f"p{p}"] = float(np.percentile(all_lengths, p))

    token_length_report = {
        "overall": overall_stats,
        "per_split": per_split_stats,
        "model_name": model_name,
        "max_length_config": max_length,
    }

    # Add samples to report if available
    if samples:
        token_length_report["samples"] = samples
        logger.info(f"Sample extraction complete. Included in token_length_report.")

    # MLflow metrics will be logged via hooks when token_length_report is saved to catalog
    return tokenized_datasets, token_length_report


def load_model(
    tokenized_datasets: DatasetDict,
    model_name: str,
    num_labels: Optional[int] = None,
    torch_dtype: str = "bfloat16",
    device_map: str = "auto",
    gradient_checkpointing: bool = True,
) -> Any:
    """
    Load pre-trained model for sequence classification.

    Args:
        tokenized_datasets: Tokenized data for inferring num_labels.
        model_name: HuggingFace model name.
        num_labels: Number of classification labels (inferred if None).
        torch_dtype: Precision type ("float16", "bfloat16", "float32").
        device_map: Device mapping strategy ("auto" or specific).
        gradient_checkpointing: Whether to enable gradient checkpointing.

    Returns:
        Loaded and configured model.

    Example:
        >>> model = load_model(
        ...     tokenized_datasets=tokenized_data,
        ...     model_name="bert-base-uncased",
        ...     gradient_checkpointing=True
        ... )
        >>> print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    """
    # Infer num_labels from data if not provided
    if num_labels is None:
        try:
            labels_feature = tokenized_datasets["train"].features.get("labels")
            if isinstance(labels_feature, ClassLabel):
                num_labels = labels_feature.num_classes
                id2label = {i: labels_feature.int2str(i) for i in range(num_labels)}
                label2id = {v: k for k, v in id2label.items()}
                logger.info(f"Inferred num_labels={num_labels} from ClassLabel feature")
            else:
                unique_labels = set(tokenized_datasets["train"]["labels"])
                num_labels = len(unique_labels)
                id2label = {i: str(i) for i in range(num_labels)}
                label2id = {str(i): i for i in range(num_labels)}
                logger.info(f"Inferred num_labels={num_labels} from unique labels")
        except Exception as e:
            logger.warning(f"Could not infer num_labels: {e}. Using default=2")
            num_labels = 2
            id2label = {0: "0", 1: "1"}
            label2id = {"0": 0, "1": 1}
    else:
        id2label = {i: str(i) for i in range(num_labels)}
        label2id = {str(i): i for i in range(num_labels)}

    # Map string dtype to torch dtype
    dtype_map = {
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
        "float32": torch.float32,
    }
    torch_dtype_obj = dtype_map.get(torch_dtype, torch.bfloat16)

    logger.info(f"Loading model: {model_name} with {num_labels} labels, dtype={torch_dtype}")
    model = AutoModelForSequenceClassification.from_pretrained(
        model_name,
        num_labels=num_labels,
        id2label=id2label,
        label2id=label2id,
        problem_type="single_label_classification",
        trust_remote_code=True,
        torch_dtype=torch_dtype_obj,
        device_map=device_map if torch.cuda.is_available() else None,
        low_cpu_mem_usage=True,
    )

    # Enable gradient checkpointing
    if gradient_checkpointing and hasattr(model, "gradient_checkpointing_enable"):
        model.gradient_checkpointing_enable()
        logger.info("Gradient checkpointing enabled")

    # Log parameter count
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"Model loaded: {total_params:,} total params, {trainable_params:,} trainable")

    return model


def apply_lora_adapters(
    model: Any,
    lora_r: int = 8,
    lora_alpha: int = 32,
    target_modules: list = None,
    lora_dropout: float = 0.05,
    bias: str = "none",
    task_type: str = "SEQ_CLS",
) -> Any:
    """
    Apply LoRA (Low-Rank Adaptation) to model using PEFT.

    Args:
        model: Pre-trained model.
        lora_r: LoRA rank (lower = fewer parameters).
        lora_alpha: LoRA scaling factor.
        target_modules: Modules to apply LoRA (e.g., ["query", "value"]).
        lora_dropout: Dropout rate for LoRA layers.
        bias: Bias training strategy ("none", "all", "lora_only").
        task_type: Task type for PEFT ("SEQ_CLS", "CAUSAL_LM", etc.).

    Returns:
        Model with LoRA adapters applied.

    Example:
        >>> lora_model = apply_lora_adapters(
        ...     model=base_model,
        ...     lora_r=8,
        ...     lora_alpha=32,
        ...     target_modules=["query", "value"]
        ... )
        >>> # Output: trainable params: 294,912 || all params: 109,483,778 || trainable%: 0.27%
    """
    if target_modules is None:
        target_modules = ["query", "value"]

    logger.info(f"Applying LoRA: r={lora_r}, alpha={lora_alpha}, targets={target_modules}")

    lora_config = LoraConfig(
        r=lora_r,
        lora_alpha=lora_alpha,
        target_modules=target_modules,
        lora_dropout=lora_dropout,
        bias=bias,
        task_type=TaskType[task_type]
    )

    model = get_peft_model(model, lora_config)

    # Print trainable parameters (rank 0 only)
    if is_rank0():
        model.print_trainable_parameters()
        # Output example:
        # trainable params: 294,912 || all params: 109,483,778 || trainable%: 0.27%

    return model


def build_training_arguments(
    output_dir: str,
    num_train_epochs: int = 3,
    per_device_train_batch_size: int = 8,
    per_device_eval_batch_size: int = 16,
    gradient_accumulation_steps: int = 4,
    learning_rate: float = 2e-5,
    weight_decay: float = 0.01,
    warmup_ratio: float = 0.1,
    eval_strategy: str = "steps",
    eval_steps: int = 500,
    save_strategy: str = "steps",
    save_steps: int = 1000,
    save_total_limit: int = 3,
    logging_steps: int = 10,
    fp16: bool = False,
    bf16: bool = True,
    gradient_checkpointing: bool = True,
    dataloader_num_workers: int = 4,
    seed: int = 42,
    load_best_model_at_end: bool = True,
    metric_for_best_model: str = "f1",
    greater_is_better: bool = True,
    deepspeed_config: Optional[Dict] = None,
) -> TrainingArguments:
    """
    Build TrainingArguments with DeepSpeed configuration.

    Args:
        output_dir: Directory for saving checkpoints.
        num_train_epochs: Number of training epochs.
        per_device_train_batch_size: Batch size per GPU for training.
        per_device_eval_batch_size: Batch size per GPU for evaluation.
        gradient_accumulation_steps: Gradient accumulation steps.
        learning_rate: Learning rate.
        weight_decay: Weight decay for regularization.
        warmup_ratio: Warmup ratio for learning rate scheduler.
        eval_strategy: Evaluation strategy ("steps" or "epoch").
        eval_steps: Evaluation frequency in steps.
        save_strategy: Save strategy ("steps" or "epoch").
        save_steps: Save frequency in steps.
        save_total_limit: Maximum number of checkpoints to keep.
        logging_steps: Logging frequency in steps.
        fp16: Whether to use fp16 mixed precision.
        bf16: Whether to use bf16 mixed precision (recommended for A100/H100).
        gradient_checkpointing: Whether to use gradient checkpointing.
        dataloader_num_workers: Number of workers for data loading.
        seed: Random seed.
        load_best_model_at_end: Whether to load best model at end.
        metric_for_best_model: Metric to use for best model selection.
        greater_is_better: Whether higher metric is better.
        deepspeed_config: DeepSpeed configuration dict (None to disable).

    Returns:
        TrainingArguments object.

    Example:
        >>> args = build_training_arguments(
        ...     output_dir="checkpoints",
        ...     num_train_epochs=3,
        ...     deepspeed_config={"zero_optimization": {"stage": 2}}
        ... )
        >>> print(f"Report to: {args.report_to}")
        ['mlflow']
    """
    logger.info(f"Building TrainingArguments: epochs={num_train_epochs}, lr={learning_rate}")

    training_args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=num_train_epochs,
        per_device_train_batch_size=per_device_train_batch_size,
        per_device_eval_batch_size=per_device_eval_batch_size,
        gradient_accumulation_steps=gradient_accumulation_steps,
        learning_rate=learning_rate,
        weight_decay=weight_decay,
        warmup_ratio=warmup_ratio,
        evaluation_strategy=eval_strategy,
        eval_steps=eval_steps if eval_strategy == "steps" else None,
        save_strategy=save_strategy,
        save_steps=save_steps if save_strategy == "steps" else None,
        save_total_limit=save_total_limit,
        logging_steps=logging_steps,
        fp16=fp16,
        bf16=bf16,
        gradient_checkpointing=gradient_checkpointing,
        dataloader_num_workers=dataloader_num_workers,
        seed=seed,
        report_to=["mlflow"],  # Auto-log to MLflow
        load_best_model_at_end=load_best_model_at_end,
        metric_for_best_model=metric_for_best_model,
        greater_is_better=greater_is_better,
        deepspeed=deepspeed_config,  # Pass dict directly (no JSON file needed)
        logging_first_step=True,
        logging_nan_inf_filter=True,
        save_safetensors=True,
    )

    if deepspeed_config:
        logger.info("DeepSpeed configuration provided (will be passed to Trainer)")

    return training_args


def train_model(
    model: Any,
    tokenized_datasets: DatasetDict,
    tokenizer: Any,
    training_args: TrainingArguments,
) -> Dict[str, Any]:
    """
    Train model using HuggingFace Trainer with callbacks.

    Args:
        model: Model to train (with or without LoRA).
        tokenized_datasets: Tokenized datasets (train/valid/test).
        tokenizer: Tokenizer for data collation.
        training_args: TrainingArguments object.

    Returns:
        Training metrics dictionary containing:
            - train_loss: Final training loss
            - train_runtime: Total training time
            - eval_accuracy, eval_f1, eval_precision, eval_recall: Best eval metrics

    Example:
        >>> metrics = train_model(
        ...     model=lora_model,
        ...     tokenized_datasets=tokenized_data,
        ...     tokenizer=tokenizer,
        ...     training_args=args
        ... )
        >>> print(f"Train loss: {metrics['train_loss']:.4f}")
    """
    logger.info("Initializing Trainer...")

    # Data collator for dynamic padding
    data_collator = DataCollatorWithPadding(
        tokenizer=tokenizer,
        pad_to_multiple_of=8  # For tensor core efficiency
    )

    # Initialize callbacks
    callbacks = [
        SpeedCallback(),
        TorchMemoryCallback(),
    ]

    # Create Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_datasets["train"],
        eval_dataset=tokenized_datasets.get("valid", tokenized_datasets.get("validation")),
        data_collator=data_collator,
        tokenizer=tokenizer,
        compute_metrics=compute_metrics,
        callbacks=callbacks,
    )

    logger.info("Starting training...")
    train_result = trainer.train()

    # Evaluate on best checkpoint
    logger.info("Evaluating model...")
    eval_result = trainer.evaluate()

    # Save final model (rank 0 only)
    if is_rank0():
        final_path = Path(training_args.output_dir) / "final_model"
        trainer.save_model(str(final_path))
        tokenizer.save_pretrained(str(final_path))
        logger.info(f"Model saved to {final_path}")

    return {
        "train_loss": train_result.metrics.get("train_loss"),
        "train_runtime": train_result.metrics.get("train_runtime"),
        "train_samples_per_second": train_result.metrics.get("train_samples_per_second"),
        "eval_loss": eval_result.get("eval_loss"),
        "eval_accuracy": eval_result.get("eval_accuracy"),
        "eval_f1": eval_result.get("eval_f1"),
        "eval_precision": eval_result.get("eval_precision"),
        "eval_recall": eval_result.get("eval_recall"),
    }


def save_training_metrics(
    metrics: Dict[str, Any],
) -> Dict[str, Any]:
    """
    Save training metrics for MLflow logging.

    This is a pass-through node that ensures metrics are properly formatted
    and logged to the catalog as training_metrics.

    Args:
        metrics: Raw training metrics from train_model.

    Returns:
        Formatted metrics dictionary.

    Example:
        >>> formatted = save_training_metrics(metrics)
        >>> # Metrics will be logged to data/08_reporting/training_metrics.json
    """
    logger.info("Saving training metrics...")

    # Format metrics for MLflow
    formatted_metrics = {
        "training": {
            "loss": metrics.get("train_loss"),
            "runtime_seconds": metrics.get("train_runtime"),
            "samples_per_second": metrics.get("train_samples_per_second"),
        },
        "evaluation": {
            "loss": metrics.get("eval_loss"),
            "accuracy": metrics.get("eval_accuracy"),
            "f1": metrics.get("eval_f1"),
            "precision": metrics.get("eval_precision"),
            "recall": metrics.get("eval_recall"),
        }
    }

    if is_rank0():
        logger.info(f"Training completed - Loss: {metrics.get('train_loss'):.4f}, "
                   f"F1: {metrics.get('eval_f1', 0):.4f}")

    return formatted_metrics