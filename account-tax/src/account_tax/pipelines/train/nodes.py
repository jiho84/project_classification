"""Training pipeline nodes for NLP models."""

import logging
import json
import tempfile
from pathlib import Path
from typing import Dict, Any, Tuple, Optional, Union
import numpy as np
import torch
from datasets import DatasetDict, ClassLabel
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    DataCollatorWithPadding,
    Trainer,
    TrainingArguments,
    EvalPrediction,
    set_seed
)
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, confusion_matrix
from peft import LoraConfig, get_peft_model, TaskType

logger = logging.getLogger(__name__)


def tokenize_datasets(
    dataset_dict: DatasetDict,
    params: Dict[str, Any]
) -> Tuple[DatasetDict, Dict[str, Any]]:
    """
    Tokenize text data for transformer models.

    Args:
        dataset_dict: DatasetDict with 'text' column from split pipeline
        params: Tokenization parameters:
            - model_name: HuggingFace model name
            - max_length: Maximum sequence length
            - truncation: Whether to truncate
            - padding: Padding strategy
            - num_proc: Number of processes for parallel tokenization

    Returns:
        Tuple of:
            - Tokenized DatasetDict with input_ids, attention_mask columns
            - Tokenization metadata for MLflow logging
    """
    model_name = params["model_name"]
    max_length = params.get("max_length", 512)
    truncation = params.get("truncation", True)
    padding = params.get("padding", False)
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
        tokenizer.pad_token_id = tokenizer.eos_token_id

    def tokenize_function(examples):
        """Batch tokenization."""
        outputs = tokenizer(
            examples["text"],
            truncation=truncation,
            max_length=max_length,
            padding=padding,  # Usually False for dynamic padding later
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

    # Compute token statistics
    token_stats = {}
    for split_name in ["train", "valid", "test"]:
        if split_name in tokenized_datasets:
            split_data = tokenized_datasets[split_name]
            token_lengths = [len(ids) for ids in split_data["input_ids"]]
            token_stats[f"{split_name}_samples"] = len(split_data)
            token_stats[f"{split_name}_avg_tokens"] = np.mean(token_lengths).item()
            token_stats[f"{split_name}_max_tokens"] = max(token_lengths)
            token_stats[f"{split_name}_min_tokens"] = min(token_lengths)
            logger.info(f"{split_name}: {len(split_data)} samples tokenized, avg tokens: {token_stats[f'{split_name}_avg_tokens']:.1f}")

    # Store tokenizer info for later use
    tokenized_datasets.tokenizer_name = model_name
    tokenized_datasets.max_length = max_length

    # Metadata for MLflow logging
    metadata = {
        "tokenizer_name": model_name,
        "max_length": max_length,
        "truncation": truncation,
        "padding": padding,
        "vocab_size": tokenizer.vocab_size,
        "pad_token": tokenizer.pad_token,
        **token_stats
    }

    return tokenized_datasets, metadata


def load_model(
    tokenized_data: DatasetDict,
    params: Dict[str, Any]
) -> Tuple[Any, Dict[str, Any]]:
    """
    Load and configure the pre-trained model.

    Args:
        tokenized_data: Tokenized DatasetDict for extracting label information
        params: Model configuration parameters:
            - name: Model name/path
            - num_labels: Number of classification labels (inferred if None)
            - problem_type: Classification problem type
            - trust_remote_code: Whether to trust remote code
            - torch_dtype: Model precision (float16, bfloat16, float32)
            - device_map: Device mapping strategy
            - low_cpu_mem_usage: Whether to use low CPU memory loading

    Returns:
        Tuple of:
            - Configured model ready for training
            - Model metadata for MLflow logging
    """
    model_name = params.get("name", tokenized_data.tokenizer_name)
    num_labels = params.get("num_labels")
    problem_type = params.get("problem_type", "single_label_classification")
    trust_remote_code = params.get("trust_remote_code", True)
    torch_dtype_str = params.get("torch_dtype", "bfloat16")
    device_map = params.get("device_map", "auto")
    low_cpu_mem_usage = params.get("low_cpu_mem_usage", True)

    # Convert string dtype to torch dtype
    dtype_mapping = {
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
        "float32": torch.float32,
    }
    torch_dtype = dtype_mapping.get(torch_dtype_str, torch.bfloat16)

    # Infer num_labels from data if not provided
    if num_labels is None:
        try:
            labels_feature = tokenized_data["train"].features.get("labels")
            if isinstance(labels_feature, ClassLabel):
                num_labels = labels_feature.num_classes
                id2label = {i: labels_feature.int2str(i) for i in range(num_labels)}
                label2id = {v: k for k, v in id2label.items()}
            else:
                unique_labels = set(tokenized_data["train"]["labels"])
                num_labels = len(unique_labels)
                id2label = {i: str(i) for i in range(num_labels)}
                label2id = {str(i): i for i in range(num_labels)}
        except Exception as e:
            logger.warning(f"Could not infer num_labels from data: {e}")
            num_labels = 2  # Binary classification default
            id2label = {0: "0", 1: "1"}
            label2id = {"0": 0, "1": 1}
    else:
        id2label = {i: str(i) for i in range(num_labels)}
        label2id = {str(i): i for i in range(num_labels)}

    logger.info(f"Loading model: {model_name} with {num_labels} labels")

    # Load model
    model = AutoModelForSequenceClassification.from_pretrained(
        model_name,
        num_labels=num_labels,
        id2label=id2label,
        label2id=label2id,
        problem_type=problem_type,
        trust_remote_code=trust_remote_code,
        torch_dtype=torch_dtype,
        device_map=device_map if torch.cuda.is_available() else None,
        low_cpu_mem_usage=low_cpu_mem_usage
    )

    # Enable gradient checkpointing if available
    if hasattr(model, "gradient_checkpointing_enable"):
        model.gradient_checkpointing_enable()
        logger.info("Gradient checkpointing enabled")

    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    # Model metadata for MLflow
    metadata = {
        "model_name": model_name,
        "num_labels": num_labels,
        "problem_type": problem_type,
        "torch_dtype": torch_dtype_str,
        "total_parameters": total_params,
        "trainable_parameters": trainable_params,
        "model_architecture": model.config.architectures[0] if hasattr(model.config, 'architectures') else "unknown",
        "hidden_size": model.config.hidden_size if hasattr(model.config, 'hidden_size') else None,
        "num_hidden_layers": model.config.num_hidden_layers if hasattr(model.config, 'num_hidden_layers') else None,
    }

    logger.info(f"Model loaded: {total_params:,} total parameters, {trainable_params:,} trainable")

    return model, metadata


def apply_optimization(
    model: Any,
    params: Dict[str, Any]
) -> Tuple[Any, Dict[str, Any]]:
    """
    Apply optimization techniques like LoRA or model compilation.

    Args:
        model: The loaded model
        params: Optimization parameters:
            - compile_model: Whether to compile with torch.compile
            - use_lora: Whether to apply LoRA
            - lora_config: LoRA configuration parameters

    Returns:
        Tuple of:
            - Optimized model
            - Optimization metadata for MLflow logging
    """
    compile_model = params.get("compile_model", False)
    use_lora = params.get("use_lora", False)
    lora_config_params = params.get("lora_config", {})

    metadata = {
        "optimizations_applied": []
    }

    # Apply LoRA if requested
    if use_lora:
        logger.info("Applying LoRA optimization")

        lora_config = LoraConfig(
            task_type=TaskType.SEQ_CLS,
            r=lora_config_params.get("r", 16),
            lora_alpha=lora_config_params.get("lora_alpha", 32),
            lora_dropout=lora_config_params.get("lora_dropout", 0.1),
            target_modules=lora_config_params.get("target_modules", ["q_proj", "v_proj"]),
            inference_mode=False,
        )

        model = get_peft_model(model, lora_config)

        # Count LoRA parameters
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        all_params = sum(p.numel() for p in model.parameters())
        trainable_percent = 100 * trainable_params / all_params

        metadata["optimizations_applied"].append("LoRA")
        metadata["lora_r"] = lora_config.r
        metadata["lora_alpha"] = lora_config.lora_alpha
        metadata["lora_dropout"] = lora_config.lora_dropout
        metadata["lora_trainable_params"] = trainable_params
        metadata["lora_trainable_percent"] = trainable_percent

        logger.info(f"LoRA applied: {trainable_params:,} trainable params ({trainable_percent:.2f}% of total)")
        model.print_trainable_parameters()

    # Apply torch.compile if requested (PyTorch 2.0+)
    if compile_model:
        if hasattr(torch, 'compile'):
            logger.info("Compiling model with torch.compile")
            model = torch.compile(model)
            metadata["optimizations_applied"].append("torch.compile")
        else:
            logger.warning("torch.compile not available in this PyTorch version")

    if not metadata["optimizations_applied"]:
        metadata["optimizations_applied"] = ["none"]

    return model, metadata


def prepare_trainer(
    model: Any,
    tokenized_data: DatasetDict,
    params: Dict[str, Any]
) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """
    Prepare HuggingFace Trainer with all configurations.

    Args:
        model: The optimized model
        tokenized_data: Tokenized datasets
        params: Trainer configuration including:
            - trainer: Training arguments
            - deepspeed: DeepSpeed configuration
            - evaluation: Evaluation settings
            - logging: Logging settings

    Returns:
        Tuple of:
            - Dictionary containing trainer components (trainer, args, collator, tokenizer)
            - Training configuration metadata for MLflow logging
    """
    trainer_params = params.get("trainer", {})
    deepspeed_params = params.get("deepspeed", {})
    eval_params = params.get("evaluation", {})
    logging_params = params.get("logging", {})

    # Load tokenizer for data collator
    model_name = params.get("model", {}).get("name", "Qwen/Qwen3-4B")
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Create data collator for dynamic padding
    data_collator = DataCollatorWithPadding(
        tokenizer=tokenizer,
        pad_to_multiple_of=8  # For tensor cores efficiency
    )

    # Prepare DeepSpeed config if enabled
    deepspeed_config = None
    if deepspeed_params.get("enable", False):
        deepspeed_config = deepspeed_params.get("config", {})
        logger.info("DeepSpeed optimization enabled")

    # Define metrics computation
    def compute_metrics(eval_pred: EvalPrediction) -> Dict[str, float]:
        """Compute evaluation metrics."""
        predictions = eval_pred.predictions
        labels = eval_pred.label_ids

        # Handle model outputs
        if isinstance(predictions, tuple):
            predictions = predictions[0]

        # Get predicted class
        preds = np.argmax(predictions, axis=1)

        # Compute metrics
        metrics = {
            "accuracy": accuracy_score(labels, preds)
        }

        if eval_params.get("compute_metrics", True):
            metric_names = eval_params.get("metric_names", ["accuracy"])
            average = eval_params.get("average", "weighted")

            if "f1" in metric_names:
                metrics["f1"] = f1_score(labels, preds, average=average)
            if "precision" in metric_names:
                metrics["precision"] = precision_score(labels, preds, average=average)
            if "recall" in metric_names:
                metrics["recall"] = recall_score(labels, preds, average=average)

        return metrics

    # Set random seed for reproducibility
    seed = trainer_params.get("seed", 42)
    set_seed(seed)

    # Create training arguments
    training_args = TrainingArguments(
        output_dir=trainer_params.get("output_dir", "./checkpoints"),
        overwrite_output_dir=trainer_params.get("overwrite_output_dir", True),
        num_train_epochs=trainer_params.get("num_train_epochs", 3),
        per_device_train_batch_size=trainer_params.get("per_device_train_batch_size", 8),
        per_device_eval_batch_size=trainer_params.get("per_device_eval_batch_size", 16),
        gradient_accumulation_steps=trainer_params.get("gradient_accumulation_steps", 4),
        gradient_checkpointing=trainer_params.get("gradient_checkpointing", True),
        learning_rate=trainer_params.get("learning_rate", 2e-5),
        warmup_ratio=trainer_params.get("warmup_ratio", 0.1),
        weight_decay=trainer_params.get("weight_decay", 0.01),
        logging_steps=trainer_params.get("logging_steps", 10),
        save_strategy=trainer_params.get("save_strategy", "epoch"),
        save_total_limit=trainer_params.get("save_total_limit", 2),
        evaluation_strategy=trainer_params.get("evaluation_strategy", "epoch"),
        eval_steps=trainer_params.get("eval_steps"),
        metric_for_best_model=trainer_params.get("metric_for_best_model", "eval_loss"),
        greater_is_better=trainer_params.get("greater_is_better", False),
        load_best_model_at_end=trainer_params.get("load_best_model_at_end", True),
        push_to_hub=trainer_params.get("push_to_hub", False),
        report_to=trainer_params.get("report_to", ["none"]),  # MLflow will be handled externally
        fp16=trainer_params.get("fp16", False),
        bf16=trainer_params.get("bf16", True),
        dataloader_num_workers=trainer_params.get("dataloader_num_workers", 4),
        remove_unused_columns=trainer_params.get("remove_unused_columns", True),
        label_smoothing_factor=trainer_params.get("label_smoothing_factor", 0.0),
        optim=trainer_params.get("optim", "adamw_torch"),
        seed=seed,
        deepspeed=deepspeed_config,
        log_level=logging_params.get("log_level", "info"),
        log_on_each_node=logging_params.get("log_on_each_node", False),
        logging_first_step=logging_params.get("logging_first_step", True),
        logging_nan_inf_filter=logging_params.get("logging_nan_inf_filter", True),
        save_safetensors=logging_params.get("save_safetensors", True),
    )

    # Create Trainer instance
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_data["train"],
        eval_dataset=tokenized_data.get("valid", tokenized_data.get("validation")),
        data_collator=data_collator,
        tokenizer=tokenizer,
        compute_metrics=compute_metrics if eval_params.get("compute_metrics", True) else None,
    )

    # Calculate training metadata
    total_training_steps = len(tokenized_data["train"]) // (
        training_args.per_device_train_batch_size *
        training_args.gradient_accumulation_steps
    ) * training_args.num_train_epochs

    metadata = {
        "trainer_class": "HuggingFaceTrainer",
        "num_train_epochs": training_args.num_train_epochs,
        "train_batch_size": training_args.per_device_train_batch_size * training_args.gradient_accumulation_steps,
        "eval_batch_size": training_args.per_device_eval_batch_size,
        "learning_rate": training_args.learning_rate,
        "warmup_ratio": training_args.warmup_ratio,
        "weight_decay": training_args.weight_decay,
        "optimizer": training_args.optim,
        "total_training_steps": total_training_steps,
        "deepspeed_enabled": deepspeed_params.get("enable", False),
        "mixed_precision": "bf16" if training_args.bf16 else ("fp16" if training_args.fp16 else "fp32"),
        "gradient_checkpointing": training_args.gradient_checkpointing,
        "seed": seed,
    }

    trainer_components = {
        "trainer": trainer,
        "args": training_args,
        "collator": data_collator,
        "tokenizer": tokenizer,
    }

    logger.info(f"Trainer prepared: {total_training_steps} total training steps")

    return trainer_components, metadata


def train_model(
    trainer_components: Dict[str, Any],
    params: Dict[str, Any]
) -> Tuple[Any, Dict[str, Any], Dict[str, Any]]:
    """
    Execute model training.

    Args:
        trainer_components: Dictionary containing trainer, args, collator, tokenizer
        params: Additional training parameters

    Returns:
        Tuple of:
            - Trained model
            - Training metrics for MLflow logging
            - Training artifacts paths for MLflow logging
    """
    trainer = trainer_components["trainer"]
    training_args = trainer_components["args"]

    logger.info("Starting model training...")

    # Train the model
    train_result = trainer.train()

    # Save the final model
    final_model_path = Path(training_args.output_dir) / "final_model"
    trainer.save_model(str(final_model_path))

    # Save tokenizer
    tokenizer = trainer_components["tokenizer"]
    tokenizer.save_pretrained(str(final_model_path))

    # Get training metrics
    metrics = {
        "train_loss": train_result.metrics.get("train_loss"),
        "train_runtime": train_result.metrics.get("train_runtime"),
        "train_samples_per_second": train_result.metrics.get("train_samples_per_second"),
        "train_steps_per_second": train_result.metrics.get("train_steps_per_second"),
        "total_flos": train_result.metrics.get("total_flos"),
        "epoch": train_result.metrics.get("epoch"),
    }

    # Log best checkpoint info
    if hasattr(trainer.state, "best_metric"):
        metrics["best_metric"] = trainer.state.best_metric
        metrics["best_model_checkpoint"] = trainer.state.best_model_checkpoint

    # Artifacts to be logged to MLflow
    artifacts = {
        "model_path": str(final_model_path),
        "checkpoints_dir": training_args.output_dir,
        "training_args": training_args.to_dict(),
    }

    logger.info(f"Training completed. Final loss: {metrics['train_loss']:.4f}")

    return trainer.model, metrics, artifacts


def evaluate_model(
    trainer_components: Dict[str, Any],
    params: Dict[str, Any]
) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """
    Evaluate the trained model on validation and test sets.

    Args:
        trainer_components: Dictionary containing trained model and trainer
        params: Evaluation parameters:
            - save_predictions: Whether to save predictions
            - save_confusion_matrix: Whether to save confusion matrix

    Returns:
        Tuple of:
            - Evaluation metrics for MLflow logging
            - Evaluation artifacts (predictions, confusion matrix) for MLflow logging
    """
    trainer = trainer_components["trainer"]
    save_predictions = params.get("save_predictions", True)
    save_confusion_matrix = params.get("save_confusion_matrix", True)

    metrics = {}
    artifacts = {}

    # Evaluate on validation set
    if trainer.eval_dataset is not None:
        logger.info("Evaluating on validation set...")
        eval_result = trainer.evaluate()

        for key, value in eval_result.items():
            metrics[f"validation_{key.replace('eval_', '')}"] = value

        # Get predictions if requested
        if save_predictions:
            predictions = trainer.predict(trainer.eval_dataset)
            preds = np.argmax(predictions.predictions, axis=1)

            # Save predictions
            val_predictions_path = Path(trainer.args.output_dir) / "validation_predictions.json"
            with open(val_predictions_path, 'w') as f:
                json.dump({
                    "predictions": preds.tolist(),
                    "labels": predictions.label_ids.tolist(),
                    "metrics": predictions.metrics
                }, f, indent=2)
            artifacts["validation_predictions"] = str(val_predictions_path)

            # Save confusion matrix if requested
            if save_confusion_matrix:
                cm = confusion_matrix(predictions.label_ids, preds)
                cm_path = Path(trainer.args.output_dir) / "validation_confusion_matrix.npy"
                np.save(cm_path, cm)
                artifacts["validation_confusion_matrix"] = str(cm_path)
                metrics["validation_confusion_matrix_shape"] = cm.shape

    # Evaluate on test set if available
    test_dataset = trainer.train_dataset.dataset_dict.get("test") if hasattr(trainer.train_dataset, 'dataset_dict') else None

    if test_dataset is not None:
        logger.info("Evaluating on test set...")
        # Temporarily set eval_dataset to test_dataset
        original_eval = trainer.eval_dataset
        trainer.eval_dataset = test_dataset

        test_result = trainer.evaluate()

        for key, value in test_result.items():
            metrics[f"test_{key.replace('eval_', '')}"] = value

        # Get test predictions if requested
        if save_predictions:
            test_predictions = trainer.predict(test_dataset)
            test_preds = np.argmax(test_predictions.predictions, axis=1)

            # Save test predictions
            test_predictions_path = Path(trainer.args.output_dir) / "test_predictions.json"
            with open(test_predictions_path, 'w') as f:
                json.dump({
                    "predictions": test_preds.tolist(),
                    "labels": test_predictions.label_ids.tolist(),
                    "metrics": test_predictions.metrics
                }, f, indent=2)
            artifacts["test_predictions"] = str(test_predictions_path)

            # Save test confusion matrix if requested
            if save_confusion_matrix:
                test_cm = confusion_matrix(test_predictions.label_ids, test_preds)
                test_cm_path = Path(trainer.args.output_dir) / "test_confusion_matrix.npy"
                np.save(test_cm_path, test_cm)
                artifacts["test_confusion_matrix"] = str(test_cm_path)
                metrics["test_confusion_matrix_shape"] = test_cm.shape

        # Restore original eval_dataset
        trainer.eval_dataset = original_eval

    logger.info(f"Evaluation completed. Validation loss: {metrics.get('validation_loss', 'N/A')}, Test loss: {metrics.get('test_loss', 'N/A')}")

    return metrics, artifacts