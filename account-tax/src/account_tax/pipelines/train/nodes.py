"""Training pipeline nodes.

This module implements the two-node training pipeline:

1. ``tokenize_datasets``: converts serialized Hugging Face datasets into
   tokenized form, saves the result on disk, and produces token length
   statistics.
2. ``launch_training``: writes a runnable training configuration and
   executes the DeepSpeed launcher as a subprocess.

Both nodes follow the project philosophy of 대칭화(패턴), 모듈화, 순서화.
"""

from __future__ import annotations

import json
import logging
import os
import subprocess
from pathlib import Path
from typing import Any, Dict, Tuple

import numpy as np
import yaml
from datasets import DatasetDict
from transformers import AutoTokenizer

try:
    import mlflow
except ImportError:
    mlflow = None

logger = logging.getLogger(__name__)


def _find_project_root() -> Path:
    """Locate the Kedro project root by searching for pyproject.toml."""

    current = Path(__file__).resolve()
    for parent in [current, *current.parents]:
        if (parent / "pyproject.toml").exists():
            return parent
    return Path.cwd()


PROJECT_ROOT = _find_project_root()


# ---------------------------------------------------------------------------
# Utility helpers
# ---------------------------------------------------------------------------


def _ensure_dir(path: Path) -> Path:
    """Create parent directories for the given path and return the path."""

    path.parent.mkdir(parents=True, exist_ok=True)
    return path


def _ensure_dirname(path: Path) -> Path:
    """Create the directory itself (not just parent) and return the path."""

    path.mkdir(parents=True, exist_ok=True)
    return path


# ---------------------------------------------------------------------------
# Node: tokenize_datasets
# ---------------------------------------------------------------------------


def tokenize_datasets(
    serialized_datasets: DatasetDict,
    tokenization_params: Dict[str, Any],
) -> Tuple[str, Dict[str, Any]]:
    """Tokenize text datasets, save to disk, and produce token length report.

    Args:
        serialized_datasets: DatasetDict containing ``text`` and ``labels``.
        tokenization_params: Dict with the following keys:
            - model_name: tokenizer source model (HF hub path)
            - max_length: maximum token length
            - truncation: bool, whether to truncate to ``max_length``
            - padding: "longest" | "max_length" | False
            - num_proc: parallel workers for ``map`` (default 4)
            - sample_size: optional sample extraction per split
            - output_dir: directory for ``save_to_disk`` output

    Returns:
        Tuple[dataset_path, token_length_report]
    """

    model_name = tokenization_params["model_name"]
    max_length = int(tokenization_params["max_length"])
    truncation = bool(tokenization_params.get("truncation", True))
    padding = tokenization_params.get("padding", "longest")
    num_proc = int(tokenization_params.get("num_proc", 4))
    sample_size = int(tokenization_params.get("sample_size", 0))
    output_dir = Path(tokenization_params["output_dir"])

    logger.info("Loading tokenizer: %s", model_name)
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id
        logger.info("Pad token not set; using EOS token %s", tokenizer.pad_token)

    # Optional sample extraction for inspection
    samples: Dict[str, Any] = {}
    if sample_size > 0:
        for split_name, split_ds in serialized_datasets.items():
            n_samples = min(sample_size, len(split_ds))
            if n_samples == 0:
                continue
            sample_ds = split_ds.shuffle(seed=42).select(range(n_samples))
            samples[split_name] = []
            for ex in sample_ds:
                tokens = tokenizer.tokenize(
                    ex["text"], truncation=truncation, max_length=max_length
                )
                samples[split_name].append(
                    {
                        "text": ex["text"],
                        "labels": ex["labels"],
                        "tokens": tokens,
                        "token_count": len(tokens),
                    }
                )

    def tokenize_function(examples: Dict[str, Any]) -> Dict[str, Any]:
        raw_tokenized = tokenizer(
            examples["text"],
            truncation=False,
            padding=False,
            add_special_tokens=True,
        )
        raw_lengths = [len(ids) for ids in raw_tokenized["input_ids"]]

        tokenized_batch = tokenizer(
            examples["text"],
            truncation=truncation,
            max_length=max_length,
            padding=padding,
            return_length=True,
        )

        tokenized_batch["raw_length"] = raw_lengths
        return tokenized_batch

    logger.info("Tokenizing datasets (padding=%s, truncation=%s)...", padding, truncation)
    tokenized = serialized_datasets.map(
        tokenize_function,
        batched=True,
        num_proc=num_proc,
        remove_columns=["text"],
        desc="Tokenizing",
    )

    # Token statistics per split
    per_split_stats: Dict[str, Dict[str, float]] = {}
    all_lengths: list[np.ndarray] = []
    all_raw_lengths: list[np.ndarray] = []
    for split_name, split_ds in tokenized.items():
        lengths = np.array(split_ds["length"], dtype=np.int64)
        raw_lengths = np.array(split_ds["raw_length"], dtype=np.int64)
        all_lengths.append(lengths)
        all_raw_lengths.append(raw_lengths)
        per_split_stats[split_name] = {
            "count": int(lengths.size),
            "mean": float(lengths.mean()),
            "max": int(lengths.max()),
            "min": int(lengths.min()),
            "p95": float(np.percentile(lengths, 95)),
            "raw_mean": float(raw_lengths.mean()),
            "raw_max": int(raw_lengths.max()),
            "raw_min": int(raw_lengths.min()),
            "raw_p95": float(np.percentile(raw_lengths, 95)),
        }
        logger.info(
            "%s: count=%s, mean=%.1f, max=%s (raw mean=%.1f, raw max=%s)",
            split_name,
            per_split_stats[split_name]["count"],
            per_split_stats[split_name]["mean"],
            per_split_stats[split_name]["max"],
            per_split_stats[split_name]["raw_mean"],
            per_split_stats[split_name]["raw_max"],
        )

    all_lengths = np.concatenate(all_lengths) if all_lengths else np.array([], dtype=np.int64)
    all_raw_lengths = np.concatenate(all_raw_lengths) if all_raw_lengths else np.array([], dtype=np.int64)
    overall_stats = {
        "count": int(all_lengths.size),
        "mean": float(all_lengths.mean()) if all_lengths.size else 0.0,
        "max": int(all_lengths.max()) if all_lengths.size else 0,
        "min": int(all_lengths.min()) if all_lengths.size else 0,
        "raw_mean": float(all_raw_lengths.mean()) if all_raw_lengths.size else 0.0,
        "raw_max": int(all_raw_lengths.max()) if all_raw_lengths.size else 0,
        "raw_min": int(all_raw_lengths.min()) if all_raw_lengths.size else 0,
    }
    for p in (50, 75, 90, 95, 99):
        overall_stats[f"p{p}"] = float(np.percentile(all_lengths, p)) if all_lengths.size else 0.0
        overall_stats[f"raw_p{p}"] = float(np.percentile(all_raw_lengths, p)) if all_raw_lengths.size else 0.0

    token_length_report = {
        "overall": overall_stats,
        "per_split": per_split_stats,
        "model_name": model_name,
        "max_length_config": max_length,
    }
    if samples:
        token_length_report["samples"] = samples

    # Save tokenized dataset for reuse
    _ensure_dirname(output_dir)
    logger.info("Saving tokenized datasets to %s", output_dir)
    tokenized.save_to_disk(str(output_dir))

    return str(output_dir), token_length_report


# ---------------------------------------------------------------------------
# Node: launch_training
# ---------------------------------------------------------------------------


def launch_training(
    tokenized_dataset_path: str,
    train_params: Dict[str, Any],
) -> Dict[str, Any]:
    """Launch the DeepSpeed training job as a subprocess.

    Args:
        tokenized_dataset_path: Path returned by ``tokenize_datasets``.
        train_params: Nested dict from ``params:train``.

    Returns:
        Dictionary containing paths to the generated config and metrics file.
    """

    cfg = train_params

    script_cfg = cfg.get("training_script", {})
    script_path = Path(script_cfg.get("path", "src/train/main_yaml.py"))
    if not script_path.is_absolute():
        script_path = (PROJECT_ROOT / script_path).resolve()
    config_output_path = Path(script_cfg.get("config_output_path", "data/06_models/run_config/train_config.yml"))
    if not config_output_path.is_absolute():
        config_output_path = (PROJECT_ROOT / config_output_path).resolve()

    data_cfg = cfg.get("data", {})
    training_args_cfg = cfg.get("training_args", {})
    model_cfg = cfg.get("model", {})
    lora_cfg = cfg.get("lora", {})
    lora_defaults = cfg.get("lora_defaults", {})
    deepspeed_cfg = cfg.get("deepspeed", {})
    metrics_cfg = cfg.get("metrics", {})
    loss_cfg = cfg.get("loss", {})
    resume_cfg = cfg.get("resume", {})

    # Get num_gpus from deepspeed config, fallback to training_script config
    num_gpus = int(deepspeed_cfg.get("num_gpus", script_cfg.get("num_gpus", 1)))

    # Merge LoRA config (defaults + overrides)
    lora_enable = bool(lora_cfg.get("enable", False))
    lora_config_dict = {}
    if lora_enable:
        lora_config_dict = {**lora_defaults, **lora_cfg.get("config", {})}

    # Compose training configuration for script
    train_config = {
        "seed": int(training_args_cfg.get("seed", 42)),
        "model": {**model_cfg},
        "data": {
            "tokenized_path": tokenized_dataset_path,
            "train_split": data_cfg.get("train_split", "train"),
            "eval_split": data_cfg.get("eval_split", "valid"),
            "test_split": data_cfg.get("test_split", "test"),
        },
        "training_args": {**training_args_cfg},
        "metrics": metrics_cfg,
        "lora": {
            "enable": lora_enable,
            "config": lora_config_dict,
        },
        "loss": {**loss_cfg},
        "resume": {**resume_cfg},
    }

    if deepspeed_cfg.get("config"):
        train_config["deepspeed"] = deepspeed_cfg["config"]

    # Ensure output dirs exist
    _ensure_dir(config_output_path)
    output_dir = Path(train_config["training_args"].get("output_dir", "data/06_models/checkpoints"))
    if not output_dir.is_absolute():
        output_dir = (PROJECT_ROOT / output_dir).resolve()
    _ensure_dirname(output_dir)
    train_config["training_args"]["output_dir"] = str(output_dir)
    metrics_path = metrics_cfg.get("path")
    if metrics_path:
        metrics_path = Path(metrics_path)
        if not metrics_path.is_absolute():
            metrics_path = (PROJECT_ROOT / metrics_path).resolve()
        train_config.setdefault("metrics", {})["path"] = str(metrics_path)
        _ensure_dir(metrics_path)

    # MLflow reporting enabled - hang prevention handled in training script
    # by overriding MLflowCallback.on_train_end()

    # Write YAML config for the training script
    logger.info("Writing training configuration to %s", config_output_path)
    with open(config_output_path, "w", encoding="utf-8") as f:
        yaml.safe_dump(train_config, f, allow_unicode=True, sort_keys=False)

    # Build DeepSpeed command
    cmd = [
        "deepspeed",
        "--num_gpus",
        str(num_gpus),
        str(script_path),
        "--config_yml",
        str(config_output_path),
    ]

    logger.info("Launching DeepSpeed training: %s", " ".join(cmd))

    # Prepare environment variables for subprocess to inherit Kedro MLflow context
    env = os.environ.copy()

    if mlflow is not None and mlflow.active_run():
        # Pass Kedro MLflow run context to subprocess via environment variables
        # Using NESTED run to prevent subprocess from closing parent run
        active_run = mlflow.active_run()
        tracking_uri = mlflow.get_tracking_uri()

        env["MLFLOW_TRACKING_URI"] = tracking_uri
        env["MLFLOW_RUN_ID"] = active_run.info.run_id
        env["MLFLOW_NESTED_RUN"] = "true"  # Critical: prevents subprocess from closing parent run

        # Optional: pass experiment name for clarity
        experiment_id = active_run.info.experiment_id
        try:
            client = mlflow.tracking.MlflowClient()
            experiment = client.get_experiment(experiment_id)
            if experiment:
                env["MLFLOW_EXPERIMENT_NAME"] = experiment.name
        except Exception:
            pass  # Experiment name is optional

        logger.info(
            "Passing MLflow context to subprocess: tracking_uri=%s, run_id=%s (nested=true)",
            tracking_uri,
            active_run.info.run_id,
        )
    else:
        logger.warning(
            "No active MLflow run found. Subprocess will not log metrics to MLflow."
        )

    # Stabilize NCCL for single-node ZeRO runs
    env.setdefault("TORCH_NCCL_ASYNC_ERROR_HANDLING", "1")
    env.setdefault("TORCH_NCCL_BLOCKING_WAIT", "1")
    env.setdefault("NCCL_IB_DISABLE", "1")
    env.setdefault("NCCL_P2P_DISABLE", "1")

    try:
        subprocess.run(cmd, env=env, check=True)
        logger.info("Training subprocess completed successfully")
    except subprocess.CalledProcessError as e:
        logger.error("Training subprocess failed with exit code %s", e.returncode)
        raise

    # Log artifacts to MLflow after training completion
    if mlflow is not None and mlflow.active_run():
        try:
            logger.info("Logging model artifacts to MLflow...")

            # Log final model directory (contains best model after load_best_model_at_end)
            if output_dir.exists():
                logger.info("Logging final/best model from %s", output_dir)
                mlflow.log_artifacts(str(output_dir), artifact_path="final_model")

            # Find and log best checkpoint if trainer_state.json exists
            trainer_state_path = output_dir / "trainer_state.json"
            if trainer_state_path.exists():
                try:
                    with open(trainer_state_path, "r", encoding="utf-8") as f:
                        trainer_state = json.load(f)

                    best_checkpoint = trainer_state.get("best_model_checkpoint")
                    if best_checkpoint:
                        best_ckpt_path = Path(best_checkpoint)
                        if best_ckpt_path.exists() and best_ckpt_path != output_dir:
                            logger.info("Logging best checkpoint from %s", best_ckpt_path)
                            mlflow.log_artifacts(str(best_ckpt_path), artifact_path="best_checkpoint")
                except (json.JSONDecodeError, KeyError) as e:
                    logger.warning("Could not read best checkpoint info from trainer_state.json: %s", e)

            logger.info("Successfully logged models to MLflow artifacts.")
        except Exception as e:
            logger.warning("Failed to log models to MLflow: %s", e)
    else:
        logger.info("MLflow not available or no active run. Models saved locally only.")

    result = {
        "config_path": str(config_output_path),
        "tokenized_path": tokenized_dataset_path,
    }
    if metrics_path and Path(metrics_path).exists():
        try:
            with open(metrics_path, "r", encoding="utf-8") as f:
                result["metrics"] = json.load(f)
        except json.JSONDecodeError:
            logger.warning("Failed to read metrics JSON at %s", metrics_path)

    return result
