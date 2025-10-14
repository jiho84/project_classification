#!/usr/bin/env python
"""Diagnosis script for training speed degradation.

Compares baseline vs current configuration and identifies bottlenecks.
"""

import yaml
from pathlib import Path
import sys

def load_yaml(path):
    with open(path, 'r') as f:
        return yaml.safe_load(f)

def analyze_config():
    # Load current config
    training_yml = Path("/home/user/projects/kedro_project/account-tax/conf/base/parameters/training.yml")
    config = load_yaml(training_yml)

    print("=" * 80)
    print("TRAINING SPEED DEGRADATION ANALYSIS")
    print("=" * 80)

    print("\n1. BASELINE CONFIGURATION (from benchmark_workers_0.log):")
    print("-" * 40)
    print("  • per_device_train_batch_size: 32")
    print("  • gradient_checkpointing: true (both model & training_args)")
    print("  • eval_strategy: likely 'no' or 'steps'")
    print("  • save_strategy: likely 'steps'")
    print("  • report_to: [mlflow] (was active)")
    print("  • num_train_epochs: unknown")
    print("  • warmup_ratio: unknown")
    print("  • dataloader_num_workers: 0")
    print("  • Speed: ~2.13s/it")
    print("  • GPU memory: 7.8GB used, 12.4GB reserved")

    print("\n2. CURRENT CONFIGURATION:")
    print("-" * 40)
    train_args = config['train']['training_args']
    model_cfg = config['train']['model']

    print(f"  • per_device_train_batch_size: {train_args['per_device_train_batch_size']}")
    print(f"  • gradient_checkpointing: {train_args.get('gradient_checkpointing', False)} (training_args)")
    print(f"  • gradient_checkpointing: {model_cfg.get('gradient_checkpointing', False)} (model)")
    print(f"  • eval_strategy: '{train_args.get('eval_strategy', 'no')}'")
    print(f"  • save_strategy: '{train_args.get('save_strategy', 'no')}'")
    print(f"  • report_to: {train_args.get('report_to', [])}")
    print(f"  • num_train_epochs: {train_args.get('num_train_epochs', 'not set')}")
    print(f"  • warmup_ratio: {train_args.get('warmup_ratio', 0)}")
    print(f"  • dataloader_num_workers: {train_args.get('dataloader_num_workers', 0)}")
    print(f"  • learning_rate: {train_args.get('learning_rate', 'not set')}")
    print("  • Speed: 6.33s/it (3x SLOWER!)")
    print("  • GPU memory: 21GB used")

    print("\n3. KEY DIFFERENCES:")
    print("-" * 40)

    differences = []

    # Batch size change
    if train_args['per_device_train_batch_size'] != 32:
        differences.append(f"  ✗ Batch size increased: 32 → {train_args['per_device_train_batch_size']} (3x larger)")
        differences.append("    Impact: Should be FASTER but is 3x SLOWER")

    # Eval strategy
    if train_args.get('eval_strategy', 'no') == 'epoch':
        differences.append("  ✗ eval_strategy: 'epoch' (evaluates after 646 steps)")
        differences.append("    Impact: Adds overhead every epoch, but shouldn't affect step time")

    # Save strategy
    if train_args.get('save_strategy', 'no') == 'epoch':
        differences.append("  ✗ save_strategy: 'epoch' (saves checkpoint after 646 steps)")
        differences.append("    Impact: Adds checkpoint overhead every epoch")

    # Epochs change
    if train_args.get('num_train_epochs', 3) == 1:
        differences.append("  ✓ num_train_epochs: 20 → 1 (reduced)")
        differences.append("    Impact: Neutral for per-step speed")

    # Warmup change
    if train_args.get('warmup_ratio', 0) != 0.05:
        differences.append(f"  ✓ warmup_ratio: 0.05 → {train_args.get('warmup_ratio', 0)} (reduced)")
        differences.append("    Impact: Neutral for per-step speed")

    for diff in differences:
        print(diff)

    print("\n4. ROOT CAUSE ANALYSIS:")
    print("-" * 40)

    print("\nPRIMARY SUSPECT: Batch Size Increase (32 → 96)")
    print("  • Memory usage increased correctly: 12GB → 21GB")
    print("  • But speed DECREASED instead of increasing!")
    print("  • This suggests a bottleneck that gets worse with larger batches")

    print("\nPOSSIBLE BOTTLENECKS:")
    print("  1. Memory Bandwidth Saturation")
    print("     - Larger batches may exceed GPU memory bandwidth")
    print("     - RTX 4090 has 1008 GB/s bandwidth")
    print("     - With gradient checkpointing, memory access patterns change")
    print("  ")
    print("  2. Gradient Checkpointing Overhead")
    print("     - Recomputes activations during backward pass")
    print("     - Overhead scales with batch size")
    print("     - May become dominant at batch=96")
    print("")
    print("  3. CUDA Kernel Launch Overhead")
    print("     - More data = more kernel launches")
    print("     - May hit kernel launch bottleneck")
    print("")
    print("  4. DeepSpeed Communication Overhead")
    print("     - ZeRO Stage 2 shards optimizer states")
    print("     - Larger batches = more communication")

    print("\n5. IMMEDIATE TEST PLAN:")
    print("-" * 40)
    print("Test 1: Disable gradient_checkpointing")
    print("  - Set both model.gradient_checkpointing: false")
    print("  - And training_args.gradient_checkpointing: false")
    print("  - Expected: 20-30% faster, but may OOM")
    print("")
    print("Test 2: Reduce batch size to 64")
    print("  - per_device_train_batch_size: 64")
    print("  - Expected: Find sweet spot between 32 and 96")
    print("")
    print("Test 3: Disable eval/save strategies")
    print("  - eval_strategy: 'no'")
    print("  - save_strategy: 'steps' with save_steps: 100")
    print("  - Expected: Remove epoch-end overhead")
    print("")
    print("Test 4: Profile with batch=32 (baseline)")
    print("  - Restore original batch=32")
    print("  - Should match 2.13s/it from baseline")
    print("")
    print("Test 5: Check compilation overhead")
    print("  - First 10-20 steps often slow (CUDA compilation)")
    print("  - Check if speed stabilizes after 20 steps")

    print("\n6. RECOMMENDED CONFIGURATION:")
    print("-" * 40)
    print("Based on analysis, try this configuration:")
    print("""
train:
  training_args:
    per_device_train_batch_size: 64  # Sweet spot
    gradient_checkpointing: false    # Disable if memory allows
    eval_strategy: "steps"            # Not every epoch
    eval_steps: 100                   # Evaluate every 100 steps
    save_strategy: "steps"            # Save checkpoints
    save_steps: 100                   # Every 100 steps

  model:
    gradient_checkpointing: false    # Disable for speed

  deepspeed:
    config:
      train_micro_batch_size_per_gpu: 64  # Match batch size
""")

    print("\n7. MONITORING COMMANDS:")
    print("-" * 40)
    print("# Watch GPU utilization and memory bandwidth:")
    print("watch -n 0.5 nvidia-smi")
    print("")
    print("# Monitor detailed GPU metrics:")
    print("nvidia-smi dmon -s pucvmet -d 1")
    print("")
    print("# Check if it's compilation overhead (first steps slow):")
    print("# Look at step times: 1-10 vs 11-20 vs 21-30")

if __name__ == "__main__":
    analyze_config()