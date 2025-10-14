#!/usr/bin/env python
"""Test different training configurations to identify performance bottleneck.

This script will test multiple configurations and report speed for each.
"""

import yaml
import shutil
from pathlib import Path
import subprocess
import time
import re

# Path to training config
CONFIG_PATH = Path("/home/user/projects/kedro_project/account-tax/conf/base/parameters/training.yml")
BACKUP_PATH = CONFIG_PATH.with_suffix(".yml.backup")

def load_config():
    with open(CONFIG_PATH, 'r') as f:
        return yaml.safe_load(f)

def save_config(config):
    with open(CONFIG_PATH, 'w') as f:
        yaml.dump(config, f, default_flow_style=False, sort_keys=False)

def run_training_test(test_name, modifications, max_steps=30):
    """Run a training test with specified modifications."""
    print(f"\n{'='*80}")
    print(f"TEST: {test_name}")
    print(f"{'='*80}")

    # Load base config
    config = load_config()

    # Apply modifications
    for key_path, value in modifications.items():
        keys = key_path.split('.')
        target = config
        for key in keys[:-1]:
            target = target[key]
        target[keys[-1]] = value
        print(f"  Set {key_path} = {value}")

    # Limit to max_steps for testing
    config['train']['training_args']['max_steps'] = max_steps
    config['train']['training_args']['logging_steps'] = 5

    # Save modified config
    save_config(config)

    # Run training
    print(f"\nRunning training for {max_steps} steps...")
    start_time = time.time()

    try:
        result = subprocess.run(
            ["kedro", "run", "--pipeline=train"],
            capture_output=True,
            text=True,
            timeout=300  # 5 minute timeout
        )

        # Extract timing information from output
        if result.returncode == 0:
            # Look for step timings in output
            step_times = re.findall(r'(\d+\.\d+)s/it', result.stderr)
            if step_times:
                avg_time = sum(float(t) for t in step_times) / len(step_times)
                print(f"✓ Average step time: {avg_time:.2f}s/it")
                print(f"  (Found {len(step_times)} timing measurements)")
            else:
                print("✗ Could not extract step timings from output")

            # Check GPU memory usage
            gpu_mem_matches = re.findall(r'(\d+\.?\d*)GB/\d+GB', result.stderr)
            if gpu_mem_matches:
                max_gpu = max(float(m) for m in gpu_mem_matches)
                print(f"  GPU Memory: {max_gpu:.1f}GB")
        else:
            print(f"✗ Training failed with return code {result.returncode}")
            print("Error output:", result.stderr[-1000:])  # Last 1000 chars

    except subprocess.TimeoutExpired:
        print("✗ Training timed out after 5 minutes")
    except Exception as e:
        print(f"✗ Error running training: {e}")

    duration = time.time() - start_time
    print(f"  Total test duration: {duration:.1f}s")

    return config

def main():
    print("TRAINING PERFORMANCE BOTTLENECK IDENTIFICATION")
    print("="*80)

    # Backup original config
    print(f"Backing up original config to {BACKUP_PATH}")
    shutil.copy2(CONFIG_PATH, BACKUP_PATH)

    try:
        # Test configurations
        tests = [
            # Baseline test
            ("Baseline (batch=32, grad_checkpoint=true)", {
                'train.training_args.per_device_train_batch_size': 32,
                'train.training_args.gradient_checkpointing': True,
                'train.model.gradient_checkpointing': True,
                'train.training_args.eval_strategy': 'no',
                'train.training_args.save_strategy': 'steps',
                'train.training_args.save_steps': 100,
                'train.deepspeed.config.train_micro_batch_size_per_gpu': 32,
            }),

            # Test gradient checkpointing impact
            ("Batch=32, NO gradient checkpointing", {
                'train.training_args.per_device_train_batch_size': 32,
                'train.training_args.gradient_checkpointing': False,
                'train.model.gradient_checkpointing': False,
                'train.training_args.eval_strategy': 'no',
                'train.training_args.save_strategy': 'steps',
                'train.training_args.save_steps': 100,
                'train.deepspeed.config.train_micro_batch_size_per_gpu': 32,
            }),

            # Test batch size 64
            ("Batch=64, grad_checkpoint=true", {
                'train.training_args.per_device_train_batch_size': 64,
                'train.training_args.gradient_checkpointing': True,
                'train.model.gradient_checkpointing': True,
                'train.training_args.eval_strategy': 'no',
                'train.training_args.save_strategy': 'steps',
                'train.training_args.save_steps': 100,
                'train.deepspeed.config.train_micro_batch_size_per_gpu': 64,
            }),

            # Test batch size 64 without gradient checkpointing
            ("Batch=64, NO gradient checkpointing", {
                'train.training_args.per_device_train_batch_size': 64,
                'train.training_args.gradient_checkpointing': False,
                'train.model.gradient_checkpointing': False,
                'train.training_args.eval_strategy': 'no',
                'train.training_args.save_strategy': 'steps',
                'train.training_args.save_steps': 100,
                'train.deepspeed.config.train_micro_batch_size_per_gpu': 64,
            }),

            # Test batch size 96 without gradient checkpointing
            ("Batch=96, NO gradient checkpointing", {
                'train.training_args.per_device_train_batch_size': 96,
                'train.training_args.gradient_checkpointing': False,
                'train.model.gradient_checkpointing': False,
                'train.training_args.eval_strategy': 'no',
                'train.training_args.save_strategy': 'steps',
                'train.training_args.save_steps': 100,
                'train.deepspeed.config.train_micro_batch_size_per_gpu': 96,
            }),

            # Current problematic config
            ("Current Config (batch=96, grad_checkpoint=true)", {
                'train.training_args.per_device_train_batch_size': 96,
                'train.training_args.gradient_checkpointing': True,
                'train.model.gradient_checkpointing': True,
                'train.training_args.eval_strategy': 'epoch',
                'train.training_args.save_strategy': 'epoch',
                'train.deepspeed.config.train_micro_batch_size_per_gpu': 96,
            }),
        ]

        results = []
        for test_name, mods in tests:
            config = run_training_test(test_name, mods, max_steps=30)
            results.append((test_name, config))
            time.sleep(5)  # Cool down between tests

        # Summary
        print("\n" + "="*80)
        print("SUMMARY OF RESULTS")
        print("="*80)
        print("\nCompare the step times (s/it) to identify the bottleneck:")
        print("- If batch=32 is ~2.1s/it → baseline confirmed")
        print("- If disabling grad_checkpoint speeds up significantly → that's the issue")
        print("- If batch=64 is optimal → memory bandwidth bottleneck at batch=96")
        print("- If batch=96 without grad_checkpoint is fast → combined overhead issue")

    finally:
        # Restore original config
        print(f"\nRestoring original config from {BACKUP_PATH}")
        shutil.copy2(BACKUP_PATH, CONFIG_PATH)
        BACKUP_PATH.unlink()  # Remove backup

if __name__ == "__main__":
    main()