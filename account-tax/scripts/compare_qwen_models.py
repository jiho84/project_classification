#!/usr/bin/env python3
"""
Compare Qwen model variants for the account-tax classification task.
This helps choose the best model based on your hardware and requirements.
"""

import json
from typing import Dict, List, Tuple


def calculate_memory(
    model_params: float,
    hidden_size: int,
    num_layers: int,
    batch_size: int,
    seq_length: int,
    lora_rank: int,
    lora_modules: int,
    lora_layers: int,
    precision_bytes: int = 2  # BF16
) -> Dict[str, float]:
    """Calculate memory requirements in GB."""

    # Model weights
    model_memory = model_params * 1e9 * precision_bytes / 1e9

    # LoRA parameters
    lora_params_per_module = hidden_size * lora_rank * 2  # A and B matrices
    total_lora_params = lora_params_per_module * lora_modules * lora_layers
    lora_memory = total_lora_params * precision_bytes / 1e9

    # Activations
    activation_per_sample = num_layers * hidden_size * seq_length
    activation_memory = activation_per_sample * precision_bytes * batch_size / 1e9

    # Gradients (same as LoRA params)
    gradient_memory = lora_memory

    # Optimizer states (Adam: 2x params)
    optimizer_memory = lora_memory * 2

    # Total
    total = model_memory + lora_memory + activation_memory + gradient_memory + optimizer_memory

    return {
        'model': model_memory,
        'lora': lora_memory,
        'activations': activation_memory,
        'gradients': gradient_memory,
        'optimizer': optimizer_memory,
        'total': total
    }


def print_model_comparison():
    """Print detailed comparison of Qwen models."""

    models = {
        'Qwen3-4B': {
            'params': 4.2,  # Billion
            'hidden_size': 2560,
            'num_layers': 36,
            'num_heads': 20,
            'num_kv_heads': 20,
            'intermediate_size': 13696,
            'vocab_size': 151936,
            'context': 32768,
            'released': 'April 2025',
            'training_data': '36T tokens',
            'languages': 119,
            'special': 'Thinking mode support',
            'hf_path': 'Qwen/Qwen3-4B',
            'recommended_lora_rank': 256,
            'recommended_batch': 16
        },
        'Qwen3-8B': {
            'params': 8.2,
            'hidden_size': 4096,
            'num_layers': 36,
            'num_heads': 32,
            'num_kv_heads': 8,  # GQA
            'intermediate_size': 12288,
            'vocab_size': 151936,
            'context': 32768,
            'released': 'April 2025',
            'training_data': '36T tokens',
            'languages': 119,
            'special': 'GQA + Thinking mode',
            'hf_path': 'Qwen/Qwen3-8B',
            'recommended_lora_rank': 128,
            'recommended_batch': 16
        },
        'Qwen2.5-7B': {
            'params': 7.6,
            'hidden_size': 3584,
            'num_layers': 28,
            'num_heads': 28,
            'num_kv_heads': 4,  # GQA
            'intermediate_size': 18944,
            'vocab_size': 152064,
            'context': 32768,
            'released': 'Sept 2024',
            'training_data': '18T tokens',
            'languages': 40,
            'special': 'Mature, stable',
            'hf_path': 'Qwen/Qwen2.5-7B',
            'recommended_lora_rank': 128,
            'recommended_batch': 16
        }
    }

    # Your training configuration
    batch_size = 16
    seq_length = 256
    lora_modules = 7  # q,k,v,o,gate,up,down
    lora_layers = 8  # Last 8 layers

    print("=" * 100)
    print("QWEN MODEL COMPARISON FOR ACCOUNT-TAX CLASSIFICATION")
    print("=" * 100)

    print("\n📊 MODEL SPECIFICATIONS")
    print("-" * 100)
    print(f"{'Model':<15} {'Params':<8} {'Hidden':<8} {'Layers':<8} {'Heads':<12} {'Context':<10} {'Released':<12} {'Training Data':<15}")
    print("-" * 100)

    for name, spec in models.items():
        kv_info = f"{spec['num_heads']}/{spec['num_kv_heads']}"
        print(f"{name:<15} {spec['params']:.1f}B     {spec['hidden_size']:<8} {spec['num_layers']:<8} {kv_info:<12} {spec['context']:<10} {spec['released']:<12} {spec['training_data']:<15}")

    print("\n💾 MEMORY REQUIREMENTS (24GB RTX 4090)")
    print("-" * 100)
    print(f"Configuration: Batch={batch_size}, Seq={seq_length}, BF16 precision")
    print("-" * 100)

    for name, spec in models.items():
        mem = calculate_memory(
            model_params=spec['params'],
            hidden_size=spec['hidden_size'],
            num_layers=spec['num_layers'],
            batch_size=batch_size,
            seq_length=seq_length,
            lora_rank=spec['recommended_lora_rank'],
            lora_modules=lora_modules,
            lora_layers=lora_layers
        )

        fits = "✅" if mem['total'] < 24 else "❌"
        headroom = 24 - mem['total']

        print(f"\n{name}: {fits} Total: {mem['total']:.2f} GB (Headroom: {headroom:+.2f} GB)")
        print(f"  - Model weights: {mem['model']:.2f} GB")
        print(f"  - LoRA (r={spec['recommended_lora_rank']}): {mem['lora']:.2f} GB")
        print(f"  - Activations: {mem['activations']:.2f} GB")
        print(f"  - Gradients+Optimizer: {mem['gradients'] + mem['optimizer']:.2f} GB")

    print("\n🚀 PERFORMANCE EXPECTATIONS")
    print("-" * 100)
    print(f"{'Model':<15} {'Accuracy':<20} {'Speed':<20} {'Memory':<20}")
    print("-" * 100)
    print(f"{'Qwen3-4B':<15} {'Baseline':<20} {'Fastest':<20} {'15 GB (40% headroom)':<20}")
    print(f"{'Qwen3-8B':<15} {'10-15% better':<20} {'~2x slower':<20} {'19 GB (20% headroom)':<20}")
    print(f"{'Qwen2.5-7B':<15} {'5-10% better':<20} {'~1.8x slower':<20} {'18 GB (25% headroom)':<20}")

    print("\n📋 RECOMMENDATIONS")
    print("-" * 100)
    print("\n1. FOR BEST ACCURACY → Qwen3-8B")
    print("   - Newest model with 2x more training data")
    print("   - GQA for efficient inference")
    print("   - Thinking mode for complex reasoning")
    print("   - 119 language support")

    print("\n2. FOR FASTEST ITERATION → Qwen3-4B (current)")
    print("   - Fastest training speed")
    print("   - Most memory headroom for larger batches")
    print("   - Same training data quality as 8B")

    print("\n3. FOR PROVEN STABILITY → Qwen2.5-7B")
    print("   - Mature model with extensive community testing")
    print("   - Good balance of size and performance")
    print("   - Well-documented fine-tuning recipes")

    print("\n⚙️ MIGRATION STEPS TO QWEN3-8B")
    print("-" * 100)
    print("1. Test model loading:")
    print("   python scripts/test_qwen3_8b.py")
    print("\n2. Update configuration:")
    print("   cp conf/base/parameters/training_qwen3_8b.yml conf/base/parameters/training.yml")
    print("\n3. Run small test:")
    print("   kedro run --pipeline=train --params 'split.extract_ratio:0.01'")
    print("\n4. Monitor memory:")
    print("   nvidia-smi -l 1  # In separate terminal")
    print("\n5. If OOM occurs, enable gradient checkpointing:")
    print("   Set gradient_checkpointing: true in training.yml")

    print("\n" + "=" * 100)


if __name__ == "__main__":
    print_model_comparison()