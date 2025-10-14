#!/usr/bin/env python3
"""
Test script to verify Qwen3-8B model loading and memory usage.
Run this before full training to ensure the model fits in memory.
"""

import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from peft import LoraConfig, get_peft_model, TaskType
import psutil
import GPUtil


def format_bytes(bytes_value):
    """Convert bytes to human readable format."""
    for unit in ['B', 'KB', 'MB', 'GB', 'TB']:
        if bytes_value < 1024.0:
            return f"{bytes_value:.2f} {unit}"
        bytes_value /= 1024.0
    return f"{bytes_value:.2f} PB"


def get_memory_info():
    """Get current memory usage."""
    # CPU Memory
    cpu_mem = psutil.virtual_memory()
    cpu_used = format_bytes(cpu_mem.used)
    cpu_total = format_bytes(cpu_mem.total)

    # GPU Memory
    gpus = GPUtil.getGPUs()
    gpu_info = []
    for gpu in gpus:
        gpu_used = f"{gpu.memoryUsed:.2f} MB"
        gpu_total = f"{gpu.memoryTotal:.2f} MB"
        gpu_info.append(f"GPU {gpu.id}: {gpu_used}/{gpu_total} ({gpu.memoryUtil*100:.1f}%)")

    return {
        'cpu': f"{cpu_used}/{cpu_total} ({cpu_mem.percent}%)",
        'gpus': gpu_info
    }


def test_model_loading():
    """Test loading Qwen3-8B with LoRA configuration."""

    print("=" * 80)
    print("Qwen3-8B Model Loading Test")
    print("=" * 80)

    # Configuration matching training_qwen3_8b.yml
    model_name = "Qwen/Qwen3-8B"
    num_labels = 260  # Your number of classes
    batch_size = 16
    max_length = 256

    # LoRA configuration
    lora_config = LoraConfig(
        task_type=TaskType.SEQ_CLS,
        r=128,
        lora_alpha=256,
        lora_dropout=0.05,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                       "gate_proj", "up_proj", "down_proj"],
        layers_to_transform=list(range(28, 36)),  # Last 8 layers
        modules_to_save=["score"],
        bias="lora_only"
    )

    print("\n1. Initial Memory Status:")
    mem_info = get_memory_info()
    print(f"   CPU: {mem_info['cpu']}")
    for gpu_info in mem_info['gpus']:
        print(f"   {gpu_info}")

    try:
        print(f"\n2. Loading tokenizer from {model_name}...")
        tokenizer = AutoTokenizer.from_pretrained(
            model_name,
            trust_remote_code=True,
            padding_side='left'  # For batch generation
        )
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        print("   ✓ Tokenizer loaded successfully")
        print(f"   Vocabulary size: {len(tokenizer)}")

        print(f"\n3. Loading model from {model_name}...")
        print("   This may take a few minutes for 8.2B parameters...")

        # Load model in BF16
        model = AutoModelForSequenceClassification.from_pretrained(
            model_name,
            num_labels=num_labels,
            torch_dtype=torch.bfloat16,
            trust_remote_code=True,
            device_map="auto"  # Automatic device placement
        )

        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"   ✓ Base model loaded")
        print(f"   Total parameters: {total_params:,} ({total_params/1e9:.2f}B)")
        print(f"   Trainable parameters: {trainable_params:,}")

        print("\n4. After model loading:")
        mem_info = get_memory_info()
        print(f"   CPU: {mem_info['cpu']}")
        for gpu_info in mem_info['gpus']:
            print(f"   {gpu_info}")

        print("\n5. Applying LoRA configuration...")
        model = get_peft_model(model, lora_config)

        # Print LoRA info
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        all_params = sum(p.numel() for p in model.parameters())
        print(f"   ✓ LoRA applied")
        print(f"   Trainable parameters: {trainable_params:,} ({trainable_params/1e6:.2f}M)")
        print(f"   All parameters: {all_params:,} ({all_params/1e9:.2f}B)")
        print(f"   Trainable ratio: {100 * trainable_params / all_params:.2f}%")

        print("\n6. Testing forward pass with dummy batch...")
        # Create dummy input
        dummy_texts = ["Test input " + str(i) for i in range(batch_size)]
        inputs = tokenizer(
            dummy_texts,
            max_length=max_length,
            truncation=True,
            padding="max_length",
            return_tensors="pt"
        )

        # Move to GPU if available
        if torch.cuda.is_available():
            inputs = {k: v.cuda() for k, v in inputs.items()}
            model = model.cuda()

        # Forward pass
        with torch.no_grad():
            outputs = model(**inputs)
            logits = outputs.logits
            print(f"   ✓ Forward pass successful")
            print(f"   Output shape: {logits.shape}")

        print("\n7. After forward pass:")
        mem_info = get_memory_info()
        print(f"   CPU: {mem_info['cpu']}")
        for gpu_info in mem_info['gpus']:
            print(f"   {gpu_info}")

        # Test with gradient computation
        print("\n8. Testing backward pass...")
        model.train()
        outputs = model(**inputs, labels=torch.randint(0, num_labels, (batch_size,)).cuda())
        loss = outputs.loss
        loss.backward()
        print(f"   ✓ Backward pass successful")
        print(f"   Loss: {loss.item():.4f}")

        print("\n9. After backward pass:")
        mem_info = get_memory_info()
        print(f"   CPU: {mem_info['cpu']}")
        for gpu_info in mem_info['gpus']:
            print(f"   {gpu_info}")

        # Cleanup
        del model
        del inputs
        torch.cuda.empty_cache()

        print("\n" + "=" * 80)
        print("✅ SUCCESS: Qwen3-8B can be loaded and trained with current configuration")
        print("=" * 80)

    except Exception as e:
        print("\n" + "=" * 80)
        print(f"❌ ERROR: {str(e)}")
        print("=" * 80)
        print("\nTroubleshooting suggestions:")
        print("1. Enable gradient checkpointing in config")
        print("2. Reduce batch size from 16 to 8")
        print("3. Enable DeepSpeed ZeRO-3 optimization")
        print("4. Reduce LoRA rank from 128 to 64")
        print("5. Use 8-bit optimizer (adamw_8bit)")
        raise


if __name__ == "__main__":
    # Check dependencies
    try:
        import GPUtil
    except ImportError:
        print("Installing GPUtil for GPU monitoring...")
        import subprocess
        subprocess.check_call(["pip", "install", "gputil"])
        import GPUtil

    test_model_loading()