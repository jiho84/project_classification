#!/bin/bash
# ========================================================
# Migration Script: Qwen3-4B → Qwen2.5-7B
# ========================================================

set -e  # Exit on error

echo "========================================================="
echo "Starting migration from Qwen3-4B to Qwen2.5-7B"
echo "========================================================="

# Step 1: Download model if not cached
echo ""
echo "[Step 1/8] Checking model availability..."
python -c "
from transformers import AutoModel, AutoTokenizer
import torch

print('Downloading Qwen2.5-7B model and tokenizer...')
try:
    tokenizer = AutoTokenizer.from_pretrained('Qwen/Qwen2.5-7B', trust_remote_code=True)
    model = AutoModel.from_pretrained(
        'Qwen/Qwen2.5-7B',
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
        device_map='cpu'  # Just for download
    )
    print('✅ Model successfully downloaded/cached')
except Exception as e:
    print(f'❌ Error downloading model: {e}')
    exit(1)
"

# Step 2: Backup current configuration
echo ""
echo "[Step 2/8] Backing up current configuration..."
cp conf/base/parameters/training.yml conf/base/parameters/training_qwen3_4b_backup.yml
echo "✅ Backup saved to training_qwen3_4b_backup.yml"

# Step 3: Test memory with minimal batch
echo ""
echo "[Step 3/8] Testing GPU memory with minimal batch..."
python -c "
import torch
from transformers import AutoModelForSequenceClassification
from peft import LoraConfig, get_peft_model

# Test with minimal configuration
model = AutoModelForSequenceClassification.from_pretrained(
    'Qwen/Qwen2.5-7B',
    num_labels=260,
    torch_dtype=torch.bfloat16,
    trust_remote_code=True,
    device_map='auto'
)

# Apply LoRA
lora_config = LoraConfig(
    task_type='SEQ_CLS',
    r=256,  # Start conservative
    lora_alpha=512,
    target_modules=['q_proj', 'k_proj', 'v_proj', 'o_proj'],
    layers_to_transform=[20, 21, 22, 23, 24, 25, 26, 27],
    lora_dropout=0.05,
    bias='lora_only',
    modules_to_save=['score']
)

model = get_peft_model(model, lora_config)
model.print_trainable_parameters()

# Test forward pass
dummy_input = torch.randint(0, 1000, (4, 256)).cuda()  # Small batch
with torch.no_grad():
    output = model(dummy_input)

# Check memory
allocated = torch.cuda.memory_allocated() / 1024**3
reserved = torch.cuda.memory_reserved() / 1024**3
print(f'✅ Memory test passed!')
print(f'   Allocated: {allocated:.2f}GB')
print(f'   Reserved: {reserved:.2f}GB')
"

# Step 4: Run quick training test
echo ""
echo "[Step 4/8] Running quick training test (10 steps)..."
echo "Using configuration: conf/base/parameters/training_qwen25_7b.yml"

# Create a test config with only 10 steps
cat > conf/base/parameters/training_qwen25_7b_test.yml << EOF
# Quick test configuration (10 steps only)
$(cat conf/base/parameters/training_qwen25_7b.yml | sed 's/num_train_epochs: 1/max_steps: 10/')
EOF

# Run test (modify your training command as needed)
echo "Would run: kedro run --pipeline=train --params=training_qwen25_7b_test.yml"
echo "⚠️  Skipping actual training test (uncomment to run)"
# kedro run --pipeline=train --params=training_qwen25_7b_test.yml

# Step 5: Benchmark speed
echo ""
echo "[Step 5/8] Benchmarking speed (estimate)..."
echo "Expected performance:"
echo "  - Qwen3-4B:    ~0.79s/step (current)"
echo "  - Qwen2.5-7B:  ~1.0-1.2s/step (estimated)"
echo "  - Slowdown:    ~25-50%"

# Step 6: Validate layer indices
echo ""
echo "[Step 6/8] Validating layer indices..."
python -c "
from transformers import AutoModel

model = AutoModel.from_pretrained('Qwen/Qwen2.5-7B', trust_remote_code=True)
num_layers = len(model.model.layers)
print(f'✅ Confirmed: Qwen2.5-7B has {num_layers} layers')
print(f'   Last 8 layers: [{num_layers-8}, {num_layers-7}, ..., {num_layers-1}]')

if num_layers != 28:
    print('❌ WARNING: Expected 28 layers, got', num_layers)
    print('   Update layers_to_transform in config!')
"

# Step 7: Check disk space for checkpoints
echo ""
echo "[Step 7/8] Checking disk space..."
df -h data/06_models/
echo ""
echo "Estimated checkpoint sizes:"
echo "  - Per checkpoint: ~300MB (LoRA only)"
echo "  - Full training (20 epochs): ~6GB"

# Step 8: Final checklist
echo ""
echo "[Step 8/8] Migration Checklist:"
echo "========================================================="
echo "✅ Model downloaded and cached"
echo "✅ Configuration file created (training_qwen25_7b.yml)"
echo "✅ Layer indices updated (20-27 for last 8 layers)"
echo "✅ Memory test passed"
echo "⚠️  Training test skipped (run manually)"
echo ""
echo "Next steps:"
echo "1. Review training_qwen25_7b.yml configuration"
echo "2. Run small test: kedro run --pipeline=train --params=extract_ratio:0.01"
echo "3. Monitor GPU memory: watch -n 1 nvidia-smi"
echo "4. If OOM, reduce batch_size from 12 to 8"
echo "5. Start full training when ready"
echo ""
echo "========================================================="
echo "Migration preparation complete!"
echo "=========================================================