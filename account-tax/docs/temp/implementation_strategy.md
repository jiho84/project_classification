# Pipeline Restructuring Implementation Strategy
> Created: 2025-09-29
> Project Manager: Claude (Project-Manager Agent)

## Executive Summary

This document outlines the implementation strategy for restructuring the Kedro pipelines to separate **Data Pipeline** (공용/shared) and **Train Pipeline** (실험/experimental), improving reusability and experimental flexibility.

## 1. Implementation Priorities & Dependencies

### Phase 1: Foundation (Prerequisites)
**Priority: HIGH | Timeline: Day 1**

1. **Parameter File Restructuring**
   - Dependencies: None
   - Risk: Low
   - Impact: All subsequent work depends on this
   - Tasks:
     - Backup existing parameter files
     - Create `conf/base/parameters.yml` (data parameters)
     - Create `conf/base/parameters_train.yml` (training parameters)
     - Validate parameter loading

2. **Pipeline Registry Update**
   - Dependencies: Parameter restructuring
   - Risk: Medium
   - Impact: Changes how pipelines are invoked
   - Tasks:
     - Update `pipeline_registry.py`
     - Define `data_pipeline` (ingestion + preprocess + feature)
     - Redefine `train_pipeline` (split + serialize + tokenize + training)

### Phase 2: Core Implementation
**Priority: HIGH | Timeline: Days 2-3**

3. **Training Node Implementation**
   - Dependencies: Pipeline registry
   - Risk: High (new functionality)
   - Impact: Critical for training capability
   - Implementation Order:
     1. `instantiate_model` - Model initialization
     2. `apply_peft_adapters` - LoRA configuration
     3. `configure_optimizer` - Training optimization
     4. `configure_deepspeed` - Distributed training
     5. `run_training` - Training execution
     6. `evaluate_model` - Performance metrics

4. **Configuration Updates**
   - Dependencies: Training nodes
   - Risk: Medium
   - Impact: Model performance and training efficiency
   - Tasks:
     - Update model from Qwen2.5-0.5B to Qwen3-4B
     - Add LoRA parameters
     - Add DeepSpeed configuration

### Phase 3: Validation & Testing
**Priority: HIGH | Timeline: Day 4**

5. **Integration Testing**
   - Dependencies: All above phases
   - Risk: Medium
   - Impact: Quality assurance
   - Tasks:
     - Test data pipeline independently
     - Validate base_table output
     - Test train pipeline end-to-end
     - Verify MLflow tracking

### Phase 4: Documentation
**Priority: MEDIUM | Timeline: Day 5**

6. **Documentation Updates**
   - Dependencies: Testing completion
   - Risk: Low
   - Impact: Maintainability
   - Tasks:
     - Update architecture.md
     - Update history.md
     - Create review.md

## 2. Technical Implementation Details

### 2.1 Parameter Structure Migration

**Current Structure:**
```
conf/base/parameters/
├── data_pipeline.yml
├── train_pipeline.yml
└── inference_pipeline.yml
```

**Target Structure:**
```
conf/base/
├── parameters.yml        # Data pipeline parameters
├── parameters_train.yml  # Training parameters
└── parameters/          # Backup directory
```

**Migration Script:**
```bash
# Backup existing parameters
cp -r conf/base/parameters conf/base/parameters.backup

# Create new parameter files
# parameters.yml will contain: ingestion, preprocess, feature
# parameters_train.yml will contain: split, train, model, lora, deepspeed
```

### 2.2 Pipeline Registry Changes

**Current Registry:**
```python
# Individual pipelines scattered
ingestion_pipeline + preprocess_pipeline + feature_pipeline + split_pipeline + train_pipeline
```

**Target Registry:**
```python
def register_pipelines():
    # Data Pipeline (공용)
    data_pipeline = (
        ingestion.create_pipeline() +
        preprocess.create_pipeline() +
        feature.create_pipeline()
    )

    # Train Pipeline (실험)
    train_pipeline = (
        split.create_pipeline() +
        train.create_enhanced_pipeline()  # New enhanced training
    )

    return {
        "__default__": data_pipeline,
        "data": data_pipeline,
        "train": train_pipeline,
        "full": data_pipeline + train_pipeline,
        # Individual pipelines for debugging
        "ingestion": ingestion.create_pipeline(),
        "preprocess": preprocess.create_pipeline(),
        "feature": feature.create_pipeline(),
        "split": split.create_pipeline(),
    }
```

### 2.3 New Training Nodes Architecture

```python
# train/nodes.py additions

def instantiate_model(model_config: dict) -> PreTrainedModel:
    """Initialize Qwen3-4B model with configuration."""
    pass

def apply_peft_adapters(model: PreTrainedModel, lora_config: dict) -> PeftModel:
    """Apply LoRA adapters to the model."""
    pass

def configure_optimizer(model: PeftModel, optimizer_config: dict) -> torch.optim.Optimizer:
    """Configure AdamW optimizer with parameters."""
    pass

def configure_deepspeed(trainer_config: dict) -> dict:
    """Setup DeepSpeed configuration for distributed training."""
    pass

def run_training(
    model: PeftModel,
    datasets: DatasetDict,
    optimizer: Optimizer,
    deepspeed_config: dict,
    training_args: dict
) -> TrainerOutput:
    """Execute training with HuggingFace Trainer."""
    pass

def evaluate_model(
    model: PeftModel,
    eval_dataset: Dataset,
    metrics_config: dict
) -> dict:
    """Evaluate model performance and return metrics."""
    pass
```

### 2.4 Configuration Parameters

**parameters_train.yml structure:**
```yaml
# Split configuration (moved from train_pipeline.yml)
split:
  label_column: acct_code
  seed: 42
  test_size: 0.2
  val_size: 0.1

# Model configuration
model:
  name: "Qwen/Qwen3-4B"  # Updated from Qwen2.5-0.5B
  revision: "main"
  trust_remote_code: true
  torch_dtype: "bfloat16"

# LoRA configuration
lora:
  r: 16
  lora_alpha: 32
  lora_dropout: 0.1
  target_modules:
    - "q_proj"
    - "k_proj"
    - "v_proj"
    - "o_proj"
  bias: "none"
  task_type: "CAUSAL_LM"

# DeepSpeed configuration
deepspeed:
  zero_optimization:
    stage: 2
    offload_optimizer:
      device: "cpu"
      pin_memory: true
    offload_param:
      device: "cpu"
      pin_memory: true
  bf16:
    enabled: true
  gradient_checkpointing: true
  gradient_accumulation_steps: 4

# Training configuration
training:
  learning_rate: 2e-4
  num_train_epochs: 3
  per_device_train_batch_size: 4
  per_device_eval_batch_size: 8
  warmup_steps: 100
  weight_decay: 0.01
  logging_steps: 10
  evaluation_strategy: "steps"
  eval_steps: 100
  save_strategy: "steps"
  save_steps: 500
```

## 3. Risk Mitigation Strategies

### 3.1 High-Risk Areas

1. **Model Memory Requirements**
   - Risk: Qwen3-4B requires significantly more memory than Qwen2.5-0.5B
   - Mitigation: Implement gradient checkpointing and DeepSpeed ZeRO-2
   - Fallback: Use quantization or smaller batch sizes

2. **Training Node Integration**
   - Risk: New nodes may not integrate properly with existing pipeline
   - Mitigation: Implement comprehensive unit tests for each node
   - Fallback: Keep existing training pipeline as backup

3. **Parameter Migration**
   - Risk: Breaking changes in parameter loading
   - Mitigation: Create backup and test parameter loading before proceeding
   - Fallback: Restore from backup if issues occur

### 3.2 Rollback Plan

1. All changes will be made in a feature branch
2. Original parameter files will be backed up
3. Pipeline registry changes can be reverted
4. New nodes are additive (don't modify existing code)

## 4. Success Criteria

### Phase 1 Success Metrics
- [ ] Parameter files successfully migrated
- [ ] Pipeline registry updated without breaking existing pipelines
- [ ] `kedro run --pipeline=data` executes successfully

### Phase 2 Success Metrics
- [ ] All 6 new training nodes implemented
- [ ] Model configuration updated to Qwen3-4B
- [ ] LoRA and DeepSpeed configurations added

### Phase 3 Success Metrics
- [ ] Data pipeline produces valid base_table
- [ ] Train pipeline completes without errors
- [ ] MLflow tracks experiments correctly
- [ ] Model artifacts saved properly

### Phase 4 Success Metrics
- [ ] Architecture documentation reflects new structure
- [ ] History log captures implementation process
- [ ] Review document outlines improvements and future work

## 5. Resource Requirements

### Development Resources
- **Pipeline Developer Agent**: Implementation of new nodes
- **Code Evaluator Agent**: Review and validation
- **History Documenter Agent**: Process documentation

### Technical Resources
- **Memory**: Minimum 16GB RAM for Qwen3-4B
- **Storage**: 20GB for model weights and artifacts
- **Compute**: GPU recommended for training (CPU fallback available)

## 6. Communication Plan

### Daily Updates
- Progress against task checklist in task.md
- Blockers or issues encountered
- Next steps and dependencies

### Milestone Communications
- End of each phase completion
- Any significant risks or changes to plan
- Final implementation summary

## 7. Next Steps

1. **Immediate Actions** (Today):
   - Begin parameter file restructuring
   - Create parameter file backups
   - Start pipeline registry updates

2. **Tomorrow**:
   - Begin training node implementation
   - Start with model instantiation node

3. **This Week**:
   - Complete all implementation phases
   - Conduct comprehensive testing
   - Deliver documentation updates

## Appendix A: File Modifications Checklist

- [ ] `/conf/base/parameters.yml` - Create
- [ ] `/conf/base/parameters_train.yml` - Create
- [ ] `/conf/base/parameters/` - Backup and reorganize
- [ ] `/src/account_tax/pipeline_registry.py` - Update
- [ ] `/src/account_tax/pipelines/train/nodes.py` - Extend
- [ ] `/src/account_tax/pipelines/train/pipeline.py` - Enhance
- [ ] `/conf/base/catalog.yml` - Update for new datasets
- [ ] `/docs/architecture.md` - Update
- [ ] `/docs/history.md` - Update
- [ ] `/docs/review.md` - Create

## Appendix B: Testing Checklist

- [ ] Unit tests for each new node
- [ ] Integration test for data pipeline
- [ ] Integration test for train pipeline
- [ ] End-to-end test for full pipeline
- [ ] MLflow artifact verification
- [ ] Model output validation
- [ ] Performance benchmarking

---
*This implementation strategy will be updated as the project progresses.*