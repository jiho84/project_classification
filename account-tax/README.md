# Account-Tax Classification Pipeline

Kedro-based MLOps pipeline for accounting classification with MLflow integration.

## Setup

```bash
# Install dependencies
pip install -r requirements.txt

# Or using uv (recommended)
uv pip install -e .
```

## Kedro Usage Guide

### 1. Run Entire Pipeline

```bash
# Default pipeline (full_preprocess: ingestion → preprocess → feature → split)
kedro run

# Full pipeline including training
kedro run --pipeline=full

# Specific pipeline
kedro run --pipeline=ingestion
kedro run --pipeline=preprocess
kedro run --pipeline=feature
kedro run --pipeline=split
kedro run --pipeline=train
```

### 2. Run Specific Nodes

```bash
# Run single node
kedro run --node=load_data
kedro run --node=standardize_columns
kedro run --node=tokenize_datasets

# Run multiple nodes
kedro run --nodes=load_data,standardize_columns
```

### 3. Run From/To Specific Nodes

```bash
# Run from start to specific node
kedro run --to-nodes=tokenize_datasets

# Run from specific node to end
kedro run --from-nodes=create_dataset

# Run from node A to node B
kedro run --from-nodes=serialize_to_text --to-nodes=tokenize_datasets

# Multiple to-nodes
kedro run --to-nodes=serialize_to_text,tokenize_datasets
```

### 4. Run with Tags

```bash
# Run all nodes with specific tag
kedro run --tag=feature
kedro run --tag=split

# Exclude nodes with tag
kedro run --tag=~skip
```

### 5. Parameter Overrides

```bash
# Override single parameter
kedro run --params "preprocess.clean.drop_duplicates:false"

# Override nested parameters
kedro run --params "split.test_size:0.3,split.val_size:0.1"
```

### 6. Debug & Development

```bash
# List all pipelines
kedro registry list

# Visualize pipeline
kedro viz

# Start Jupyter
kedro jupyter lab
kedro jupyter notebook

# MLflow UI
kedro mlflow ui
```

## Pipeline Structure

### Registered Pipelines

- `__default__`: Alias for `full_preprocess`
- `full`: Complete pipeline (ingestion → preprocess → feature → split → train)
- `full_preprocess`: Data preparation (ingestion → preprocess → feature → split)
- `data_prep`: Feature ready (ingestion → preprocess → feature)
- Individual stages: `ingestion`, `preprocess`, `feature`, `split`, `train`

### Pipeline Flow

1. **Ingestion**: Load raw parquet → standardize columns → extract metadata
2. **Preprocess**: Clean data → filter columns → normalize values → validate
3. **Feature**: Add holiday features → build features → select features
4. **Split**: Create HuggingFace Dataset → stratified split → serialize to text
5. **Train**: Tokenize datasets → prepare for trainer

### Data Outputs

- `data/01_raw/`: Raw input data (dataset.parquet)
- `data/02_intermediate/`: Standardized data
- `data/03_primary/`: Validated data after preprocessing
- `data/04_feature/`: Base table with engineered features
- `data/05_model_input/`: Serialized datasets (text + labels)
- `data/06_models/`: Tokenized datasets (MLflow artifact)
- `data/08_reporting/`: Statistics and reports

## Common Commands

```bash
# Run full pipeline to tokenization
kedro run --pipeline=full --to-nodes=tokenize_datasets

# Run only data preparation (no training)
kedro run --pipeline=data_prep

# Run feature engineering only
kedro run --pipeline=feature

# Override split ratios
kedro run --pipeline=split --params "split.test_size:0.25,split.val_size:0.05"
```
