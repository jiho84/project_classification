# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Kedro-based MLOps project for accounting classification with MLflow integration. The active implementation is in the `account-tax/` subdirectory, which processes accounting data through a multi-stage pipeline (ingestion → preprocessing → feature engineering → splitting → training).

## Development Commands

### Environment Setup
```bash
# Install dependencies using uv (recommended)
uv pip install -e .

# Or using standard pip
pip install -e .
```

### Running Pipelines
```bash
# Run the default pipeline (full_preprocess: ingestion → preprocess → feature → split)
kedro run

# Run the full pipeline including training
kedro run --pipeline=full

# Run specific pipelines
kedro run --pipeline=ingestion
kedro run --pipeline=preprocess
kedro run --pipeline=feature
kedro run --pipeline=split
kedro run --pipeline=train

# Run with parameter overrides
kedro run --params "preprocess.clean.drop_duplicates:false"

# Run specific nodes
kedro run --node=load_data
kedro run --node=standardize_columns
```

### Testing & Linting
```bash
# Currently no testing/linting tools installed
# To install dev dependencies (when available):
# uv pip install -e ".[dev]"
```

### Jupyter Development
```bash
# Launch Jupyter Lab
kedro jupyter lab

# Launch Jupyter Notebook
kedro jupyter notebook
```

### MLflow UI
```bash
# Start MLflow UI (default port 5000)
kedro mlflow ui

# Or directly with tracking store
mlflow ui --backend-store-uri ./mlruns
```

### Visualization
```bash
# Start Kedro-Viz for pipeline visualization
kedro viz
```

## Architecture

### Directory Structure
- **Root level**: Framework setup and configuration
- **account-tax/**: Active accounting classification implementation
  - `conf/`: Configuration (base, repro environments)
  - `data/`: Data layers (01_raw to 05_model_input)
  - `src/account_tax/pipelines/`: Pipeline modules
  - `mlruns/`: MLflow tracking store

### Pipeline Flow

1. **Ingestion**: Load raw parquet → standardize columns → extract metadata
2. **Preprocess**: Clean data → filter columns → normalize values → validate
3. **Feature**: Add holiday features → build features → select features → prepare base table
4. **Split**: Create HuggingFace Dataset → stratified split → apply ClassLabel schema → serialize for NLP
5. **Train**: Tokenize datasets → prepare for trainer (currently not connected to actual training)

### Registered Pipelines
- `__default__`: Alias for `full_preprocess`
- `full_preprocess`: ingestion + preprocess + feature + split
- `full`: full_preprocess + train
- `data_prep`: ingestion + preprocess + feature
- Individual pipelines: `ingestion`, `preprocess`, `feature`, `split`, `train`

### Data Contract
- **Input**: `raw_account_data` (parquet) → `pandas.DataFrame`
- **Processing**: DataFrame maintained through preprocessing and feature engineering
- **Split Output**: HuggingFace `DatasetDict` with train/validation/test splits
- **Final Output**: `trainer_ready_data` (tokenized datasets ready for model training)

### Configuration
- **catalog.yml**: Dataset definitions with MLflow artifact tracking
- **parameters/**: Pipeline-specific parameters (data_pipeline, train_pipeline, inference_pipeline)
- **mlflow.yml**: Experiment tracking configuration
- **globals.yml**: Global variables (BRANCH, AS_OF date)

### Known Issues
- Evaluation pipeline disabled (references undefined nodes)
- No pytest or linting tools currently installed
- Train pipeline prepare_for_trainer node not connected to actual model training

## Design Philosophy (대칭화 · 모듈화 · 순서화)

- **대칭화(Pattern)**: Consistent patterns across nodes, pipelines, and documentation
- **모듈화(Modularity)**: Node-based separation with clear input/output contracts
- **순서화(Ordering)**: Clear causality in folder structure and execution flow