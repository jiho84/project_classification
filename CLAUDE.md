# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This repository contains a Kedro-based MLOps project for accounting classification with two main components:
1. **Main Project**: Located at root level - appears to be a template/framework setup
2. **Account-Tax Project**: Located in `account-tax/` - the active implementation for account classification

The active development is in the `account-tax/` subdirectory, which implements a multi-stage data pipeline with MLflow integration for experiment tracking and model management.

## Development Commands

### Environment Setup
```bash
# Install dependencies using uv (recommended)
uv pip install -e .

# Or using standard pip
pip install -e .
```

### Running Pipelines (from account-tax directory)
```bash
# Change to account-tax directory first
cd account-tax

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

### Testing
```bash
# Run tests (when available)
pytest tests/
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
# Start MLflow UI (default port 5000) - from account-tax directory
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

### Project Structure
```
kedro_project/
├── account-tax/          # Main accounting classification project
│   ├── conf/            # Configuration files
│   │   ├── base/        # Default configs (catalog, parameters, mlflow)
│   │   └── repro/       # Reproducibility configs
│   ├── data/            # Data layers (01_raw to 05_model_input)
│   ├── src/account_tax/ # Source code
│   │   ├── pipelines/   # Pipeline modules
│   │   │   ├── ingestion/
│   │   │   ├── preprocess/
│   │   │   ├── feature/
│   │   │   ├── split/
│   │   │   ├── train/
│   │   │   └── evaluation/ (currently disabled)
│   │   └── pipeline_registry.py
│   ├── mlruns/          # MLflow tracking store
│   └── notebooks/       # Jupyter notebooks
└── pyproject.toml       # Root project dependencies
```

### Account-Tax Pipeline Architecture

The account-tax project implements a comprehensive data processing and ML pipeline:

#### Pipeline Flow
1. **Ingestion Pipeline**:
   - `load_data`: Validates and loads raw parquet data
   - `standardize_columns`: Maps Korean → English columns
   - `extract_metadata`: Collects statistics and schema

2. **Preprocess Pipeline**:
   - `clean_data`: Removes duplicates, handles missing values
   - `filter_data`: Excludes unnecessary columns
   - `normalize_value`: Converts code values to readable text
   - `validate_data`: Applies business rules filtering

3. **Feature Pipeline**:
   - `add_holiday_features`: Creates `day_type` based on holidays/weekends
   - `build_features`: Generates derived columns
   - `select_features`: Selects features based on configuration
   - `prepare_dataset_inputs`: Prepares base table for HuggingFace conversion

4. **Split Pipeline**:
   - `create_dataset`: Creates HuggingFace Dataset with label slots
   - `to_hf_and_split`: Performs stratified train/val/test split
   - `labelize_and_cast`: Applies ClassLabel schema and metadata
   - `serialize_for_nlp`: Text serialization for NLP processing

5. **Train Pipeline**:
   - `tokenize_datasets`: Tokenization and input ID generation
   - `prepare_for_trainer`: Packages data for HuggingFace Trainer (currently not connected)

#### Registered Pipelines
- `__default__`: Alias for `full_preprocess`
- `full_preprocess`: ingestion + preprocess + feature + split
- `full`: full_preprocess + train
- `data_prep`: ingestion + preprocess + feature
- Individual pipelines: `ingestion`, `preprocess`, `feature`, `split`, `train`

### Key Components

- **Pipeline Registry** (`account-tax/src/account_tax/pipeline_registry.py`): Central registration of all pipelines
- **Settings** (`account-tax/src/account_tax/settings.py`): Kedro configuration with OmegaConfigLoader
- **Data Catalog** (`account-tax/conf/base/catalog.yml`): Dataset definitions with MLflow artifact tracking
- **Parameters** (`account-tax/conf/base/parameters/`): Pipeline-specific parameters
- **MLflow Config** (`account-tax/conf/base/mlflow.yml`): Experiment tracking configuration

### Data Contract
- **Input**: `raw_account_data` (parquet) → `pandas.DataFrame`
- **Processing**: DataFrame maintained through preprocessing and feature engineering
- **Split Output**: HuggingFace `DatasetDict` with train/validation/test splits
- **Final Output**: `trainer_ready_data` (tokenized datasets ready for model training)

### MLflow Integration
- Automatic parameter logging from parameters files
- Artifact storage via `MlflowArtifactDataset` for key outputs
- Experiment tracking with configurable experiment names
- Run naming with random name generation

## Configuration

### Environment-Specific Settings
- **base/**: Default configuration for all environments
- **repro/**: Reproducibility-focused configuration
- **local/**: Local development overrides (gitignored)

### Key Configuration Files
- `catalog.yml`: Dataset definitions with paths and types
- `parameters/*.yml`: Pipeline-specific parameters (data_pipeline, train_pipeline, inference_pipeline)
- `mlflow.yml`: MLflow server and tracking configuration
- `globals.yml`: Global variables (BRANCH, AS_OF date)

## Dependencies

The project uses Python 3.12-3.13 and key dependencies include:
- **Kedro** (>=1.0.0): Pipeline orchestration
- **Kedro-datasets** (>=8.1.0): Data I/O
- **Kedro-viz** (12.0.0): Pipeline visualization
- **Kedro-mlflow**: MLflow integration
- **Pandas** (>=2.3.2): Data manipulation
- **PySpark** (>=4.0.1): Large-scale data processing
- **Scikit-learn** (>=1.7.1): ML utilities
- **Jupyter/IPython**: Interactive development
- **HuggingFace Datasets**: For NLP data processing