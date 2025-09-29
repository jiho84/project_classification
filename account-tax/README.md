# Account-Tax Classification Pipeline

Simplified Kedro pipeline for account classification with only essential nodes.

## Setup

```bash
pip install -r requirements.txt
```

## Run Pipeline

```bash
# From account-tax directory
kedro run
```

## Pipeline Structure

The pipeline consists of only essential stages:
1. **Preprocess** (S2-S3): Data cleaning and integration
2. **Feature** (S4-S5): Feature engineering and selection  
3. **Train** (S6-S8): Data splitting, preprocessing, and feature selection

Only final results are saved to disk in `data/05_model_input/`.
