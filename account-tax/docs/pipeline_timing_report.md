# Pipeline Timing Report (full_preprocess)

- **Execution date:** 2025-09-24
- **Mlflow experiment:** `account_tax_experiment`
- **Mlflow run ID:** `47a4d67ca7d34779b042acea095488bb` (pipeline scope)
- **Pipeline duration:** 287.06 s (~4m 47s)

## Node Durations
| Node | Duration (s) | Notes |
| --- | --- | --- |
| `load_data` | 0.01 | Parquet load + empty check |
| `standardize_columns` | 0.00 | Column renaming only |
| `clean_data` | 5.55 | Drop duplicates / nulls |
| `filter_data` | 0.31 | Column drop |
| `normalize_value` | 4.48 | Code-to-text mappings |
| `validate_data` | 0.72 | Amount/date filtering |
| `build_features` | 3.95 | Holiday feature |
| `select_features` | 0.19 | Column ordering |
| `prepare_dataset_inputs` | 3.96 | Clean DataFrame ready for HF conversion |
| `create_dataset` | 5.85 | HF Dataset + label slots |
| `to_hf_and_split` | 0.42 | Stratified split with fallback |
| `labelize_and_cast` | 153.07 | Batched label encoding (`num_proc=4`) |
| `serialize_for_nlp` | 79.85 | Batched text generation + column pruning |

> The metrics originate from MLflow run `92e83cb7966a48db9dcbb1ca8208b139` (node-level run created by `kedro-mlflow`).

## How to View in MLflow UI
1. `cd account-tax`
2. `mlflow ui --port 5000`
3. Open `http://localhost:5000`, select `account_tax_experiment`, and inspect the runs listed above.

- Pipeline duration metric key: `pipeline_duration__full_preprocess`
- Node duration metrics follow the pattern `node_duration__<node_name>`

## Observations & Next Steps
- `serialize_for_nlp` and `labelize_and_cast` dominate runtime (~90% of total). Consider batching optimisations or parallelism for these nodes if runtime becomes a bottleneck.
- All metrics are logged automatically via the `TimingHook` registered in `src/account_tax/settings.py`.
- Subsequent runs will append new entries; use `mlflow ui` filters or run tags to compare timing across iterations.
