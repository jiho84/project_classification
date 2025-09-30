# Architecture Notes

- Use this file to record high-level structural decisions.
- Cross-reference detailed diagrams below; maintain this document as the single source of architectural truth.
- Whenever pipelines or module boundaries change, summarize the update here and link to supporting docs.
- **Last Updated**: 2025-09-29
- **Architecture Version**: 2.0.0

## 설계 철학 (대칭화 · 모듈화 · 순서화)

- **대칭화(Pattern)**: 동일한 본질의 기능은 유사한 패턴으로 작성하여 구조 파악 시 불필요한 인지 비용을 줄인다. 노드 함수, `pipeline.py` 구성, 문서 구조 모두 일관된 형식을 유지한다.
- **모듈화(Modularity)**: 노드 기반으로 기능을 분리하고, 각 모듈이 명확한 입력·출력 계약을 갖도록 유지한다. 파이프라인은 노드 조합만으로 복잡한 동작을 표현해야 한다.
- **순서화(Ordering)**: 폴더/파일 구조(정적)와 실행 흐름(동적)의 인과를 명확히 기록한다. 문서에는 단계별 순서, 데이터 흐름, 의존 관계를 명시한다.

## Change Log

### 2025-09-29 · Architecture Documentation Update v2.0
- Corrected pipeline structure: 5 main pipelines (ingestion → preprocess → feature → split → train), not 10 stages
- Added comprehensive Package Version Management section
- Updated function inventory tables to match actual implementations
- Added MLflow integration architecture details
- Documented evaluation pipeline status and missing implementations
- Added system-wide compatibility matrix

### 2025-09-24 · Dataset 리팩터링

- Feature 파이프라인 마지막 노드를 `prepare_dataset_inputs`로 단순화하여 `base_table`만 반환하도록 변경.
- Split 파이프라인에 `create_dataset`을 추가해 HuggingFace `Dataset` 생성과 라벨 슬롯 구성을 담당하도록 이동.
- `serialize_for_nlp`가 `retain_columns` 옵션을 사용해 `text`, `labels`, `acct_code`만 유지하도록 정리.
- 상세 흐름과 함수 목록은 아래 “세부 아키텍처” 섹션을 확인하세요.

## Pipeline Status & Boundaries

| Pipeline | Status | Notes |
|----------|--------|-------|
| Ingestion | ✅ Complete | Raw parquet → standardized frame |
| Preprocess | ✅ Complete | Clean/filter/normalize + missing value unification |
| Feature | ✅ Complete | Date/holiday features, selection |
| Split | ✅ Complete | HF Dataset creation, labelize, serialization, token diagnostics |
| Train | ⚠️ In progress | LoRA/Deepspeed training integration pending |
| Inference | ⏳ Planned | Mirrors tokenization + model loading |

### Split 전 · 후 경계

```
┌─────────────────────────────┐  Split 이전 (공용 데이터 생성)
│ Ingestion ─ Preprocess ─ Feature ─ (Scaling) │
│  • 결측치/코드/수치 정규화
│  • 파생 피처 생성
└────────────┬────────────────┘
             ↓ base_table (공유)
┌─────────────────────────────┐  Split 이후 (학습/실험 영역)
│ create_dataset → to_hf_and_split → labelize_and_cast │
│ serialize_for_nlp → analyze_token_lengths → export_*  │
└─────────────────────────────┘
```

- **Split 이전 파이프라인**은 재사용 가능한 DataFrame(스케일링 포함)을 만드는 데 집중합니다.
- **Split 이후 파이프라인**은 실험마다 바뀌는 요소(Split 비율, 라벨 정수화, 텍스트 직렬화, 토큰 진단, 학습 설정)를 담당합니다.
- `train_pipeline.yml`의 `split.*` 파라미터로 동일 입력에 대해 여러 학습 버전을 유연하게 관리할 수 있습니다.

## Package Version Management

### Core Dependencies Matrix

#### Python Runtime
- **Required**: Python >= 3.12, < 3.14
- **Tested**: Python 3.12.x, 3.13.x
- **Rationale**: Balance between modern features and ecosystem compatibility

#### Framework Stack

| Package | Version | Purpose | Compatibility Notes |
|---------|---------|---------|--------------------|
| **Kedro** | 1.0.0 (exact) | Pipeline orchestration | Major version pinned for stability |
| **kedro-datasets** | >=8.1.0 | Extended dataset types | Includes pandas support |
| **kedro-mlflow** | >=1.0.0, <1.1.0 | MLflow integration | Tight version range for API stability |
| **kedro-viz** | 12.0.0 (exact) | Pipeline visualization | Fixed to prevent ipython conflicts |

#### Data Processing Stack

| Package | Version | Purpose | Compatibility Notes |
|---------|---------|---------|--------------------|
| **pandas** | >=2.3.2 | DataFrame operations | Latest 2.x for performance |
| **pyarrow** | >=19.0.0, <20.0.0 | Parquet I/O | Major version constraint |
| **PySpark** | >=4.0.1 | Large-scale processing | 4.x for Spark 3.x compatibility |
| **numpy** | >=1.21.0 | Numerical operations | Compatible with pandas 2.x |

#### ML/AI Stack

| Package | Version | Purpose | Compatibility Notes |
|---------|---------|---------|--------------------|
| **scikit-learn** | >=1.7.1 | ML utilities | Latest for evaluation metrics |
| **datasets** | >=3.0.0, <3.1.0 | HuggingFace datasets | Major version constraint |
| **mlflow** | >=2.22, <3.0 | Experiment tracking | 2.x for stability |
| **mlflow-skinny** | >=2.22, <3.0 | Lightweight MLflow | Must match mlflow version |

#### Development Stack

| Package | Version | Purpose | Compatibility Notes |
|---------|---------|---------|--------------------|
| **ipython** | 8.26.0 (exact) | Interactive shell | <9.0 for kedro-viz compatibility |
| **jupyter** | Latest | Notebooks | Via jupyterlab>=4.4.7 |
| **holidays** | >=0.81 | Holiday features | For date engineering |

### Version Conflict Resolution

1. **IPython < 9.0 constraint**: Required by kedro-viz 12.0.0 to prevent visualization conflicts
2. **PyArrow 19.x**: Pinned to major version for Parquet format stability
3. **MLflow 2.x**: Avoiding 3.x migration until ecosystem matures
4. **Datasets 3.0.x**: Minor version constraint for HuggingFace API stability

### Upgrade Path Recommendations

- **Q1 2026**: Evaluate Kedro 2.x migration when ecosystem stabilizes
- **Q2 2026**: Consider MLflow 3.x after plugin compatibility verified
- **Continuous**: Monitor pyarrow for performance improvements in 20.x

## 세부 아키텍처

### System Architecture Overview

```mermaid
graph LR
    A[Raw Data<br/>Parquet] -->|load| B[Ingestion<br/>Pipeline]
    B -->|standardize| C[Preprocess<br/>Pipeline]
    C -->|clean/filter| D[Feature<br/>Pipeline]
    D -->|engineer| E[Split<br/>Pipeline]
    E -->|HF Dataset| F[Train<br/>Pipeline]
    F -->|tokenize| G[Model Ready<br/>Data]

    B -.->|MLflow| M[Tracking<br/>Store]
    C -.->|params| M
    D -.->|params| M
    E -.->|artifacts| M
    F -.->|artifacts| M
```

### Block Architecture Principles

설계자는 폴더 → 파일 → 함수 단위의 블록 구성을 유지하면서 데이터 계약과 의존 관계를 추적합니다. 본 문서는 현재 구현 상태를 반영하며, 모든 함수의 책임과 흐름을 `대분류/중분류/함수이름/역할/입력/출력/후속블록` 기준으로 관리합니다.

#### 1. 파이프라인 아키텍처 개요

##### 1.1 Pipeline Structure (5 Main Pipelines)
```
Ingestion → Preprocess → Feature → Split → Train
   ↓           ↓           ↓         ↓       ↓
[3 nodes]   [4 nodes]   [4 nodes] [5 nodes] [2 nodes]
```

##### 1.2 Registered Pipeline Combinations
- **`__default__`**: Alias for `full_preprocess` (Ingestion → Split)
- **`full_preprocess`**: ingestion + preprocess + feature + split
- **`full`**: full_preprocess + train (complete pipeline)
- **`data_prep`**: ingestion + preprocess + feature (pre-split preparation)
- **Individual**: `ingestion`, `preprocess`, `feature`, `split`, `train`
- **Disabled**: `evaluation` (missing node implementations)

##### 1.3 Data Flow Architecture
- **Input**: `raw_account_data` (Parquet) → pandas.DataFrame
- **Processing**: DataFrame maintained through feature engineering
- **Transformation**: HuggingFace Dataset creation at split stage
- **Output**: `trainer_ready_data` (tokenized DatasetDict)

##### 1.4 MLflow Integration Points
- **Artifact Storage**: `MlflowArtifactDataset` for `prepared_datasets_mlflow`, `text_datasets_mlflow`
- **Parameter Logging**: Automatic from parameters/*.yml files
- **Experiment Tracking**: Via `account_tax_experiment` with random run names
- **Tracking Store**: Local `mlruns/` directory

#### 2. 데이터 계약 핵심
- 입력 원천: `raw_account_data` (`data/01_raw/row_data.parquet`) → `pandas.DataFrame`
- 전처리: `standardized_data`~`validated_data`까지 `DataFrame` 유지, 파라미터는 `conf/base/parameters.yml`의 `preprocess.*`
- 특성화: `prepare_dataset_inputs`가 `base_table`(`DataFrame`)을 제공하며 라벨 슬롯은 Split 단계에서 관리
- 분할: `create_dataset`이 HF `Dataset`과 라벨 슬롯을 생성하고, `to_hf_and_split`이 층화 분할을 수행, `labelize_and_cast`가 `ClassLabel` 스키마와 `label_metadata`를 부착해 `prepared_datasets`로 전달
- 학습 준비: `serialize_for_nlp`에서 텍스트 직렬화 → `tokenize_datasets`에서 토큰화 → `prepare_for_trainer`가 `trainer_ready_data`(datasets, collator, model_config)를 최종 산출

#### 3. Pipeline Node Documentation

## Ingestion Pipeline (3 nodes)

### 1. **load_data**
   - **입력**: `raw_data: pd.DataFrame` (from raw_account_data parquet file)
   - **출력**: `validated_raw_data: pd.DataFrame` (validated non-empty DataFrame)
   - **설명**: Loads raw accounting data from parquet file and validates it's not empty. Logs data shape and columns for monitoring.
   - **후속블록**: standardize_columns

### 2. **standardize_columns**
   - **입력**: `data: pd.DataFrame` (validated_raw_data)
   - **출력**: `standardized_data: pd.DataFrame` (with English column names)
   - **설명**: Standardizes Korean column names to English using predefined mapping. Handles 54 column mappings including accounting codes, party information, and tax fields.
   - **후속블록**: extract_metadata, clean_data

### 3. **extract_metadata**
   - **입력**: `data: pd.DataFrame` (standardized_data)
   - **출력**: `metadata: Dict[str, Any]` (statistics and schema info)
   - **설명**: Extracts comprehensive metadata including row/column counts, data types, memory usage, null counts, and column type categorization.
   - **후속블록**: External monitoring/logging

## Preprocess Pipeline (4 nodes)

### 1. **clean_data**
   - **입력**: `data: pd.DataFrame`, `clean_params: Dict` (from params:preprocess.clean)
   - **출력**: `cleaned_data: pd.DataFrame`
   - **설명**: Removes duplicate rows and handles missing values. Drops rows with nulls in critical columns specified in dropna_columns parameter.
   - **후속블록**: filter_data

### 2. **filter_data**
   - **입력**: `data: pd.DataFrame`, `filter_params: Dict` (from params:preprocess.filter)
   - **출력**: `filtered_data: pd.DataFrame`
   - **설명**: Excludes unnecessary columns specified in exclude_columns parameter. Typically removes id, created_at, batch_no columns.
   - **후속블록**: normalize_value

### 3. **normalize_value**
   - **입력**: `df: pd.DataFrame`, `code_mappings: Optional[Dict]`
   - **출력**: `normalized_data: pd.DataFrame`
   - **설명**: Converts code values to human-readable text. Maps VAT message codes, document types, and party type codes to meaningful labels.
   - **후속블록**: validate_data

### 4. **validate_data**
   - **입력**: `data: pd.DataFrame`, `params: Dict`
   - **출력**: `validated_data: pd.DataFrame`
   - **설명**: Applies business rule validation. Filters by amount thresholds and validates date formats, removing invalid records.
   - **후속블록**: build_features

### 5. **normalize_missing_values** (optional, not in pipeline)
   - **입력**: `data: pd.DataFrame`, `placeholders: Dict`
   - **출력**: `normalized_data: pd.DataFrame`
   - **설명**: Standardizes missing values with appropriate placeholders per data type (categorical: "__missing__", numeric: 0, etc.).
   - **후속블록**: Not currently integrated

## Feature Pipeline (4 nodes)

### 1. **build_features**
   - **입력**: `data: pd.DataFrame`, `params: Dict` (from params:feature.engineering)
   - **출력**: `feature_data: pd.DataFrame`
   - **설명**: Creates derived features including holiday features. Calls add_holiday_features internally to add day_type column.
   - **후속블록**: select_features

### 2. **add_holiday_features** (helper function)
   - **입력**: `df: pd.DataFrame`, `date_column: str`
   - **출력**: `df: pd.DataFrame` (with day_type column)
   - **설명**: Adds day_type column marking weekends and Korean public holidays as 'holiday', others as 'workday'.
   - **후속블록**: Called within build_features

### 3. **select_features**
   - **입력**: `data: pd.DataFrame`, `params: Dict` (from params:feature.selection)
   - **출력**: `selected_features: pd.DataFrame`
   - **설명**: Selects specific features and label columns in configured order. Typically selects 24 features + 1 label column.
   - **후속블록**: prepare_dataset_inputs

### 4. **prepare_dataset_inputs**
   - **입력**: `data: pd.DataFrame`, `params: Dict` (from params:feature.dataset_conversion)
   - **출력**: `base_table: pd.DataFrame`
   - **설명**: Prepares cleaned base table for HuggingFace Dataset conversion. Removes null labels and duplicates, logs unique label count.
   - **후속블록**: create_dataset

## Split Pipeline (6 main nodes + 3 helpers)

### Helper Functions

#### **_initialize_label_slots** (helper)
   - **입력**: `max_classes: int`, `dummy_prefix: str`
   - **출력**: `List[str]` (dummy label slots)
   - **설명**: Creates deterministic dummy label slots like ["dummy1", "dummy2", ...] up to max_classes.

#### **_upsert_labels_into_slots** (helper)
   - **입력**: `names: List[str]`, `new_labels: List[str]`, `dummy_prefix: str`
   - **출력**: `names: List[str]` (updated with real labels)
   - **설명**: Replaces dummy slots with real labels while preserving index positions.

#### **make_label2id** / **make_id2label** (helpers)
   - **입력**: `names: List[str]`
   - **출력**: `Dict[str, int]` or `Dict[int, str]`
   - **설명**: Creates bidirectional mappings between label names and integer IDs.

### Main Nodes

### 1. **create_dataset**
   - **입력**: `base_table: pd.DataFrame`, `params: Dict` (from params:split)
   - **출력**: `dataset: Dataset`, `names: List[str]`
   - **설명**: Creates HuggingFace Dataset from pandas DataFrame and initializes label slot system with max_classes slots.
   - **후속블록**: to_hf_and_split

### 2. **to_hf_and_split**
   - **입력**: `dataset: Dataset`, `label_col: str`, `seed: int`, `test_size: float`, `val_size: float`
   - **출력**: `splits: DatasetDict` (train/valid/test)
   - **설명**: Performs stratified train/validation/test split. Falls back to random split if stratification fails.
   - **후속블록**: labelize_and_cast

### 3. **labelize_and_cast**
   - **입력**: `splits: DatasetDict`, `names: List[str]`, `label_col: str`, `dummy_label: str`, `num_proc: int`
   - **출력**: `labeled_datasets: DatasetDict` (with integer labels and ClassLabel schema)
   - **설명**: Maps string labels to integers and applies ClassLabel schema. Attaches label_metadata to DatasetDict.
   - **후속블록**: serialize_for_nlp

### 4. **serialize_for_nlp**
   - **입력**: `dataset_dict: DatasetDict`, `params: Dict` (from params:train.serialization)
   - **출력**: `text_datasets: DatasetDict` (with text column)
   - **설명**: Serializes structured data into text format for NLP. Creates "column: value" format strings, retains only text and labels columns.
   - **후속블록**: analyze_token_lengths or tokenize_datasets

### 5. **analyze_token_lengths** (optional diagnostic)
   - **입력**: `dataset_dict: DatasetDict`, `tokenization_params: Dict`, `diagnostics_params: Dict`
   - **출력**: `dataset_dict: DatasetDict`, `report: Dict` (token statistics)
   - **설명**: Generates token length statistics and samples. Logs percentiles and mean/max token counts to MLflow.
   - **후속블록**: tokenize_datasets

### 6. **export_prepared_partitions** / **export_text_partitions**
   - **입력**: `dataset_dict: DatasetDict`
   - **출력**: `partitions: Dict[str, pd.DataFrame]`
   - **설명**: Converts DatasetDict splits back to partitioned pandas DataFrames for storage.
   - **후속블록**: MLflow artifact storage

## Train Pipeline (6 implemented nodes, 2 in pipeline)

### 1. **tokenize_datasets**
   - **입력**: `dataset_dict: DatasetDict`, `params: Dict` (from params:train.tokenization)
   - **출력**: `tokenized_datasets: DatasetDict`, `metadata: Dict`
   - **설명**: Tokenizes text data using HuggingFace tokenizer. Produces input_ids and attention_mask, removes text column to save memory.
   - **후속블록**: load_model or prepare_trainer

### 2. **load_model** (not in pipeline)
   - **입력**: `tokenized_data: DatasetDict`, `params: Dict` (from params:train.model)
   - **출력**: `model: AutoModelForSequenceClassification`, `metadata: Dict`
   - **설명**: Loads pre-trained model with proper configuration. Infers num_labels from data, enables gradient checkpointing, handles device mapping.
   - **후속블록**: apply_optimization

### 3. **apply_optimization** (not in pipeline)
   - **입력**: `model: Any`, `params: Dict` (from params:train.optimization)
   - **출력**: `optimized_model: Any`, `metadata: Dict`
   - **설명**: Applies LoRA optimization and/or torch.compile. Reduces trainable parameters from billions to millions with LoRA.
   - **후속블록**: prepare_trainer

### 4. **prepare_trainer** (in pipeline as prepare_for_trainer)
   - **입력**: `model: Any`, `tokenized_data: DatasetDict`, `params: Dict`
   - **출력**: `trainer_components: Dict`, `metadata: Dict`
   - **설명**: Prepares HuggingFace Trainer with all configurations. Sets up data collator, training arguments, DeepSpeed config, and metrics computation.
   - **후속블록**: train_model

### 5. **train_model** (not in pipeline)
   - **입력**: `trainer_components: Dict`, `params: Dict`
   - **출력**: `model: Any`, `metrics: Dict`, `artifacts: Dict`
   - **설명**: Executes model training using HuggingFace Trainer. Saves final model and tokenizer, returns training metrics.
   - **후속블록**: evaluate_model

### 6. **evaluate_model** (not in pipeline)
   - **입력**: `trainer_components: Dict`, `params: Dict`
   - **출력**: `metrics: Dict`, `artifacts: Dict`
   - **설명**: Evaluates trained model on validation and test sets. Generates predictions and confusion matrices if requested.
   - **후속블록**: Model deployment or further analysis

## Evaluation Pipeline (2 implemented, 4 missing)

### Implemented Nodes

### 1. **evaluate_predictions**
   - **입력**: `y_true: pd.Series`, `y_pred: pd.Series`
   - **출력**: `metrics: Dict` (accuracy, precision, recall, f1, confusion matrix)
   - **설명**: Calculates classification metrics using sklearn. Computes accuracy, precision, recall, F1 score with weighted averaging.
   - **상태**: ✅ Implemented but not connected to pipeline

### 2. **calculate_tax_impact**
   - **입력**: `predictions: pd.DataFrame` (with actual and predicted tax categories)
   - **출력**: `impact: Dict` (misclassified_amount, percentage, category breakdown)
   - **설명**: Estimates financial impact of tax misclassifications. Calculates misclassified amounts and percentages by category.
   - **상태**: ✅ Implemented but not connected to pipeline

### Missing Nodes (defined in pipeline but not implemented)

### 3. **evaluate_classification_model** ❌
   - **Expected 입력**: `y_test`, `y_pred`, `y_proba`
   - **Expected 출력**: `classification_metrics`
   - **설명**: Should evaluate classification model performance
   - **상태**: ❌ Pipeline expects this but node not implemented

### 4. **evaluate_tax_classification** ❌
   - **Expected 입력**: `test_predictions`, `tax_categories`
   - **Expected 출력**: `tax_metrics`
   - **설명**: Should evaluate tax-specific classification accuracy
   - **상태**: ❌ Pipeline expects this but node not implemented

### 5. **calculate_business_metrics** ❌
   - **Expected 입력**: `test_predictions`, `test_actuals`
   - **Expected 출력**: `business_metrics`
   - **설명**: Should calculate business impact metrics
   - **상태**: ❌ Pipeline expects this but node not implemented

### 6. **generate_evaluation_report** ❌
   - **Expected 입력**: `classification_metrics`, `tax_metrics`, `model_info`
   - **Expected 출력**: `evaluation_report`
   - **설명**: Should generate comprehensive evaluation report
   - **상태**: ❌ Pipeline expects this but node not implemented

## Utility Functions (not integrated into pipelines)

### 1. **categorize_accounts**
   - **입력**: `df: pd.DataFrame`, `account_mapping: Dict`
   - **출력**: `df: pd.DataFrame` (with account_category column)
   - **설명**: Maps accounts to categories based on provided mapping dictionary. Adds account_category column to DataFrame.
   - **상태**: ⚠️ Implemented but unused

### 2. **calculate_tax_categories**
   - **입력**: `df: pd.DataFrame`
   - **출력**: `df: pd.DataFrame` (with tax_category column)
   - **설명**: Derives tax categories based on keyword matching in account descriptions. Creates tax_category column.
   - **상태**: ⚠️ Implemented but unused

### 3. **aggregate_by_period**
   - **입력**: `df: pd.DataFrame`, `period_column: str`, `aggregation_level: str`
   - **출력**: `df: pd.DataFrame` (with period column)
   - **설명**: Aggregates data by time period (daily/weekly/monthly/quarterly/yearly). Adds period column for grouping.
   - **상태**: ⚠️ Implemented but unused

## System Registry & Entry Points

### 1. **register_pipelines** (pipeline_registry.py)
   - **입력**: None
   - **출력**: `Dict[str, Pipeline]` (all registered pipelines)
   - **설명**: Central registration point for all pipelines. Defines __default__, full_preprocess, full, data_prep, and individual pipeline mappings.
   - **후속블록**: Kedro CLI commands

### 2. **main** (__main__.py)
   - **입력**: CLI arguments
   - **출력**: Exit code (0 for success)
   - **설명**: Entry point for Kedro session execution. Handles kedro run and other CLI commands.
   - **후속블록**: Terminal/shell execution

#### 4. Architecture Health Check

##### 4.1 Implementation Status
| Component | Status | Notes |
|-----------|--------|-------|
| Ingestion Pipeline | ✅ Complete | 3 nodes fully implemented |
| Preprocess Pipeline | ✅ Complete | 4 nodes with business rules |
| Feature Pipeline | ✅ Complete | 4 nodes including holidays |
| Split Pipeline | ✅ Complete | 5 nodes with HF integration |
| Train Pipeline | ✅ Complete | 2 nodes for tokenization |
| Evaluation Pipeline | ⚠️ Partial | 2/6 nodes implemented |
| Utils Module | ⚠️ Unused | 3 functions not integrated |

##### 4.2 Critical Issues
1. **Evaluation Pipeline Gaps**:
   - Missing: `evaluate_classification_model`, `evaluate_tax_classification`
   - Missing: `generate_evaluation_report`, `calculate_business_metrics`
   - Impact: Cannot run evaluation pipeline without implementation
   - Resolution: Either implement missing nodes or update pipeline definition

2. **Unused Utilities**:
   - `categorize_accounts`: Account categorization logic unused
   - `calculate_tax_categories`: Tax category calculation unused
   - `aggregate_by_period`: Period aggregation unused
   - Resolution: Consider integration or removal

##### 4.3 Architectural Strengths
- ✅ Clear separation of concerns across pipelines
- ✅ Consistent data contracts between stages
- ✅ Label metadata preservation through split→train
- ✅ MLflow artifact tracking at key checkpoints
- ✅ Memory-efficient with MemoryDataset for intermediates

##### 4.4 Configuration Dependencies
- ⚠️ `split.max_classes` and `split.dummy_prefix` control label slot allocation
- ⚠️ Changes to these parameters affect downstream ClassLabel schema
- ⚠️ `feature.selection.features` list must match actual DataFrame columns

#### 5. Future Improvements

##### 5.1 Short-term (Q4 2025)
- [ ] Complete evaluation pipeline implementation
- [ ] Integrate or remove unused utility functions
- [ ] Add data quality monitoring hooks
- [ ] Implement model versioning strategy

##### 5.2 Medium-term (Q1 2026)
- [ ] Migrate to Kedro 2.x when stable
- [ ] Add distributed processing with Spark
- [ ] Implement A/B testing framework
- [ ] Add real-time inference pipeline

##### 5.3 Long-term (2026)
- [ ] Multi-model ensemble support
- [ ] AutoML integration
- [ ] Cloud deployment patterns
- [ ] CI/CD pipeline automation

## MLflow Integration Architecture

### Configuration Structure
```yaml
# conf/base/mlflow.yml
server:
  mlflow_tracking_uri: mlruns    # Local tracking store
tracking:
  experiment:
    name: "account_tax_experiment"
  run:
    name: "${km.random_name:}"   # Dynamic run naming
    nested: false
  params:
    dict_params:
      flatten: true              # Flatten nested params
      recursive: true
      sep: "."
```

### MLflow Dataset Integration

#### Artifact Storage Pattern
```yaml
# MlflowArtifactDataset wrapper pattern
prepared_datasets_mlflow:
  type: kedro_mlflow.io.artifacts.MlflowArtifactDataset
  dataset:
    type: account_tax.datasets.partitioned_parquet.PartitionedParquetDataset
    path: data/05_model_input/prepared_datasets
  artifact_path: data/prepared_datasets
```

#### Integration Points
1. **Parameter Logging**: Automatic from `parameters/*.yml`
2. **Artifact Storage**: Key datasets saved as MLflow artifacts
3. **Metrics Tracking**: Currently manual (evaluation pipeline incomplete)
4. **Model Registry**: Not yet implemented (future enhancement)

### Storage Strategy Options

| Strategy | Use Case | Configuration |
|----------|----------|---------------|
| **mlflow_only** | Cloud/distributed teams | All artifacts in MLflow |
| **local_only** | Development/debugging | No MLflow tracking |
| **both** (current) | Hybrid approach | Local + MLflow artifacts |

## Data Catalog Architecture

### Memory Management Strategy

#### Dataset Types by Layer
```
01_raw/        : ParquetDataset (persistent)
02_intermediate/: MemoryDataset (transient)
03_primary/    : MemoryDataset (transient)
04_feature/    : MemoryDataset (transient)
05_model_input/: MlflowArtifactDataset (persistent)
06_models/     : MlflowArtifactDataset (persistent)
08_reporting/  : MlflowArtifactDataset (persistent)
```

#### Memory Optimization Patterns
1. **MemoryDataset**: Used for intermediate transformations
2. **Garbage Collection**: Automatic between pipeline stages
3. **Checkpointing**: MLflow artifacts at key stages
4. **Lazy Loading**: Parquet files loaded on demand

### Custom Dataset Implementation

#### PartitionedParquetDataset
- **Purpose**: Handle DatasetDict as partitioned parquet files
- **Location**: `src/account_tax/datasets/partitioned_parquet.py`
- **Features**:
  - Saves train/validation/test splits as separate parquet files
  - Maintains partition metadata
  - Integrates with MLflow artifact storage

## Configuration Patterns

### Multi-Environment Support
```
conf/
├── base/           # Default configuration
│   ├── catalog.yml
│   ├── mlflow.yml
│   └── parameters/
│       ├── data_pipeline.yml
│       ├── train_pipeline.yml
│       └── inference_pipeline.yml
├── repro/          # Reproducibility configs
│   ├── catalog.yml (overrides)
│   └── parameters/ (fixed seeds)
└── local/          # Local dev (gitignored)
```

### Parameter Hierarchy
1. **Global**: `conf/base/globals.yml` (BRANCH, AS_OF)
2. **Pipeline**: `parameters/{pipeline}_pipeline.yml`
3. **Node**: Specific parameters within pipeline configs
4. **Runtime**: CLI overrides via `--params`

---

본 섹션은 설계자·플래너·개발자가 동일한 함수 블록 정보를 공유하기 위한 최신 기준입니다. 새로운 함수를 추가하거나 책임을 변경할 때는 해당 표를 즉시 갱신해 블록화 원칙을 유지하십시오.
