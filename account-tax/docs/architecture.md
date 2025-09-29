# Architecture Notes

- Use this file to record high-level structural decisions.
- Cross-reference detailed diagrams below; maintain this document as the single source of architectural truth.
- Whenever pipelines or module boundaries change, summarize the update here and link to supporting docs.

## 설계 철학 (대칭화 · 모듈화 · 순서화)

- **대칭화(Pattern)**: 동일한 본질의 기능은 유사한 패턴으로 작성하여 구조 파악 시 불필요한 인지 비용을 줄인다. 노드 함수, `pipeline.py` 구성, 문서 구조 모두 일관된 형식을 유지한다.
- **모듈화(Modularity)**: 노드 기반으로 기능을 분리하고, 각 모듈이 명확한 입력·출력 계약을 갖도록 유지한다. 파이프라인은 노드 조합만으로 복잡한 동작을 표현해야 한다.
- **순서화(Ordering)**: 폴더/파일 구조(정적)와 실행 흐름(동적)의 인과를 명확히 기록한다. 문서에는 단계별 순서, 데이터 흐름, 의존 관계를 명시한다.

## 2025-09-24 · Dataset 리팩터링

- Feature 파이프라인 마지막 노드를 `prepare_dataset_inputs`로 단순화하여 `base_table`만 반환하도록 변경.
- Split 파이프라인에 `create_dataset`을 추가해 HuggingFace `Dataset` 생성과 라벨 슬롯 구성을 담당하도록 이동.
- `serialize_for_nlp`가 `retain_columns` 옵션을 사용해 `text`, `labels`, `acct_code`만 유지하도록 정리.
- 상세 흐름과 함수 목록은 아래 “세부 아키텍처” 섹션을 확인하세요.

## 세부 아키텍처

### Account-Tax 블록 아키텍처

설계자는 폴더 → 파일 → 함수 단위의 블록 구성을 유지하면서 데이터 계약과 의존 관계를 추적합니다. 본 문서는 현재 구현 상태를 반영하며, 모든 함수의 책임과 흐름을 `대분류/중분류/함수이름/역할/입력/출력/후속블록` 기준으로 관리합니다.

#### 1. 파이프라인 흐름 요약
- 데이터 플로우: Ingestion → Preprocess → Feature → Split → Train (`serialize_for_nlp` → `tokenize_datasets` → `prepare_for_trainer`)
- 기본 실행: `__default__`는 Ingestion~Split까지 수행하는 `full_preprocess`, `full`은 Train까지 이어져 `trainer_ready_data`를 산출
- Evaluation 파이프라인은 레지스트리에서 비활성화되어 있으며, `evaluation/pipeline.py`가 존재하지 않는 노드를 참조하여 실행 불가 상태

#### 2. 데이터 계약 핵심
- 입력 원천: `raw_account_data` (`data/01_raw/row_data.parquet`) → `pandas.DataFrame`
- 전처리: `standardized_data`~`validated_data`까지 `DataFrame` 유지, 파라미터는 `conf/base/parameters.yml`의 `preprocess.*`
- 특성화: `prepare_dataset_inputs`가 `base_table`(`DataFrame`)을 제공하며 라벨 슬롯은 Split 단계에서 관리
- 분할: `create_dataset`이 HF `Dataset`과 라벨 슬롯을 생성하고, `to_hf_and_split`이 층화 분할을 수행, `labelize_and_cast`가 `ClassLabel` 스키마와 `label_metadata`를 부착해 `prepared_datasets`로 전달
- 학습 준비: `serialize_for_nlp`에서 텍스트 직렬화 → `tokenize_datasets`에서 토큰화 → `prepare_for_trainer`가 `trainer_ready_data`(datasets, collator, model_config)를 최종 산출

#### 3. 함수 인벤토리 (대분류/중분류/함수이름/역할/입력/출력/후속블록)

##### 3.1 Ingestion
| 대분류 | 중분류 | 함수이름 | 역할 | 입력 | 출력 | 후속블록 |
| --- | --- | --- | --- | --- | --- | --- |
| ingestion | nodes.py | load_data | 원본 parquet 검증/로딩 | `raw_account_data` (DataFrame) | `validated_raw_data` (DataFrame) | `standardize_columns` |
| ingestion | nodes.py | standardize_columns | KR→EN 컬럼 매핑 및 정규화 | `validated_raw_data` | `standardized_data` (DataFrame) | `clean_data` |
| ingestion | nodes.py | extract_metadata | 통계·스키마 메타 수집 | `standardized_data` | `ingestion_metadata` (dict) | 외부 모니터링 |
| ingestion | pipeline.py | create_pipeline | Ingestion 노드 결합 정의 | Kedro catalog (`raw_account_data`) | Pipeline(ingestion) | `preprocess.create_pipeline` |

##### 3.2 Preprocess
| 대분류 | 중분류 | 함수이름 | 역할 | 입력 | 출력 | 후속블록 |
| --- | --- | --- | --- | --- | --- | --- |
| preprocess | nodes.py | clean_data | 중복 제거·핵심 컬럼 결측 제거 | `standardized_data`, `params:preprocess.clean` | `cleaned_data` (DataFrame) | `filter_data` |
| preprocess | nodes.py | filter_data | 불필요 컬럼 제외 | `cleaned_data`, `params:preprocess.filter` | `filtered_data` (DataFrame) | `normalize_value` |
| preprocess | nodes.py | normalize_value | 코드값→가독 텍스트 변환 | `filtered_data`, `params:preprocess.code_mappings` | `normalized_data` (DataFrame) | `validate_data` |
| preprocess | nodes.py | validate_data | 금액·날짜 비즈니스 규칙 필터링 | `normalized_data`, `parameters` | `validated_data` (DataFrame) | `build_features` |
| preprocess | pipeline.py | create_pipeline | Preprocess 노드 결합 정의 | Kedro catalog (`standardized_data`, preprocess params) | Pipeline(preprocess) | `feature.create_pipeline` |

##### 3.3 Feature
| 대분류 | 중분류 | 함수이름 | 역할 | 입력 | 출력 | 후속블록 |
| --- | --- | --- | --- | --- | --- | --- |
| feature | nodes.py | add_holiday_features | 공휴일/주말 기반 `day_type` 생성 | `validated_data`, `date_column` | DataFrame(`day_type` 포함) | `build_features` |
| feature | nodes.py | build_features | 파생 컬럼 조합·로깅 | `validated_data`, `params:feature.engineering` | `feature_data` (DataFrame) | `select_features` |
| feature | nodes.py | select_features | 설정 순서대로 특성·라벨 선택 | `feature_data`, `params:feature.selection` | `selected_features` (DataFrame) | `prepare_dataset_inputs` |
| feature | nodes.py | prepare_dataset_inputs | HF 전환용 테이블 정리 | `selected_features`, `params:feature.dataset_conversion` | `base_table` (DataFrame) | `create_dataset` |
| feature | pipeline.py | create_pipeline | Feature 노드 결합 정의 | Kedro catalog (`validated_data`, feature params) | Pipeline(feature) | `split.create_pipeline` |

##### 3.4 Split
| 대분류 | 중분류 | 함수이름 | 역할 | 입력 | 출력 | 후속블록 |
| --- | --- | --- | --- | --- | --- | --- |
| split | nodes.py(helper) | make_label2id | 라벨→ID 매핑 생성 | `label_names` | `Dict[str, int]` | `labelize_and_cast` |
| split | nodes.py(helper) | make_id2label | ID→라벨 매핑 생성 | `label_names` | `Dict[int, str]` | `labelize_and_cast` |
| split | nodes.py | create_dataset | HF Dataset 생성 및 라벨 슬롯 구성 | `base_table`, `params:split` | `hf_dataset`, `label_names` | `to_hf_and_split` |
| split | nodes.py | to_hf_and_split | 층화 분할 | `hf_dataset`, `label_col`, `seed`, `test_size`, `val_size` | `split_datasets` (DatasetDict) | `labelize_and_cast` |
| split | nodes.py | labelize_and_cast | 정수 라벨 부여·`ClassLabel` 캐스팅 | `split_datasets`, `label_names`, `label_col`, `dummy_label`, `labelize_num_proc` | `prepared_datasets` (DatasetDict) | `serialize_for_nlp` |
| split | nodes.py | serialize_for_nlp | 텍스트 직렬화 및 열 축소 | `prepared_datasets`, `params:train.serialization` | `text_datasets` (DatasetDict), `serialized_text_datasets` (artifact) | `tokenize_datasets` |
| split | pipeline.py | create_pipeline | Split 노드 결합 정의 (Dataset 생성·분할·직렬화) | Kedro catalog (`base_table`, split params, `params:train.serialization`) | Pipeline(split) | `train.tokenize_datasets` |

##### 3.5 Train
| 대분류 | 중분류 | 함수이름 | 역할 | 입력 | 출력 | 후속블록 |
| --- | --- | --- | --- | --- | --- | --- |
| train | nodes.py | tokenize_datasets | 토큰화 및 입력 ID 생성 | `text_datasets`, `params:train.tokenization` | `tokenized_datasets` (DatasetDict) | `prepare_for_trainer` |
| train | nodes.py | prepare_for_trainer | HF Trainer 준비 패키지 구성 (현재 파이프라인 미연결) | `tokenized_datasets`, `params:train.trainer_prep` | `trainer_ready_data` (dict) | 외부 Trainer 실행 |
| train | pipeline.py | create_pipeline | Tokenization 전용 파이프라인 (직렬화는 split에서 수행) | Kedro catalog (`text_datasets`, train params) | Pipeline(train) | 모델 학습 스크립트 |

##### 3.6 Evaluation
| 대분류 | 중분류 | 함수이름 | 역할 | 입력 | 출력 | 후속블록 |
| --- | --- | --- | --- | --- | --- | --- |
| evaluation | nodes.py | evaluate_predictions | 분류 지표 계산(sklearn) | `y_true` (Series), `y_pred` (Series) | 지표 dict (accuracy/precision/...) | `generate_evaluation_report` (미구현) |
| evaluation | nodes.py | calculate_tax_impact | 세무 영향 추정 | `predictions` (DataFrame) | `misclassified_amount/percentage` | 리포트/모니터링 |
| evaluation | pipeline.py | create_pipeline | 평가 파이프라인 정의(불완전) | `y_test`, `y_pred`, `y_proba`, `test_predictions`, `tax_categories`, `test_actuals`, `model_info` | Pipeline(evaluation) | ❗ `evaluate_classification_model` 등 미정의로 실행 불가 |

##### 3.7 Utilities & Registry
| 대분류 | 중분류 | 함수이름 | 역할 | 입력 | 출력 | 후속블록 |
| --- | --- | --- | --- | --- | --- | --- |
| utils | utils.py | categorize_accounts | 계정→카테고리 분류 | `df`, `account_mapping` | DataFrame(`account_category`) | 미연결 |
| utils | utils.py | calculate_tax_categories | 키워드 기반 세무 카테고리 | `df` | DataFrame(`tax_category`) | 미연결 |
| utils | utils.py | aggregate_by_period | 기간 단위 집계용 period 계산 | `df`, `period_column`, `aggregation_level` | DataFrame(`period`) | 미연결 |
| registry | pipeline_registry.py | register_pipelines | Kedro 파이프라인 레지스터 | 없음 | `dict[str, Pipeline]` | Kedro CLI, `kedro run` |
| cli | __main__.py | main | Kedro 세션 실행 엔트리포인트 | CLI args | 종료 코드 | 터미널 실행 |

#### 4. 설계-구현 점검
- ✅ Ingestion~Split 파이프라인은 직렬화까지 책임을 이관했고, Train 단계는 토크나이징 이후 작업만 담당하도록 분리되었습니다.
- ⚠️ `evaluation/pipeline.py`가 존재하지 않는 `evaluate_classification_model`, `evaluate_tax_classification`, `generate_evaluation_report`, `calculate_business_metrics` 함수를 참조하여 설계 의도를 위배합니다. 실행하려면 해당 함수 정의 추가 또는 파이프라인 업데이트가 필요합니다.
- ⚠️ `utils.py`의 세 함수는 현재 어떤 파이프라인에도 연결되지 않았습니다. 사용 계획이 없다면 별도 블록으로 이동하거나 문서에서 미연결 상태를 유지할지 결정해야 합니다.
- ✅ Split 단계의 라벨 메타데이터 보존(`label_metadata`)은 Train 단계에서 `ClassLabel` 추출과 fallback 로직이 호환됨을 확인했습니다.
- ⚠️ Split 단계 파라미터(`split.max_classes`, `split.dummy_prefix`)가 라벨 슬롯 크기를 결정하므로, 값 변경 시 `create_dataset`와 후속 노드 제약을 함께 검토해야 합니다.

---

본 섹션은 설계자·플래너·개발자가 동일한 함수 블록 정보를 공유하기 위한 최신 기준입니다. 새로운 함수를 추가하거나 책임을 변경할 때는 해당 표를 즉시 갱신해 블록화 원칙을 유지하십시오.
