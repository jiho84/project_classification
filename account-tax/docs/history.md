# History Log

> 핵심 사건만 육하원칙(When, Where, Who, What, Why, How)에 따라 요약합니다.
> 기록 전에는 반드시 질문을 통해 사건의 중요도와 맥락을 확인하세요.

### Session 2025-09-24

| When | Where | Who | Context | Why | How |
| --- | --- | --- | --- | --- | --- |
| 2025-09-24T21:10Z | src/account_tax/hooks.py:TimingHook | Architect | 파이프라인/노드 실행 시간을 자동 관측하고 싶음 | 수동 로깅 누락으로 가시성 부족 | TimingHook에서 `mlflow.log_metric` 호출로 파이프라인·노드 타임 기록 |
| 2025-09-24T22:30Z | account_tax/pipelines/feature + split | Architect | Feature와 Split 간 책임 경계가 모호함 | 라벨 슬롯 생성·Dataset 변환이 Feature에 혼재 | Feature는 `prepare_dataset_inputs`, Split은 `create_dataset`/분할/라벨링 담당하도록 재구성 |
| 2025-09-24T23:00Z | docs/ | Architect + Historian | 문서가 여러 파일에 흩어져 역할별 참조 어려움 | 중복 정보로 기준 문서 부재 | `task.md`, `architecture.md`, `review.md`, `analysis.md`, `history.md` 체계를 확립하고 가이드 업데이트 |
| 2025-09-24T23:28Z | account_tax/pipelines/split/nodes.py:serialize_for_nlp | Architect + Developer | 직렬화 노드가 9분 이상 소요되며 병목 발생 | Batched map 미사용으로 순차 실행 | `num_proc=4` 적용 및 불필요 컬럼 제거로 480s→80s 단축 |
| 2025-09-24T23:40Z | conf/base/mlflow.yml | Architect | 단일 run에서 메트릭을 확인하고 싶음 | Nested run으로 부모/자식 분리되어 관리 번거로움 | `tracking.run.nested=false` 설정으로 단일 run 구조 유지 |
| 2025-09-25T03:44Z | conf/base/catalog.yml / src/account_tax/pipelines/split/nodes.py / src/account_tax/datasets/partitioned_parquet.py / notebooks/load_intermediate_outputs.py | Developer | Split 산출물을 Jupyter와 MLflow 모두에서 재사용하고 싶음 | 단일 Parquet 저장과 수동 세션 생성으로 분석이 번거롭고 MLflow 아티팩트 구조가 불일치 | 커스텀 PartitionedParquetDataset을 도입해 MLflow ArtifactDataset과 결합하고, Split 노드가 분할 DataFrame 딕셔너리를 반환하도록 조정, 노트북 헬퍼에 KedroSession 로딩+요약 출력 추가, `kedro run`으로 생성물 검증 |

### Session 2025-09-25

| When | Where | Who | Context | Why | How |
| --- | --- | --- | --- | --- | --- |
| 2025-09-25T06:27Z | account_tax/.viz/* / conf/base/catalog.yml | Historian | Kedro Viz가 어떤 데이터를 근거로 차트와 통계를 보여주는지 명확히 하고자 함 | Viz 사이드바 통계가 N/A로 보이는 원인과 Plotly 시각화 동작을 이해해야 분석 신뢰도를 확보할 수 있음 | `.viz/stats.json`과 `kedro_pipeline_events.json`이 파이프라인 실행 시 생성되는 캐시임을 확인하고, PlotlyDataset이 카탈로그 설정(`plotly_args`)을 이용해 JSON 사양만 저장하며 Viz는 이를 다시 로드해 브라우저에서 렌더링함을 문서화 |
| 2025-09-25T06:27Z | src/account_tax/settings.py / src/account_tax/hooks.py | Developer | MLflow run 수명 주기를 Kedro가 자동으로 관리하게 하고 싶음 | 수동 TimingHook 유지 시 MLflow run이 열리지 않아 메트릭이 남지 않는 문제가 있음 | `hooks.py`의 TimingHook을 제거하고 `settings.py` HOOKS에 `MlflowHook()`을 등록해 파이프라인 실행 시 MLflow run이 자동으로 생성·종료되도록 정비 |
| 2025-09-25T06:57Z | docs/ (history.md) | Historian | Kedro의 관측 가능 단위를 다시 정립하고자 함 | 노드 밖에서 생성되는 임의 계산은 Kedro/MLflow가 자동 추적하지 못해 관리가 어려움 | “노드가 Kedro의 최소 통제 단위”라는 원칙을 정리하고, 노드 경계에 맞춰 Artifact·Metric을 설계하면 Kedro-MLflow 훅으로 자동 로깅이 가능함을 플로우차트와 예시로 문서화 |
| 2025-09-25T07:39Z | docs/ (history.md) | Historian | 노드 출력과 MLflow 저장 구조를 더 체계적으로 이해하고자 함 | 중복 계산을 피하면서 메트릭과 아티팩트를 모두 남기고, `mlruns/` 폴더 구조를 빠르게 파악할 필요가 있음 | “노드 하나 → 다중 산출물(Artifact + Metric)” 패턴으로 TrackingDataset을 활용하면 중복 노드 없이 자동 로깅이 가능함을 명시하고, `mlruns/<experiment>/<run>/` 하위의 `params/`, `metrics/`, `artifacts/`, `tags/`, `meta.yaml` 목적을 학습·데이터 파이프라인 관점에서 정리 |
| 2025-09-25T09:21Z | fine-tuning 연구 노트 / transformers runtime | Historian | Qwen 4B LoRA 미세조정 시 OOM을 막는 전략을 명확히 정리하고자 함 | 순전파 동안 불필요한 activation 저장 때문에 메모리 피크가 발생해 학습이 중단될 위험 | “학습 대상 모듈만 그래프에 남기고 나머지는 `requires_grad=False` + `torch.no_grad()`로 감싸거나 gradient checkpointing을 적용”하는 원칙을 확립하고, LoRA 모듈은 저차원 파라미터만 업데이트하도록 유지해 메모리 사용을 통제하는 절차를 기록 |

#### Kedro ↔ MLflow 관측 흐름 (요약 다이어그램)
```
Kedro Pipeline Run
└─ Node A ──┐
           │   (노드가 반환하는 출력 = Kedro가 통제하는 최소 단위)
           ├─ Node B ──┐
           │           │
           │           └─ Catalog → MlflowArtifactDataset → 자동 log_artifact()
           │
           └─ Node C ──┐
                       └─ Catalog → MlflowMetricDataset → 자동 log_metric()
                         (혹은 노드 내부에서 mlflow.log_metric("node_c/custom", value))
```
- 노드 내부에서만 생성되는 임시 값은 Kedro가 자동으로 인지하지 못한다.
- 따라서 **중요한 산출물은 노드 출력으로 승격**하거나, 노드 안에서 직접 `mlflow.log_*`를 호출해야 한다.
- 기본 원칙: `노드 → 카탈로그 → MlflowDataset` 경로를 설계하면 Kedro-MLflow 훅이 자동으로 로깅한다.

#### 예시
1. `serialize_for_nlp` 노드에서 텍스트 데이터셋을 반환 → 카탈로그 항목 `text_datasets_mlflow`를 `MlflowArtifactDataset`으로 정의 → 파이프라인 실행 시 자동으로 Parquet + MLflow artifact 기록.
2. 같은 노드에서 라벨 커버리지 비율을 계산했다면:
   ```python
   coverage = labelled_rows / total_rows
   mlflow.log_metric("split/serialize_for_nlp/label_coverage", coverage)
   ```
   Hook가 열어 둔 run 안에 메트릭이 저장되어 분류 체계를 유지한다.
```

#### 노드 한 개에서 두 산출물을 반환해 중복 계산 제거하기
```
serialize_for_nlp(base_table) -> (text_dataset_dict, label_coverage)
└─ text_dataset_dict  → Catalog: text_datasets_mlflow (MlflowArtifactDataset)
└─ label_coverage     → Catalog: serialize_metrics (MlflowMetricDataset)
```
- 동일 노드에서 이미 계산한 값을 다시 노드로 분리할 필요 없이, **여러 출력으로 반환**하면 Kedro-MLflow 훅이 각각을 Artifact/Metric으로 기록한다.
- Artifact는 Parquet/JSON 등 파일 형태로, Metric은 단일 스칼라 값으로 저장되며 `mlflow ui`에서 즉시 비교 가능하다.

#### `mlruns/` 디렉터리 구조와 목적
```
mlruns/
└─ <experiment_id>/
   └─ <run_id>/
      ├─ meta.yaml   # run 이름, 시간, 사용자, git SHA 등 요약
      ├─ params/     # log_param으로 기록된 설정값 (데이터 split 기준, 하이퍼파라미터 등)
      ├─ metrics/    # log_metric 값 (데이터 품질, 모델 성능 등)
      ├─ tags/       # 파이프라인 이름, 환경 등 메타데이터
      └─ artifacts/  # 로그된 파일들 (전처리 산출물, 학습된 모델, 리포트 등)
```
- **데이터 준비 파이프라인(run)**: `params/`에 split·라벨 관련 설정, `metrics/`에 데이터 품질 지표, `artifacts/`에 정제된 데이터셋/통계 리포트가 들어간다.
- **학습·추론 파이프라인(run)**: `params/`에 학습 초모수, `metrics/`에 손실·정확도, `artifacts/`에 모델 가중치·토크나이저·추론 결과 파일이 저장된다.
- 이렇게 저장된 구조 덕분에 run을 열어 보면 “무엇을 입력으로 삼았고, 어떤 값을 기록했으며, 어떤 산출물을 남겼는지” 한눈에 추적할 수 있다.

#### OOM 회피를 위한 메모리 제어 레시피
```
Frozen Block (no_grad)
    ↓ detach()
Trainable Block (LoRA)
    ↓ (선택) gradient_checkpointing
Frozen Block (no_grad)
```
- **Freeze + no_grad**: 학습 대상이 아닌 모듈은 `param.requires_grad_(False)` 후 `with torch.no_grad():`로 감싸 forward를 수행하면 activation이 그래프에 저장되지 않는다.
- **LoRA 영역**: 원본 W는 freeze 상태로 두고, 저차원 파라미터 `A`, `B`만 `requires_grad=True`로 유지해 메모리 사용을 최소화한다.
- **그래프 분리**: `hidden = hidden.detach()` 뒤 `hidden.requires_grad_(True)`로 다음 학습 구간을 시작하면 앞 구간의 activation/gradient가 완전히 제거된다.
- **Checkpointing**: 여전히 메모리가 부족하면 Hugging Face `model.gradient_checkpointing_enable()` 또는 `torch.utils.checkpoint`로 LoRA 구간만 재계산 전략을 적용한다.
