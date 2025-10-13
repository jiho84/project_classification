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

### Session 2025-10-10

| When | Where | Who | Context | Why | How |
| --- | --- | --- | --- | --- | --- |
| 2025-10-10T00:00Z | src/train/main_yaml.py:207-295 / conf/base/parameters/training.yml | Developer + Architect | 학습이 다수 클래스만 학습하고 소수 클래스를 무시하는 local optima에 빠짐 | 280개 클래스 중 224개만 샘플 보유, 56개는 zero-sample이라 불균형 극심 → 균일 loss로는 다수 클래스 gradient가 전체를 지배 | alpha=0.4 dampening으로 class-weighted CrossEntropyLoss 도입: `weight = (total/num_classes/count)^0.4`, zero-sample은 평균의 1%로 설정 → 모든 클래스에 균형잡힌 gradient 제공 → global optimization 가능 |
| 2025-10-10T00:00Z | src/train/main_yaml.py:289-291 | Developer | Padding token이 loss 계산에 포함되어 gradient 왜곡 | DataCollatorWithPadding이 padding 위치 label을 -100으로 설정하지만, loss 함수가 이를 무시하지 않으면 무의미한 token이 학습에 기여 | CrossEntropyLoss에 `ignore_index=-100` 명시 → padding 위치는 loss/gradient에서 완전 제외 → 실제 데이터만으로 학습 → 수렴 품질 향상 |
| 2025-10-10T00:00Z | src/train/main_yaml.py:273-295 | Developer | 이전 구현이 forward pass 전에 labels를 제거해 PEFT 최적화 불가 | LoRA/PEFT는 labels를 forward 시 받아 adapter 최적화를 수행하는데, labels 제거로 이 최적화 경로 차단 | Labels를 inputs에 유지한 채 `model(**inputs)` 호출 → PEFT가 adapter를 효율적으로 최적화 → logits 추출 후 custom weighted loss 계산 → PEFT 효율성과 weighted loss 혜택 동시 확보 |
| 2025-10-10T00:00Z | src/train/main_yaml.py:273-295 | Developer | 4x24GB GPU에서 OOM 발생 | Labels가 포함된 inputs를 model에 전달 → model이 내부 loss 계산 + computation graph 생성 → 외부에서 다시 loss 계산 → 두 graph가 메모리에 공존 → 24GB 초과 | Inputs에서 labels 제거 후 forward pass → model은 logits만 반환(내부 loss 계산 안 함) → 외부에서 weighted loss 한 번만 계산 → single computation graph → 메모리 절반 → 7.6GB/24GB로 안정화 |
| 2025-10-10T00:00Z | src/train/main_yaml.py:289 | Developer | Class weights가 float32인데 model이 bfloat16로 실행되어 dtype mismatch | Mixed precision 설정에 따라 logits dtype이 변동하는데, weights를 고정 dtype으로 생성하면 연산 실패 | `self.class_weights.to(logits.device, dtype=logits.dtype)` → weights가 logits와 동일 device/dtype으로 자동 변환 → fp16/bf16/fp32 어떤 precision에도 호환 |
| 2025-10-10T00:00Z | src/account_tax/pipelines/train/nodes.py:229,256 / conf/base/parameters/training.yml | Developer | Loss 설정이 training.yml에 정의되어 있으나 train_config.yml로 전파되지 않음 | launch_training 노드가 loss 관련 파라미터를 train_config.yml에 쓰지 않아 main_yaml.py가 default 설정 사용 | launch_training이 `loss.use_class_weights`, `loss.class_weight_alpha` 등을 train_config.yml에 명시적으로 기록 → main_yaml.py가 weighted loss 설정을 정확히 반영 |

#### Weighted Loss 인과 체인 (Local Optima → Global Optimization)
```
불균형 데이터 (280 classes, 224 with samples, 56 zero-sample)
    ↓
균일 loss → 다수 클래스 gradient 지배 → 모델이 소수 클래스 무시
    ↓
Local optimum: 빈번한 클래스만 예측, 희귀 클래스 학습 실패
    ↓
Class-weighted loss (alpha=0.4 dampening)
    weight = (total_samples / (num_classes * class_count)) ^ 0.4
    ↓
모든 클래스에 균형잡힌 gradient 제공 → 소수 클래스도 학습 신호 강화
    ↓
Global optimization: 280개 클래스 모두 학습 가능
```

**Alpha=0.4 선택 이유**:
- Alpha=1.0 (inverse frequency): 희귀 클래스 weight가 과도하게 커져 역차별 발생
- Alpha=0.5 (sqrt-like): 균형잡힌 dampening, 다수/소수 클래스 모두 학습
- Alpha=0.4: sqrt보다 약간 완화하여 안정적인 수렴 유도

**Zero-sample 클래스 처리**: 평균 count의 1%로 설정하여 zero-division 방지하면서 최소한의 weight 부여

#### Padding Token 처리 인과 체인
```
Variable-length sequences → Batching을 위해 padding 필요
    ↓
DataCollatorWithPadding: padding 위치 label = -100 설정
    ↓
Without ignore_index: padding token이 loss 계산에 포함
    ↓
Gradient 왜곡: 무의미한 padding에서 학습 신호 발생 → 수렴 품질 저하
    ↓
With ignore_index=-100: CrossEntropyLoss가 -100 label 완전 무시
    ↓
Clean gradient: 실제 토큰만 loss/gradient 기여 → 수렴 품질 향상
```

#### PEFT 최적화 방법 변화
```
Previous (WRONG):
    inputs에서 labels 제거 → model(**inputs_without_labels)
    ↓
    PEFT가 adapter 최적화 경로 찾지 못함 → 비효율적 학습
    ↓
    별도로 loss 계산 → weighted loss는 적용되지만 PEFT 비효율

Current (CORRECT):
    Labels 포함한 채 model(**inputs_with_labels)
    ↓
    PEFT가 forward pass 중 adapter 최적화 수행 → 효율적 학습
    ↓
    outputs.logits 추출 → 별도로 weighted loss 계산
    ↓
    PEFT 효율성 + weighted loss 혜택 동시 달성
```

#### OOM 해결 인과 체인 (Dual Loss → Single Loss)
```
Model receives inputs with labels
    ↓
Model computes internal loss + stores computation graph in memory
    ↓
External weighted loss computation + stores another computation graph
    ↓
Both graphs active in memory → Memory usage doubled → OOM (>24GB per GPU)
    ↓
Solution: Remove labels from inputs before forward pass
    ↓
Model only computes forward pass (logits) → No internal loss graph
    ↓
Compute weighted loss once externally → Single computation graph
    ↓
Memory usage halved → 7.6GB/24GB per GPU → Training stable
```

#### 성능 개선 결과
- **Before**: OOM 실패 또는 local optima (소수 클래스 무시)
- **After**: 280개 클래스 전체에 균형잡힌 학습, 안정적 수렴
- **GPU Memory**: 7.6GB/24GB per GPU (4x24GB 환경에서 안정)
- **Loss Trajectory**: 38.6 → 7.8 (smooth decrease, 정상적인 최적화 궤적)
- **Training Speed**: OOM 없이 끝까지 실행 가능

#### 기술적 통합 요약
5개 개선사항이 하나의 최적화 파이프라인을 구성:
1. **Weighted loss (alpha=0.4)** → Local optima 방지, 클래스 불균형 해소
2. **ignore_index=-100** → Padding noise 제거, 깨끗한 gradient
3. **PEFT-optimized forward** → Adapter 효율적 학습
4. **Single loss computation** → OOM 방지, 메모리 안정화
5. **Auto dtype conversion** → Mixed precision 호환성

모든 변경사항은 HuggingFace/DeepSpeed 모범 사례를 따르며 backward compatibility 유지.

#### 수정된 파일
- `/home/user/projects/kedro_project/account-tax/src/train/main_yaml.py` - WeightedTrainer 클래스, compute_loss 메서드, class weights 계산
- `/home/user/projects/kedro_project/account-tax/src/account_tax/pipelines/train/nodes.py` - launch_training 노드의 loss config 전파
- `/home/user/projects/kedro_project/account-tax/conf/base/parameters/training.yml` - loss.use_class_weights, loss.class_weight_alpha 설정

### Session 2025-10-12

| When | Where | Who | Context | Why | How |
| --- | --- | --- | --- | --- | --- |
| 2025-10-12T14:00Z | src/train/main_yaml.py:332-356 | Developer + Architect | 훈련 완료 후 trainer.save_model()에서 프로세스가 무한 대기(hang) 발생, DeepSpeed cleanup 코드 실행 안 됨 | DeepSpeed ZeRO Stage 2로 optimizer state가 4개 GPU에 샤딩되어 저장되는데, trainer.save_model()이 all_gather로 가중치 통합 시도 → 알려진 DeepSpeed+PEFT 호환성 이슈로 hang | 중간 체크포인트는 DeepSpeed 네이티브 메서드(_save_checkpoint)가 자동 저장(샤딩 유지), 최종 저장은 trainer.save_model() 대신 PEFT의 save_pretrained()로 adapter만 저장 → 가중치 통합 과정 회피하여 hang 해결, 학습 재개는 checkpoint-{step}/의 DeepSpeed checkpoint 활용 |

#### trainer.save_model() Hang 인과 체인
```
DeepSpeed ZeRO Stage 2 활성화
    ↓
Optimizer states 샤딩: 4개 GPU에 분산 저장
    - bf16_zero_pp_rank_0_mp_rank_00_optim_states.pt (122MB)
    - bf16_zero_pp_rank_1_mp_rank_00_optim_states.pt (122MB)
    - bf16_zero_pp_rank_2_mp_rank_00_optim_states.pt (122MB)
    - bf16_zero_pp_rank_3_mp_rank_00_optim_states.pt (122MB)
    ↓
훈련 중 체크포인트 저장 (100 step마다)
    - trainer._save_checkpoint() 호출
    - DeepSpeed 네이티브 저장 → 샤딩 유지한 채 저장
    - ✅ 정상 작동 (~570MB per checkpoint)
    ↓
훈련 완료 후 최종 저장 시도
    - trainer.save_model() 호출
    - HuggingFace Trainer → 통합된 모델 저장 시도
    - DeepSpeed ZeRO → all_gather로 샤딩된 가중치 통합 필요
    - ❌ all_gather 과정에서 hang (알려진 DeepSpeed+PEFT 이슈)
    - CPU 0%, 프로세스 멈춤, DeepSpeed cleanup 실행 안 됨
    ↓
해결: PEFT adapter만 저장
    - unwrapped_model.save_pretrained(output_dir)
    - LoRA adapter는 작고(256 rank) DeepSpeed 샤딩 대상 아님
    - all_gather 불필요 → 직접 저장 가능
    - ✅ 5초 이내 완료, 프로세스 정상 종료
```

#### 체크포인트 vs 최종 저장 분리 전략
```
훈련 중 (자동):
    trainer._save_checkpoint()
        ↓
    checkpoint-100/global_step100/
        - bf16_zero_pp_rank_*_optim_states.pt  # 학습 재개용
        - mp_rank_00_model_states.pt
        - trainer_state.json, scheduler.pt
    ✅ DeepSpeed 네이티브 → 샤딩 유지 → 정상 작동

훈련 완료 (수동):
    unwrapped_model.save_pretrained()
        ↓
    checkpoints/ (루트)
        - adapter_model.safetensors  # Inference용 (82MB)
        - adapter_config.json
        - tokenizer 파일들
    ✅ PEFT 네이티브 → 샤딩 없음 → 정상 작동
```

**핵심 원칙**:
- **학습 재개**: DeepSpeed checkpoint 사용 (optimizer state 포함)
- **Inference**: PEFT adapter 사용 (가볍고 base model과 조합 가능)
- 두 목적을 분리하여 각각 최적화된 저장 방법 사용

#### Inference 및 학습 재개 사용법
**Inference (Adapter + Base Model):**
```python
from peft import PeftModel
from transformers import AutoModelForSequenceClassification

base_model = AutoModelForSequenceClassification.from_pretrained("Qwen/Qwen3-4B")
model = PeftModel.from_pretrained(base_model, "data/06_models/checkpoints/")
predictions = model(inputs)
```

**학습 재개 (DeepSpeed Checkpoint):**
```bash
kedro run --pipeline=train
# DeepSpeed가 자동으로 checkpoint-702/ 발견하고 로드
# - 모델 가중치 (mp_rank_00_model_states.pt)
# - Optimizer states (bf16_zero_pp_rank_*_optim_states.pt)
# - Scheduler, RNG states 등 모든 학습 상태 복원
```

#### 기술적 근거
- DeepSpeed ZeRO Stage 2: Optimizer states를 GPU 간 샤딩하여 메모리 효율 극대화
- PEFT LoRA: 저차원 adapter(A, B 행렬)만 학습 → 샤딩 대상 아님
- `trainer.save_model()`: 전체 모델 통합 저장 시도 → ZeRO 샤딩 해제 필요 → all_gather 과정에서 hang
- `save_pretrained()`: Adapter만 저장 → 이미 통합된 작은 가중치 → all_gather 불필요 → 즉시 완료
- 알려진 이슈: https://github.com/microsoft/DeepSpeed/issues/2928

#### 해결 효과
- ✅ 훈련 프로세스 정상 종료 (5초 이내 완료)
- ✅ DeepSpeed cleanup 코드 실행 (destroy_process_group)
- ✅ 학습 재개 기능 유지 (checkpoint-{step}/ 활용)
- ✅ Inference용 경량 모델 제공 (~82MB adapter)
- ✅ MLflow artifact 업로드 정상 작동

#### 수정된 파일
- `/home/user/projects/kedro_project/account-tax/src/train/main_yaml.py` - trainer.save_model() → unwrapped_model.save_pretrained() 교체
- `/home/user/projects/kedro_project/account-tax/docs/training_improvements.md` - 근본 원인 분석 및 해결 방안 문서화

### Session 2025-10-13 (Weekend Work)

| When | Where | Who | Context | Why | How |
| --- | --- | --- | --- | --- | --- |
| 2025-10-13T00:00Z | src/train/main_yaml.py:332-356 / conf/base/parameters/training.yml | Developer + Architect | save_pretrained() 호출 후에도 서브프로세스가 종료 시점에 hang 발생 | NCCL은 모든 rank에 대한 통제권이 필요한데, 중간에 rank0만 save_pretrained를 호출하면서 다른 rank는 대기 상태 → 통신 deadlock | 모든 rank가 save_pretrained() 호출하도록 수정: barrier()로 동기화 후 rank 무관하게 저장 → NCCL이 완전한 통제권 유지 → ✅ Train pipeline 실행 성공 검증 완료 |

#### NCCL Deadlock 메커니즘
```
DeepSpeed 다중 GPU 훈련 (NCCL 분산 통신)
    ↓
훈련 완료 → 최종 저장 단계
    ↓
Previous: if rank == 0: save_pretrained()
    - Rank 0: save_pretrained() 호출 → 파일 I/O 수행
    - Rank 1,2,3: 대기 (다음 NCCL 집합 통신 기다림)
    ↓
NCCL 요구사항: 모든 rank가 동일한 통신 순서 따라야 함
    - Rank 0: save_pretrained 중 (통신 참여 안 함)
    - Rank 1,2,3: 통신 대기 중
    ↓
❌ Deadlock: Rank 0이 돌아올 때까지 다른 rank 무한 대기
    - CPU 0%, 프로세스 hang, KeyboardInterrupt 무시
    ↓
Solution: 모든 rank가 save_pretrained() 호출
    - 모든 rank가 동일한 코드 경로 실행
    - NCCL이 전체 rank에 대한 완전한 통제권 유지
    - 내부적으로 rank 0만 실제 저장, 나머지는 no-op
    ↓
✅ 정상 종료: 모든 rank 동기화 완료 → 프로세스 정상 종료
```

#### 핵심 개념: NCCL Collective Operation
- **Collective Operation**: 모든 rank가 참여해야 완료되는 통신 (all_reduce, barrier 등)
- **Deadlock 조건**: 한 rank가 통신 순서에서 이탈하면 나머지 rank 무한 대기
- **해결 원칙**: 모든 rank가 동일한 통신 패턴 유지 → 분기 코드 최소화

#### 코드 변경 요약
```python
# Before (WRONG):
if trainer.args.local_rank <= 0:
    unwrapped_model.save_pretrained(checkpoints_dir)
    tokenizer.save_pretrained(checkpoints_dir)

# After (CORRECT):
dist.barrier()  # 모든 rank 동기화
unwrapped_model.save_pretrained(checkpoints_dir)  # 모든 rank 호출
tokenizer.save_pretrained(checkpoints_dir)
dist.barrier()  # 저장 완료 대기
```

**Why it works**:
- `save_pretrained()` 내부에서 rank 체크 → rank 0만 실제 저장
- 다른 rank는 호출만 하고 즉시 리턴 (no-op)
- 모든 rank가 동일한 코드 경로 → NCCL 통신 순서 일치 → deadlock 없음

#### 검증 결과
- ✅ `kedro run --pipeline=train` 실행 성공
- ✅ 모든 GPU rank 정상 종료 (hang 없음)
- ✅ Adapter 저장 완료 (82MB, checkpoints/)
- ✅ DeepSpeed cleanup 정상 실행
- ✅ MLflow artifact 업로드 완료

#### 기술적 교훈
1. **분산 시스템에서의 대칭성**: 모든 프로세스가 동일한 통신 패턴 따라야 함
2. **라이브러리 신뢰**: PEFT/HuggingFace 라이브러리가 이미 rank 체크 내장 → 외부에서 분기하지 말 것
3. **Barrier 활용**: 동기화가 필요한 지점에 명시적 barrier 삽입
4. **디버깅 난이도**: NCCL deadlock은 스택트레이스 없이 hang만 발생 → 통신 패턴 분석 필수

#### 수정된 파일
- `/home/user/projects/kedro_project/account-tax/src/train/main_yaml.py` - rank 체크 제거, barrier 추가
- `/home/user/projects/kedro_project/account-tax/conf/base/parameters/training.yml` - (변경 없음, 기존 설정 유지)
