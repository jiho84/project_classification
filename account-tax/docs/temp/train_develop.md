# Train Pipeline Configuration Guide

이 문서는 `conf/base/parameters/train_pipeline.yml`(및 `conf/repro/...`)에 설정해야 할 항목을
범주별로 정리하고, 예시 YAML 스켈레톤을 제공합니다. 대규모 모델(Qwen 계열)을 LoRA +
Deepspeed 조합으로 학습하는 시나리오를 기준으로 작성했습니다.

---

## 1. Split 관련 파라미터
| 키 | 설명 |
|----|------|
| `split.label_column` | 분류 라벨 컬럼명 (예: `acct_code`) |
| `split.seed` | 데이터 분리 랜덤 시드 |
| `split.test_size`, `split.val_size` | train/valid/test 비율 |
| `split.max_classes` | 라벨 슬롯 총량 (필수) |
| `split.dummy_prefix`, `split.dummy_label` *(선택)* | 추가 슬롯 이름 제어 |
| `split.labelize_num_proc` | 라벨 인코딩 시 병렬 프로세스 수 |

---

## 2. Tokenization 설정
| 키 | 설명 |
|----|------|
| `train.tokenization.model_name` | Hugging Face 모델 이름 |
| `train.tokenization.max_length` | 입력 토큰 최대 길이 |
| `train.tokenization.truncation` | truncate 여부 |
| `train.tokenization.padding`(선택) | 패딩 방식 (`"max_length"`, `"longest"`) |
| `train.tokenization.num_proc` | 토크나이징 병렬 수 (`num_proc`) |
| `train.tokenization.diagnostics` | 토큰 길이 분석 노드용 매개변수<br>(`sample_size`, `percentiles`, `seed`) |

---

## 3. 분류 헤드 / 모델 초기화
| 키 | 설명 |
|----|------|
| `train.model.num_labels` | 분류 클래스 수 |
| `train.model.problem_type` | 예: `single_label_classification` |
| `train.model.id2label` / `label2id` | 정수-라벨 매핑 딕셔너리 |
| `train.model.dropout` *(선택)* | 분류 헤드 dropout |
| `train.model.gradient_checkpointing` | Hugging Face gradient checkpointing 사용 여부 |

> 모델 차원의 gradient checkpointing은 Hugging Face `Trainer` 옵션입니다. Deepspeed를 함께 사용할 때도 이 플래그가 먼저 적용되고, Deepspeed config에 동일 옵션이 있으면 config가 최종값을 덮어씁니다.

---

## 4. LoRA / PEFT 설정
| 키 | 설명 |
|----|------|
| `train.lora.enable` | LoRA 사용 여부 |
| `train.lora.r` | 랭크 r |
| `train.lora.lora_alpha`, `train.lora.lora_dropout` | 스케일/드롭아웃 |
| `train.lora.target_modules` | LoRA를 적용할 모듈 이름 리스트<br>(예: `['q_proj', 'k_proj', 'v_proj', 'o_proj']`) |
| `train.lora.bias` | bias 처리 방식 (`none`, `lora_only`, `all`) |
| `train.lora.task_type` | `SEQ_CLS`, `CAUSAL_LM` 등 |

---

## 5. Deepspeed 설정
| 키 | 설명 |
|----|------|
| `train.deepspeed.enable` | Deepspeed 사용 여부 |
| `train.deepspeed.config_path` | 외부 JSON 설정 경로 |
| `train.deepspeed.stage` | ZeRO stage (예: `2` 또는 `3`) |
| `train.deepspeed.offload` | CPU/NVMe 오프로딩 여부 |

※ 세부 항목은 Deepspeed config 파일을 별도로 관리하고, 여기서는 경로·옵션만 명시하는 방식을 권장합니다. YAML과 config JSON에 동일 키가 존재하면 **config JSON이 최종 오버라이드** 합니다.

---

## 6. 기타 학습 파라미터
| 키 | 설명 |
|----|------|
| `train.training.batch_size` | 학습 배치 크기 |
| `train.training.eval_batch_size` | 평가 배치 크기 |
| `train.training.learning_rate` | 기본 학습률 |
| `train.training.epochs` | 총 Epoch 수 |
| `train.training.warmup_ratio` or `warmup_steps` | 러닝레이트 워밍업 |
| `train.training.weight_decay` | 가중치 감쇠 |
| `train.training.mixed_precision` | `bf16`, `fp16`, `no` 등 |
| `train.training.gradient_accumulation_steps` | 그래디언트 누적 |
| `train.training.logging_steps` | 로깅 주기 |
| `train.training.save_steps` | 체크포인트 저장 주기 |

> 참고: Deepspeed를 사용하는 경우 learning rate, gradient accumulation 등 일부 값이 config 파일에서 다시 정의될 수 있습니다. 충돌 시 Deepspeed config가 최종값으로 적용됩니다.

---

## 7. 예상 `train_pipeline.yml` 스켈레톤

```yaml
split:
  label_column: acct_code
  seed: 42
  test_size: 0.2
  val_size: 0.1
  max_classes: 1000
  dummy_prefix: dummy
  dummy_label: dummy1
  labelize_num_proc: 4

train:
  serialization:
    text_columns: []
    separator: ", "
    include_column_names: true
    num_proc: 4
    retain_columns:
      - text
      - labels
      - acct_code
    label_column: acct_code

  tokenization:
    model_name: "Qwen/Qwen2.5-0.5B"
    max_length: 512
    truncation: true
    padding: "max_length"
    num_proc: 4
    diagnostics:
      sample_size: 5
      percentiles: [50, 75, 90, 95, 99]
      seed: 42

  model:
    num_labels: 150
    problem_type: single_label_classification
    dropout: 0.1
    gradient_checkpointing: true
    id2label: {}
    label2id: {}

  lora:
    enable: true
    r: 8
    lora_alpha: 32
    lora_dropout: 0.05
    bias: none
    target_modules:
      - q_proj
      - k_proj
      - v_proj
      - o_proj
    task_type: SEQ_CLS

  deepspeed:
    enable: true
    config_path: conf/base/deepspeed/zero2.json
    stage: 2
    offload: false

  training:
    batch_size: 8
    eval_batch_size: 8
    learning_rate: 2.0e-5
    epochs: 3
    weight_decay: 0.01
    warmup_ratio: 0.1
    gradient_accumulation_steps: 4
    mixed_precision: bf16
    logging_steps: 100
    save_steps: 500
```

> 필요에 따라 LoRA 또는 Deepspeed 섹션을 비활성화할 수 있습니다 (`enable: false`).
> `id2label`/`label2id`는 실제 라벨 맵이 결정된 후 자동 생성하거나 파이프라인에서 주입하세요.

---

이 스켈레톤을 기반으로 `conf/base/parameters/train_pipeline.yml` 및 `conf/repro/...`에 환경별 값을 채워넣으면, 학습 파이프라인 전체가 재현 가능한 설정으로 정비됩니다.

---

## 8. Train Pipeline Node Layout (권장)

학습 파이프라인에서 모델 관련 노드를 다음과 같이 모듈화하면 시나리오별 재사용이 쉽습니다.

1. **`instantiate_model`**  
   - **입력**: `train.model.*`, Hugging Face config/tokenizer  
   - **출력**: 기본 모델 (LoRA 미적용)  
   - **설명**: 모델 가중치를 로드하고 `num_labels`, `problem_type`, dropout, gradient checkpointing 등을 적용한다.

2. **`apply_peft_adapters`** *(옵션)*  
   - **입력**: base model, tokenizer, `train.lora.*`  
   - **출력**: LoRA/어댑터 적용 모델  
   - **설명**: LoRA/PEFT 설정이 활성화됐을 때만 저차원 어댑터를 장착하고, 비활성화면 입력 모델을 그대로 반환한다.

3. **`configure_optimizer`**  
   - **입력**: (LoRA 적용 후) 모델, `train.training.*`  
   - **출력**: optimizer, scheduler  
   - **설명**: learning rate, weight decay, warmup 등을 기반으로 학습 루프에 전달할 옵티마이저/스케줄러 객체를 생성한다.

4. **`configure_deepspeed`** *(옵션)*  
   - **입력**: 모델, optimizer/scheduler, `train.deepspeed.*`, Deepspeed config JSON  
   - **출력**: Trainer/런타임이 사용할 Deepspeed 설정 또는 엔진  
   - **설명**: Deepspeed runtime을 감싸며, 동일 키가 config JSON에 있으면 JSON이 최종 오버라이드한다.

5. **`run_training`**  
   - **입력**: 모델(Deepspeed/LoRA 포함), dataloaders, optimizer/scheduler, 학습 파라미터  
   - **출력**: 학습된 모델, training/validation metric  
   - **설명**: Hugging Face `Trainer` 또는 커스텀 루프를 실행해 학습·체크포인트·MLflow 로깅을 담당한다.

6. **`evaluate_model`** *(선택)*  
   - **입력**: 학습된 모델, 평가 데이터셋, metric 함수  
   - **출력**: 평가 결과(accuracy/F1 등)  
   - **설명**: 추론/검증 파이프라인에서 재사용 가능한 평가 단계로, MLflow metric 기록이나 리포트 생성에 활용한다.

**노드별 장점**
- **다단계 래핑**: 모델 → LoRA → Deepspeed 순으로 래핑하면 중간 단계에서 모델 상태를 저장하거나 rollback하기 수월합니다.
- **옵티마이저 분리**: 학습률, warmup 등은 Deepspeed config와 충돌할 수 있으므로 이 노드에서 “config JSON 우선” 로직을 한 번에 제어할 수 있습니다.
- **평가 분리**: 학습 후 평가를 독립 노드로 유지하면 카탈로그/MLflow 기록을 재사용해 추론 파이프라인이나 A/B 테스트에 쉽게 연결할 수 있습니다.

**참고 사항**
- LoRA·Deepspeed를 사용하지 않는 경우 해당 노드를 pass-through로 구현하면 동일 파이프라인으로 다양한 실험을 처리할 수 있습니다.
- 토크나이저, 라벨 맵 등은 추론 단계에서도 필요하므로, `train.tokenization`과 `train.model` 설정을 inference config와 공유하거나 별도 공통 파일로 분리하도록 권장합니다.

LoRA·Deepspeed를 사용하지 않는 경우 해당 노드를 pass-through로 구현해 동일 파이프라인을 다양한 실험에 유연하게 적용할 수 있습니다. 또한 inference 단계에서도 `train.tokenization`과 `train.model` 설정을 공유해야 하므로, 공통 파라미터 블록을 별도 config로 분리하는 것을 권장합니다.
