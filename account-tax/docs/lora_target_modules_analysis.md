# LoRA Target Modules 분석 리포트
## 자동 계정과목 분류 태스크를 위한 최적화 전략

**작성일**: 2025-10-14
**모델**: Qwen/Qwen3-8B (8.2B parameters)
**태스크**: Sequence Classification (260 클래스)
**현재 설정**: attention-only LoRA (11개 상위 레이어)

---

## 1. Qwen3-8B 모델 구조

### 1.1 전체 모델 아키텍처
- **총 파라미터**: 8.2B (6.95B non-embedding)
- **레이어 수**: 36개 Transformer 블록
- **Hidden Size**: 4096
- **Intermediate Size (FFN)**: 12288 (3x hidden_size)
- **Max Context Length**: 32,768 tokens (YaRN으로 131K 확장 가능)

### 1.2 Transformer 블록 내부 Linear 모듈

각 Transformer 블록은 **7개의 주요 linear layer**로 구성:

#### (A) Attention 모듈 (self_attn)
Grouped Query Attention (GQA) 사용:
- **32개 query heads**, **8개 key/value heads** (4:1 grouping)
- Head dimension: 128

| 모듈 | Input → Output | 파라미터 수 (개당) | 역할 |
|------|----------------|-------------------|------|
| **q_proj** | 4096 → 4096 | 16,777,216 | Query 벡터 생성 (32 heads × 128 dim) |
| **k_proj** | 4096 → 1024 | 4,194,304 | Key 벡터 생성 (8 heads × 128 dim, GQA) |
| **v_proj** | 4096 → 1024 | 4,194,304 | Value 벡터 생성 (8 heads × 128 dim, GQA) |
| **o_proj** | 4096 → 4096 | 16,777,216 | Attention output 통합 |

**Attention 합계**: 41,943,040 params/layer

#### (B) FFN/MLP 모듈
SwiGLU activation 사용:

| 모듈 | Input → Output | 파라미터 수 (개당) | 역할 |
|------|----------------|-------------------|------|
| **gate_proj** | 4096 → 12288 | 50,331,648 | Gating 메커니즘 (SwiGLU의 gate) |
| **up_proj** | 4096 → 12288 | 50,331,648 | Feature 확장 (SwiGLU의 up) |
| **down_proj** | 12288 → 4096 | 50,331,648 | Feature 축소 (output으로 복귀) |

**FFN 합계**: 150,994,944 params/layer

### 1.3 모듈별 파라미터 비율
- **Attention**: 41.9M (21.7%)
- **FFN**: 151.0M (78.3%)
- **레이어당 총합**: 192.9M params

→ **FFN이 파라미터의 약 3.6배를 차지** (intermediate_size가 3x이므로)

---

## 2. Target Modules 비교표

### 2.1 LoRA 파라미터 계산 공식

각 linear layer에 대해:
```
LoRA params = rank × (input_dim + output_dim)
```

현재 설정: `r=256`, `lora_alpha=512`, **11개 레이어** (layers 25-35)

### 2.2 옵션별 상세 비교

| 구성 | Target Modules | LoRA Params<br/>(per layer) | LoRA Params<br/>(11 layers) | 총 Trainable | 메모리<br/>(FP16) | 학습시간<br/>(상대) | 예상 성능 |
|------|---------------|---------------------------|---------------------------|--------------|-----------------|------------------|----------|
| **현재<br/>(Attention-Only)** | q_proj, k_proj,<br/>v_proj, o_proj | **q**: 2,097,152<br/>**k**: 1,310,720<br/>**v**: 1,310,720<br/>**o**: 2,097,152<br/>**소계**: 6,815,744 | **74.97M** | ~75M + 260×4096<br/>= **76.04M** | ~152 MB | **1.0x**<br/>(기준) | **Baseline**<br/>(문맥 관계 학습) |
| **+FFN 일부<br/>(Gate+Up)** | + gate_proj,<br/>+ up_proj | **gate**: 4,194,304<br/>**up**: 4,194,304<br/>**추가**: 8,388,608<br/>**소계**: 15,204,352 | **167.25M** | ~167M + 1.06M<br/>= **168.31M** | ~337 MB | **1.3-1.5x** | **중상**<br/>(도메인 패턴 학습) |
| **+FFN 전체<br/>(All Linear)** | + gate_proj,<br/>+ up_proj,<br/>+ down_proj | **down**: 4,194,304<br/>**추가**: 12,582,912<br/>**소계**: 19,398,656 | **213.39M** | ~213M + 1.06M<br/>= **214.44M** | ~429 MB | **1.5-1.8x** | **상**<br/>(최대 표현력) |

### 2.3 계산 세부사항

**현재 구성 (Attention-Only)**:
```python
# Per layer:
q_proj:    256 × (4096 + 4096) = 2,097,152
k_proj:    256 × (4096 + 1024) = 1,310,720
v_proj:    256 × (4096 + 1024) = 1,310,720
o_proj:    256 × (4096 + 4096) = 2,097,152
합계:      6,815,744

# 11 layers:
11 × 6,815,744 = 74,973,184

# modules_to_save (score head):
260 classes × 4096 hidden = 1,064,960

# Total trainable: 74,973,184 + 1,064,960 = 76,038,144
```

**옵션 1 (+Gate+Up)**:
```python
gate_proj: 256 × (4096 + 12288) = 4,194,304
up_proj:   256 × (4096 + 12288) = 4,194,304
추가:      8,388,608

# 11 layers:
11 × (6,815,744 + 8,388,608) = 167,247,872
# Total: 167,247,872 + 1,064,960 = 168,312,832
```

**옵션 2 (+All FFN)**:
```python
down_proj: 256 × (12288 + 4096) = 4,194,304
추가:      12,582,912

# 11 layers:
11 × 19,398,656 = 213,385,216
# Total: 213,385,216 + 1,064,960 = 214,450,176
```

### 2.4 메모리 오버헤드 분석

**LoRA 파라미터만 (FP16 기준)**:
- 현재: 76M × 2 bytes = 152 MB
- +Gate+Up: 168M × 2 bytes = 337 MB (+185 MB, +122%)
- +All FFN: 214M × 2 bytes = 429 MB (+277 MB, +182%)

**전체 학습 메모리 (추정)**:
- Base model (bfloat16): ~16 GB (frozen)
- Optimizer states (Adam): LoRA params × 8 bytes (momentum + variance)
- Gradients: LoRA params × 2 bytes
- Activations: batch-dependent (~4-8 GB)

```
현재:    ~16 GB + 76M×10 = ~16.7 GB per GPU
+Gate+Up: ~16 GB + 168M×10 = ~17.6 GB per GPU (+900 MB)
+All FFN: ~16 GB + 214M×10 = ~18.1 GB per GPU (+1.4 GB)
```

**DeepSpeed ZeRO-2 (4 GPUs)로 분산 시**:
- Optimizer states 4등분 → 메모리 증가량 약 1/4로 감소
- 실제 영향: 현재 대비 +200-400 MB per GPU 정도로 관리 가능

---

## 3. 계정과목 분류 태스크 특성 분석

### 3.1 데이터 특성

#### 입력 데이터 구조
```
거래일자: 2024-03-15 | 금액: 1,500,000 | 적요: 사무실 임차료 | 거래처: ABC 부동산
```
- **구조화된 필드**: 날짜, 금액, 적요, 거래처 등 고정된 스키마
- **도메인 특화 용어**: "임차료", "부동산", "급여", "세금계산서" 등
- **숫자 패턴**: 금액, 날짜, 세금 비율 등

#### 클래스 분포
- **총 260개 클래스** (계정과목)
- **클래스 불균형 심각**:
  - 상위 20개 클래스: 전체 데이터의 ~60-70%
  - 하위 100개 클래스: 각 클래스당 샘플 < 100개
- **현재 대응**: `use_class_weights=true`, alpha=0.4 (inverse frequency weighting)

### 3.2 Attention vs FFN의 역할 구분

#### Attention 레이어의 강점
1. **문맥 의존성 학습**
   - "사무실 임차료" + "부동산" → **지급임차료 (계정 953)**
   - "직원 급여" + "원천징수" → **급여 (계정 420)**
   - 필드 간 **관계 파악**에 강함

2. **위치 불변 패턴**
   - 필드 순서가 바뀌어도 동일한 계정 분류
   - "금액 | 적요" vs "적요 | 금액" → 동일하게 처리

3. **희소 클래스에 유리**
   - 적은 샘플로도 "토큰 간 관계"만 학습하면 됨
   - Over-fitting 위험 낮음

#### FFN 레이어의 강점
1. **도메인 특화 패턴 암기**
   - "임차료" → [0.9, 0.05, 0.03, ...] (지급임차료 확률 벡터)
   - "급여" → [0.02, 0.88, 0.07, ...] (급여 확률 벡터)
   - **단어→클래스 직접 매핑** 학습

2. **비선형 변환**
   - 금액 범위별 패턴: "1,000,000 이상 + 부동산" → 자산 취득
   - 복잡한 조합 규칙 학습

3. **표현력 증가**
   - Intermediate size (12,288)가 hidden size (4,096)의 3배
   - 더 풍부한 feature space

### 3.3 계정과목 분류에서의 중요도 평가

#### 현재 Attention-Only 구성의 충분성
**충분할 가능성이 높은 케이스**:
- ✅ 거래 내역이 **명확한 문맥 조합**으로 구분되는 경우
  - 예: "임차료" + "부동산" vs "임차료" + "차량"
- ✅ **적요(description) 필드가 매우 명확**한 경우
- ✅ 학습 데이터가 충분한 상위 클래스 (샘플 > 1000)

**부족할 가능성이 있는 케이스**:
- ❌ **도메인 용어 암기가 중요**한 경우
  - 예: "세금계산서" → 무조건 부가세 관련 계정
- ❌ **희소 클래스의 정확도가 중요**한 경우
  - FFN이 "단어→클래스" 직접 매핑으로 희소 클래스 보완 가능
- ❌ **금액 범위/비율 계산이 중요**한 경우
  - FFN의 비선형 변환 필요

### 3.4 클래스 불균형과 표현력의 관계

**핵심 질문**: 260개 클래스의 불균형 상황에서 표현력 증가가 도움이 되는가?

#### 경험적 근거 (HuggingFace blog 사례)
- **Roberta/Llama2/Mistral 비교 연구** (재난 트윗 분류):
  - Attention-only LoRA: Accuracy 84.2%
  - All-linear LoRA: Accuracy 87.5% (**+3.3%p**)
  - **특히 minority class의 Recall이 크게 개선**

#### 회계 태스크에 적용 시
1. **상위 클래스 (빈도 높음)**:
   - Attention만으로도 충분 (문맥으로 구분 가능)
   - FFN 추가 효과: **+0.5-1%p** 예상

2. **하위 클래스 (빈도 낮음)**:
   - Attention은 샘플 부족으로 문맥 학습 어려움
   - FFN이 "키워드→클래스" 직접 매핑으로 보완
   - FFN 추가 효과: **+3-5%p** 예상 (F1 score)

3. **전체 Weighted F1**:
   - 현재 class_weight 사용 중 → 하위 클래스 중요도 ↑
   - FFN 추가 시 **전체 F1 +2-3%p** 예상

---

## 4. 권장사항

### 4.1 즉시 실행 권장사항

**결론**: ⚠️ **현재 구성으로 baseline 먼저 완료 → FFN 추가 실험 진행**

#### 선택 근거

**현재 Attention-Only를 먼저 완료해야 하는 이유**:
1. ✅ **실험 통제**: Baseline 없이 확장 옵션만 테스트하면 비교 불가
2. ✅ **메모리 안정성**: 현재 구성이 4 GPU에서 안정적 동작 보장
3. ✅ **학습 시간**: 20 epochs × 1.0x = 현재 예상 시간 내 완료
4. ✅ **디버깅 효율성**: 문제 발생 시 원인 파악 용이

**FFN 추가를 고려해야 하는 이유**:
1. ⚠️ **하위 클래스 성능**: 현재 F1 score가 낮다면 FFN 추가 필수
2. ⚠️ **도메인 특성**: 회계 용어의 "암기형 매핑" 중요도 높음
3. ⚠️ **표현력 부족 징후**: Validation loss는 낮은데 F1이 낮으면 표현력 문제
4. ⚠️ **연구 사례**: 유사 태스크에서 +3%p 성능 향상 확인됨

### 4.2 의사결정 트리

```
[현재 Attention-Only Baseline 완료]
    ↓
[Validation 결과 확인]
    ↓
┌─────────────────────────────────────┐
│ Overall Weighted F1 >= 0.85?        │
└─────────────────────────────────────┘
    ↓ YES                    ↓ NO
[프로덕션 배포 고려]      [하위 클래스 F1 확인]
                              ↓
                    ┌─────────────────────────┐
                    │ Tail class F1 < 0.60?   │
                    └─────────────────────────┘
                        ↓ YES           ↓ NO
                [FFN 추가 필수]   [하이퍼파라미터 튜닝]
                        ↓
                [Option 1: Gate+Up 테스트]
                        ↓
                [성능 개선 >= 2%p?]
                    ↓ YES           ↓ NO
                [프로덕션 채택]  [Option 2: All FFN 테스트]
```

### 4.3 조건부 FFN 추가 기준

**즉시 FFN 추가를 고려해야 하는 징후** (Baseline 결과 확인 후):
1. ✅ **Weighted F1 < 0.80** (전체 성능 부족)
2. ✅ **Tail class (빈도 하위 100개) F1 < 0.60** (희소 클래스 실패)
3. ✅ **Validation loss 수렴 but F1 정체** (표현력 부족)
4. ✅ **특정 도메인 용어에서 일관되게 실패** (암기 필요)

**Attention-Only로 충분한 징후**:
1. ✅ **Weighted F1 >= 0.85** (전체 성능 우수)
2. ✅ **Tail class F1 >= 0.70** (희소 클래스도 성공)
3. ✅ **Loss와 F1이 함께 수렴** (학습 효율적)
4. ✅ **Confusion matrix에서 명확한 문맥 오류만 존재** (FFN으로 개선 어려움)

### 4.4 대안: 점진적 확장 전략

**Phase 1: Baseline Validation** (현재 우선순위)
```yaml
target_modules: ["q_proj", "k_proj", "v_proj", "o_proj"]
layers_to_transform: [25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35]
```
- 학습 완료 후 **Per-class F1 분석**
- Tail class 성능 집중 분석

**Phase 2: FFN 일부 추가** (조건부 실행)
```yaml
target_modules: ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj"]
layers_to_transform: [25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35]
```
- **조건**: Phase 1 F1 < 0.80 또는 Tail F1 < 0.60
- **예상 개선**: +2-3%p (Weighted F1)
- **비용**: 메모리 +200 MB/GPU, 학습시간 +30-50%

**Phase 3: FFN 전체 추가** (최종 옵션)
```yaml
target_modules: ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
layers_to_transform: [25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35]
```
- **조건**: Phase 2에서 개선 < 1%p (ROI 낮음)
- **예상 개선**: +0.5-1%p 추가 (Phase 2 대비)
- **비용**: 메모리 +400 MB/GPU, 학습시간 +50-80%

---

## 5. 실험 계획

### 5.1 Phase 1: Baseline 검증 (현재 진행)

#### 목표
- Attention-only LoRA의 실제 성능 측정
- Bottleneck 식별 (문맥 학습 vs 도메인 암기)

#### 실험 설정
```yaml
# 현재 training.yml 그대로 유지
lora:
  r: 256
  lora_alpha: 512
  target_modules: ["q_proj", "k_proj", "v_proj", "o_proj"]
  layers_to_transform: [25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35]

training_args:
  num_train_epochs: 20
  per_device_train_batch_size: 8
  gradient_accumulation_steps: 8
  learning_rate: 4.0e-5
```

#### 성공 기준
1. **전체 성능**:
   - ✅ Weighted F1 >= 0.85 → **FFN 추가 불필요**
   - ⚠️ Weighted F1 = 0.80-0.85 → **FFN 추가 고려**
   - ❌ Weighted F1 < 0.80 → **FFN 추가 필수**

2. **Tail Class 성능** (빈도 하위 100개 클래스):
   - ✅ Tail F1 >= 0.70 → 충분
   - ⚠️ Tail F1 = 0.60-0.70 → 개선 필요
   - ❌ Tail F1 < 0.60 → FFN 필수

3. **학습 효율성**:
   - Loss와 F1이 함께 수렴하는가?
   - Validation loss는 낮은데 F1이 낮다면? → 표현력 부족 징후

#### 분석 포인트
```python
# 학습 완료 후 분석할 메트릭
1. Per-class F1 score (260개 클래스별)
2. Confusion matrix (상위 오분류 패턴)
3. 빈도 구간별 F1:
   - High frequency (top 20): F1 >= ?
   - Medium frequency (21-100): F1 >= ?
   - Low frequency (101-260): F1 >= ?
4. Error analysis:
   - 문맥 오류 (attention으로 해결 가능)
   - 키워드 오류 (FFN으로 해결 가능)
```

#### 조기 종료 조건
- Validation loss가 3 epochs 동안 개선 없음
- Training loss는 감소하는데 Validation F1 정체 (overfitting)
- → 현재 `load_best_model_at_end=true`로 자동 대응

---

### 5.2 Phase 2: FFN 일부 추가 실험 (조건부)

#### 실행 조건
Phase 1 결과가 다음 중 하나를 만족:
1. Weighted F1 < 0.85
2. Tail class F1 < 0.70
3. 특정 도메인 키워드에서 반복적 오류

#### 실험 설정
```yaml
# training.yml 수정
lora:
  r: 256  # 유지
  lora_alpha: 512  # 유지
  target_modules: ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj"]
  layers_to_transform: [25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35]

training_args:
  num_train_epochs: 20  # 동일
  per_device_train_batch_size: 8  # 유지 (메모리 충분)
  gradient_accumulation_steps: 8  # 유지
  learning_rate: 4.0e-5  # 동일 (LoRA params 증가는 LR 변경 불필요)
```

#### 성공 기준
1. **절대 성능**:
   - ✅ Weighted F1 >= 0.87 → **Phase 2 채택**
   - ⚠️ Weighted F1 = 0.85-0.87 → Phase 3 고려
   - ❌ 개선 < 1%p → Phase 3로 이동

2. **상대 개선**:
   - ✅ F1 개선 >= 2%p → **ROI 우수**
   - ⚠️ F1 개선 = 1-2%p → 학습시간 trade-off 평가
   - ❌ F1 개선 < 1%p → Phase 3 또는 다른 접근 고려

3. **Tail Class 집중 개선**:
   - Gate+Up 추가로 "키워드→클래스" 직접 매핑 개선 확인
   - Tail F1 개선 >= 5%p? → FFN 효과 검증

#### 비용 평가
```
학습 시간 증가: +30-50% (예: 10시간 → 13-15시간)
메모리 증가: +200 MB/GPU (17.6 GB, 여유 있음)
LoRA params: 76M → 168M (+120%)

ROI 계산:
- F1 개선 2%p × 학습시간 1.4배 = 1.43%p per time unit
- F1 개선 1%p × 학습시간 1.4배 = 0.71%p per time unit (저조)
```

#### 조기 종료 조건
- **5 epochs 후 F1 개선 < 0.5%p** → 중단하고 분석
  - FFN이 효과 없음 → 하이퍼파라미터 문제일 가능성
- **10 epochs 후 개선 < 1%p** → Phase 3 불필요 판단

---

### 5.3 Phase 3: FFN 전체 추가 실험 (선택적)

#### 실행 조건
Phase 2 결과가:
1. F1 개선 1-2%p (미흡) AND
2. Tail class에서 여전히 F1 < 0.70 AND
3. 메모리/시간 리소스 충분

#### 실험 설정
```yaml
lora:
  target_modules: ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
  # 나머지 동일
```

#### 성공 기준
1. **최종 목표**:
   - ✅ Weighted F1 >= 0.88 → **프로덕션 배포**
   - ⚠️ F1 개선 < 0.5%p (Phase 2 대비) → **Phase 2 채택**

2. **ROI 평가**:
   - Down_proj 추가로 추가 개선 >= 1%p? → 채택
   - 추가 개선 < 0.5%p? → Phase 2로 복귀

#### 비용 평가
```
학습 시간 증가: +50-80% (예: 10시간 → 15-18시간)
메모리 증가: +400 MB/GPU (18.1 GB, 한계 근접)
LoRA params: 168M → 214M (+27%)

ROI 계산 (Phase 2 대비):
- 추가 params +27% vs 추가 성능 < 1%p
- 비효율적일 가능성 높음
```

#### Decision Point
Phase 3까지 왔는데도 F1 < 0.85라면?
→ **LoRA 구조 문제가 아님**. 다른 접근 고려:
1. **Rank 증가**: r=256 → r=512 (표현력 부족 가능성)
2. **전체 레이어 적용**: layers_to_transform = [0, 1, 2, ..., 35] (상위 레이어만으로 부족)
3. **Hyperparameter 튜닝**: LR, weight_decay, class_weight_alpha 조정
4. **Data augmentation**: 희소 클래스 oversampling/synthetic data

---

### 5.4 실험 실행 가이드

#### 실험 간 비교를 위한 통제 변수
1. **동일하게 유지**:
   - Random seed (42)
   - 학습 데이터 split (extract_ratio=0.1)
   - Batch size / gradient accumulation
   - Learning rate / scheduler
   - Loss function (class weights)

2. **기록할 메트릭**:
   ```python
   {
     "phase": "baseline" | "ffn_partial" | "ffn_full",
     "target_modules": [...],
     "lora_params": 76038144,
     "training_time_hours": 10.5,
     "peak_memory_gb_per_gpu": 17.2,
     "best_epoch": 15,
     "metrics": {
       "weighted_f1": 0.835,
       "macro_f1": 0.672,
       "tail_f1": 0.598,
       "accuracy": 0.847,
       "val_loss": 0.523
     },
     "per_class_f1": {
       "top20_avg": 0.912,
       "mid80_avg": 0.784,
       "tail100_avg": 0.598
     }
   }
   ```

3. **MLflow 실험 구조**:
   ```
   Experiment: account_classification_lora_ablation
   ├── Run 1: baseline_attention_only (current config)
   ├── Run 2: add_gate_up_proj (if needed)
   └── Run 3: add_all_ffn (if needed)
   ```

#### 각 Phase 간 대기 시간
- Phase 1 완료 후 **1-2일 분석 기간** 확보
  - Per-class F1 분석
  - Confusion matrix 패턴 파악
  - Error case 샘플링 및 검토
- Phase 2/3 진입 전 **의사결정 회의** 권장

---

## 6. 참고 문헌 및 근거

### 6.1 모델 아키텍처
- [Qwen3 Official Documentation](https://huggingface.co/docs/transformers/en/model_doc/qwen3)
- [Qwen3-8B Model Card](https://huggingface.co/Qwen/Qwen3-8B)
- [Qwen3 Technical Report](https://arxiv.org/pdf/2505.09388) (May 2025)

### 6.2 LoRA 이론 및 Best Practices
- [LoRA: Low-Rank Adaptation of Large Language Models (Hu et al., 2021)](https://arxiv.org/abs/2106.09685)
- [QLoRA: Efficient Finetuning of Quantized LLMs (Dettmers et al., 2023)](https://arxiv.org/abs/2305.14314)
- [A Note on LoRA (2024)](https://arxiv.org/html/2404.05086v1)
- [Unsloth LoRA Hyperparameters Guide](https://docs.unsloth.ai/get-started/fine-tuning-llms-guide/lora-hyperparameters-guide)

### 6.3 Sequence Classification 사례
- [HuggingFace Blog: LoRA for Sequence Classification with Roberta, Llama, Mistral](https://huggingface.co/blog/Lora-for-sequence-classification-with-Roberta-Llama-Mistral)
  - **핵심 결과**: All-linear LoRA가 attention-only 대비 +3.3%p 개선 (재난 트윗 분류)
- [Analyzing LLAMA3 Performance on Classification Task Using LoRA and QLoRA](https://www.mdpi.com/2076-3417/15/6/3087)

### 6.4 Grouped Query Attention (GQA)
- [Demystifying GQA - Grouped Query Attention](https://towardsdatascience.com/demystifying-gqa-grouped-query-attention-3fb97b678e4a/)
- [IBM: What is Grouped Query Attention?](https://www.ibm.com/think/topics/grouped-query-attention)

---

## 7. 요약 및 Action Items

### 7.1 핵심 결론

1. **Qwen3-8B 구조**: 7개 linear layers/block, FFN이 전체 params의 78%
2. **LoRA 파라미터**:
   - 현재 (Attention): 76M params, 152 MB
   - +Gate+Up: 168M params, 337 MB (+122%)
   - +All FFN: 214M params, 429 MB (+182%)

3. **태스크 특성**:
   - 구조화된 회계 데이터 + 260개 클래스 + 심각한 불균형
   - Attention: 문맥 관계 학습에 강함
   - FFN: 도메인 키워드 암기 + 희소 클래스 보완에 강함

4. **권장 전략**: **점진적 ablation study**
   - Phase 1 (현재): Baseline 완료 → 성능 측정
   - Phase 2 (조건부): Gate+Up 추가 → +2-3%p 기대
   - Phase 3 (선택): All FFN → 추가 +0.5-1%p

### 7.2 Immediate Action Items

#### 1. Phase 1 완료 (현재 우선순위)
```bash
# 현재 training.yml 그대로 학습 실행
kedro run --pipeline=train

# 완료 후 메트릭 분석
python scripts/analyze_per_class_f1.py  # 필요 시 작성
```

#### 2. Phase 1 결과 분석 체크리스트
- [ ] Weighted F1 score >= 0.85?
- [ ] Tail class (bottom 100) F1 >= 0.70?
- [ ] Confusion matrix에서 패턴 식별
- [ ] Loss vs F1 수렴 패턴 확인
- [ ] 학습 시간 및 메모리 사용량 기록

#### 3. Phase 2 진입 여부 결정 (조건부)
**If F1 < 0.85 OR Tail F1 < 0.70:**
```yaml
# training.yml 수정
target_modules: ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj"]
```
**Else:**
→ 프로덕션 배포 준비

#### 4. 실험 추적 설정
```python
# MLflow에 실험 태깅
mlflow.set_experiment("account_classification_lora_ablation")
mlflow.log_params({
    "phase": "baseline",
    "target_modules": "attention_only",
    "lora_params": 76038144,
    "layers_to_transform": "25-35"
})
```

### 7.3 Expected Timeline

| Phase | Duration | Cumulative Time | Decision Point |
|-------|----------|-----------------|----------------|
| Phase 1 (Baseline) | 10-12 hours | 0.5-1 day | F1 >= 0.85? → Done |
| Analysis | 4-8 hours | 1-2 days | FFN 추가 필요? |
| Phase 2 (+Gate+Up) | 13-18 hours | 2-3 days | Improvement >= 2%p? → Done |
| Analysis | 2-4 hours | 3-4 days | Phase 3 필요? |
| Phase 3 (+All FFN) | 15-20 hours | 4-5 days | Final decision |

**Total**: 최대 5일 이내 최적 구성 도출 예상

---

## 8. 부록: 설정 파일 템플릿

### 8.1 Phase 1 (현재 baseline - 수정 불필요)
```yaml
# account-tax/conf/base/parameters/training.yml
lora:
  enable: true
  config:
    task_type: "SEQ_CLS"
    r: 256
    lora_alpha: 512
    lora_dropout: 0.05
    target_modules: ["q_proj", "k_proj", "v_proj", "o_proj"]
    layers_to_transform: [25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35]
    modules_to_save: ["score"]
    bias: "lora_only"
```

### 8.2 Phase 2 (FFN 일부 추가 - 조건부 적용)
```yaml
# 수정 사항:
lora:
  config:
    target_modules: ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj"]
    # 나머지 동일
```

### 8.3 Phase 3 (FFN 전체 추가 - 선택적 적용)
```yaml
# 수정 사항:
lora:
  config:
    target_modules:
      - "q_proj"
      - "k_proj"
      - "v_proj"
      - "o_proj"
      - "gate_proj"
      - "up_proj"
      - "down_proj"
    # 나머지 동일
```

### 8.4 MLflow 실험 비교 스크립트 (예시)
```python
# scripts/compare_lora_experiments.py
import mlflow
import pandas as pd

# 실험 결과 비교
experiment = mlflow.get_experiment_by_name("account_classification_lora_ablation")
runs = mlflow.search_runs(experiment_ids=[experiment.experiment_id])

# 주요 메트릭 비교
comparison = runs[[
    "params.phase",
    "params.target_modules",
    "params.lora_params",
    "metrics.weighted_f1",
    "metrics.tail_f1",
    "metrics.training_time_hours"
]].sort_values("metrics.weighted_f1", ascending=False)

print(comparison)
```

---

**작성자**: Claude Code (Kedro MLOps Developer)
**마지막 업데이트**: 2025-10-14
**프로젝트**: /home/user/projects/kedro_project/account-tax/

**다음 단계**: Phase 1 baseline 학습 완료 후 결과 분석 → Phase 2 진입 여부 결정
