# 진짜 문제: Gradient Explosion

**작성일**: 2025-10-10
**근본 원인**: Gradient Norm이 200 수준 → Gradient Clipping에 의해 99.5% 잘림

---

## 🔥 발견한 진짜 문제

### Gradient Norm 추이

```
Step   Grad Norm   정상 범위 (0.1-1.0)
----   ---------   -------------------
10     197.22      197배 높음! 😱
20     215.00      215배 높음!
50     186.94      187배 높음
100    141.44      141배 높음
200    57.88       58배 높음
500    ???         (확인 필요)
```

**Gradient Norm이 200 수준이면**:
```python
gradient_clipping = 1.0

# 실제 gradient: 200
# Clipping 후: 200 × (1.0 / 200) = 1.0

→ Gradient의 99.5%가 잘려나감!
→ 학습이 거의 안 됨
```

---

## 💥 왜 Gradient가 폭발하는가?

### 원인 1: **초기 Loss가 너무 높음** (77.4)

Cross-entropy loss에서:

```python
loss = -log(p_correct)

# 정상: 모델이 정답 클래스에 1/280 확률 배정
p_correct ≈ 1/280 = 0.00357
loss = -log(0.00357) = 5.63 (random)

# 비정상: 모델이 정답 클래스에 매우 낮은 확률 배정
p_correct ≈ 1e-34
loss = -log(1e-34) = 78.2

→ Loss가 78이면 gradient도 비례하여 폭발!
```

---

### 원인 2: **Logit 초기화 문제**

모델 초기화 시 logit이 잘못된 스케일:

```python
# 정상 초기화: Xavier/He initialization
logits ∈ [-0.1, 0.1]  # 작은 값
→ softmax 후: 모든 클래스가 비슷한 확률 (1/280 ≈ 0.00357)
→ Loss ≈ 5.63

# 비정상 초기화
logits ∈ [-100, 100]  # 너무 큰 값!
→ softmax 후: 일부 클래스가 1.0, 나머지 0
→ 정답 클래스가 0이면 loss = -log(0) = infinity!
→ 실제로는 수치 안정성으로 매우 큰 값 (77.4)
```

**의심**: LoRA 초기화 또는 Classification Head 초기화 문제

---

## 🔍 증거

### 1. Loss와 Gradient Norm의 상관관계

```
Step   Loss    Grad Norm
----   -----   ---------
10     77.4    197.2
50     75.8    186.9
100    63.6    141.4
200    30.0    57.9

→ Loss가 감소하면 Gradient Norm도 감소
→ 둘 다 연관되어 있음
```

### 2. Gradient Clipping의 영향

```python
Gradient Clipping = 1.0

Step 10:
- 원래 gradient norm: 197.2
- Clipping 후: 1.0
- 감소율: 99.49%

→ 거의 학습이 안 됨
→ Loss가 천천히만 감소
```

### 3. 초기 Loss의 비정상성

```
Random 예측 Loss: 5.63
실제 초기 Loss: 77.4
비율: 13.7배

→ 초기화가 random보다 14배 나쁨
→ 모델이 의도적으로 틀린 예측을 하고 있음
```

---

## 🎯 근본 원인 추정

### 가설 1: **Classification Head 초기화 실패**

```python
# AutoModelForSequenceClassification.from_pretrained()
# 내부적으로 classification head를 추가:

model = Qwen3ForSequenceClassification(
    num_labels=280,
    ...
)

# Classification head (Linear layer):
self.score = nn.Linear(hidden_size, num_labels)

# 문제: 이 layer의 weight 초기화가 잘못됨
# 정상: N(0, 0.02^2)
# 비정상: N(0, 1^2) 또는 초기화 안 됨
```

**검증 방법**:
```python
# 모델 로드 후
print(model.score.weight.std())

# 정상: 0.01 ~ 0.05
# 비정상: 0.5 ~ 10
```

---

### 가설 2: **LoRA와 Classification Head 불일치**

LoRA 설정:
```yaml
lora:
  target_modules: ["q_proj", "k_proj", "v_proj", "o_proj"]
  modules_to_save: ["score"]  # Classification head 저장
```

**문제**:
- LoRA는 Transformer layer만 적용
- Classification head(`score`)는 LoRA 없이 그대로 학습
- 하지만 `modules_to_save`로 지정되어 있음

**의심**:
- `score`가 제대로 초기화되지 않음
- 또는 LoRA weight와 full weight가 충돌

---

### 가설 3: **Gradient Scaling 문제**

DeepSpeed + bfloat16 + LoRA 조합:

```python
# LoRA는 작은 rank로 학습 (r=256)
# 하지만 gradient는 전체 parameter에서 계산
# → Gradient가 비정상적으로 큼

# 예: Full weight의 gradient를 LoRA로 압축할 때
# Scaling이 잘못되면 gradient 폭발
```

---

## 💊 해결 방안

### 1. **Gradient Clipping 완화** (즉시 조치)

**현재** (training.yml:102, deepspeed.yml):
```yaml
gradient_clipping: 1.0  # 너무 강함!
```

**수정**:
```yaml
gradient_clipping: 10.0  # 10배 완화
```

**이유**:
- Gradient norm이 200 수준이므로 clipping을 10으로 올려야 함
- 10으로 올리면 gradient의 95%가 살아남음 (vs 현재 0.5%)
- 학습 속도 20배 증가 예상

---

### 2. **Classification Head 재초기화** (코드 수정 필요)

`main_yaml.py`에서 모델 로드 후:

```python
# Line 173 이후에 추가
model = AutoModelForSequenceClassification.from_pretrained(...)

# Classification head 재초기화
if hasattr(model, 'score'):
    torch.nn.init.normal_(model.score.weight, std=0.02)
    if model.score.bias is not None:
        torch.nn.init.zeros_(model.score.bias)
    LOGGER.info("Re-initialized classification head")
```

**효과**:
- 초기 loss가 5.63 수준으로 정상화
- Gradient norm도 1.0 수준으로 정상화

---

### 3. **Learning Rate 증가** (gradient clipping 완화 후)

Gradient clipping을 10으로 완화한 후:

```yaml
learning_rate: 2.0e-5  # 1e-5 → 2e-5
```

**이유**:
- 현재는 gradient가 99.5% 잘려서 LR 1e-5의 효과가 5e-8 수준
- Clipping 완화하면 실효 LR이 급증
- LR을 2배로 올려서 더 빠른 수렴

---

### 4. **LoRA 설정 검토** (선택사항)

**현재**:
```yaml
lora:
  r: 256
  lora_alpha: 512
  modules_to_save: ["score"]
```

**의심 사항**:
- `modules_to_save: ["score"]`가 제대로 작동하는지 확인
- LoRA와 full parameter가 충돌하지 않는지 확인

**검증**:
```python
# 모델 로드 후
for name, param in model.named_parameters():
    if 'score' in name:
        print(f"{name}: requires_grad={param.requires_grad}, shape={param.shape}")
```

---

## 📊 예상 개선 효과

### Gradient Clipping 완화 (1.0 → 10.0)

```
현재 (clipping=1.0):
Step 10:
- Gradient norm: 197.2 → clipped to 1.0 (0.5% 유지)
- 실효 학습률: 1e-5 × 0.005 = 5e-8
- Loss 감소: 77.4 → 75.8 (2% 감소)

수정 후 (clipping=10.0):
Step 10:
- Gradient norm: 197.2 → clipped to 10.0 (5% 유지)
- 실효 학습률: 1e-5 × 0.05 = 5e-7 (10배 증가)
- Loss 감소 예상: 77.4 → 50.0 (35% 감소, 10배 빠름)
```

### Classification Head 재초기화

```
현재:
- 초기 loss: 77.4 (random의 14배)
- Gradient norm: 197.2

재초기화 후:
- 초기 loss: 5.6 (random 수준)
- Gradient norm: 1.0 (정상)
- 200 step에 loss < 1.0 달성 가능
```

---

## 🔬 디버깅 체크리스트

### 1. Classification Head 상태 확인

```python
import torch
from transformers import AutoModelForSequenceClassification

model = AutoModelForSequenceClassification.from_pretrained(
    "Qwen/Qwen3-4B",
    num_labels=280,
    torch_dtype=torch.bfloat16
)

print("=== Classification Head ===")
print(f"Weight std: {model.score.weight.std().item():.6f}")
print(f"Weight mean: {model.score.weight.mean().item():.6f}")
print(f"Weight min: {model.score.weight.min().item():.6f}")
print(f"Weight max: {model.score.weight.max().item():.6f}")

# 정상: std ≈ 0.02, mean ≈ 0, min/max ∈ [-0.1, 0.1]
# 비정상: std > 0.5 또는 min/max가 극단적
```

### 2. 초기 예측 분포 확인

```python
# 첫 배치로 예측
outputs = model(input_ids=batch['input_ids'], attention_mask=batch['attention_mask'])
logits = outputs.logits

print("=== Logits 분석 ===")
print(f"Logits mean: {logits.mean().item():.2f}")
print(f"Logits std: {logits.std().item():.2f}")
print(f"Logits min: {logits.min().item():.2f}")
print(f"Logits max: {logits.max().item():.2f}")

# 정상: mean ≈ 0, std ≈ 1, min/max ∈ [-5, 5]
# 비정상: std > 10 또는 min/max가 극단적 (±50 이상)
```

### 3. LoRA 상태 확인

```python
from peft import get_peft_model

# LoRA 적용 후
print("=== LoRA Parameters ===")
trainable_params = 0
all_params = 0
for name, param in model.named_parameters():
    all_params += param.numel()
    if param.requires_grad:
        trainable_params += param.numel()
        if 'score' in name:
            print(f"Trainable: {name} - shape: {param.shape}")

print(f"Trainable: {trainable_params:,} / {all_params:,} ({trainable_params/all_params*100:.2f}%)")
```

---

## 📝 즉시 실행 체크리스트

### ✅ 긴급 수정 (지금 바로)

1. [ ] **Gradient Clipping 완화**
   - `training.yml:102` gradient_clipping: 1.0 → **10.0**
   - `deepspeed config` gradient_clipping: 1.0 → **10.0**

2. [ ] **재학습 시작**
   - 기존 체크포인트 삭제
   - 새로 학습 시작
   - 초기 grad_norm 확인 (여전히 200 수준인지)

3. [ ] **50 step 후 확인**
   - Loss가 50 이하로 떨어지는지
   - Grad norm이 50 이하로 떨어지는지

### ✅ 추가 확인 (1단계 실패 시)

4. [ ] **Classification Head 재초기화 코드 추가**
   - `src/train/main_yaml.py:173` 이후에 재초기화 코드 추가
   - 재학습
   - 초기 loss 5-6 범위 확인

5. [ ] **num_labels 명시**
   - `training.yml:55` num_labels: null → **280**
   - (이건 이미 자동으로 280으로 설정되고 있음)

---

## 🎓 결론

**진짜 문제는**:
1. ❌ Warmup이 아님 (백업도 동일)
2. ❌ 클래스 불균형이 아님 (2차 문제)
3. ✅ **Gradient Explosion** (gradient norm 200)
4. ✅ **과도한 Gradient Clipping** (99.5% 잘림)
5. ✅ **초기화 문제** (초기 loss 77.4)

**즉시 조치**:
- Gradient clipping: 1.0 → 10.0
- 재학습 시작
- 50 step에 loss < 50 확인

**예상 결과**:
- 100 step에 loss < 20
- 200 step에 loss < 5
- 500 step에 loss < 1 (소수점 진입!)

---

## 📌 참고: Gradient Clipping 이론

### Gradient Clipping이란?

```python
# Gradient norm이 threshold를 초과하면 스케일 다운
if grad_norm > threshold:
    grad = grad × (threshold / grad_norm)
```

### 적절한 Clipping 값

```
Gradient Norm 범위    권장 Clipping
------------------    --------------
0.1 ~ 1.0             1.0 (정상)
1.0 ~ 10.0            5.0
10.0 ~ 100.0          50.0
100.0 ~ 1000.0        100.0 이상

현재: 200 수준 → 10.0으로 시작, 필요시 50.0까지
```

### Clipping이 너무 강하면

```python
# Gradient의 대부분이 잘림
# → 학습이 거의 안 됨
# → Loss가 천천히만 감소
# → "부분최적화에 빠진 것처럼 보임"

# 실제로는 gradient가 너무 약해서 최적화가 안 되는 것!
```
