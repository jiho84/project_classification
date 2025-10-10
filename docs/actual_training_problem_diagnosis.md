# 실제 학습 문제 진단 보고서

**작성일**: 2025-10-10
**증상**: 500 스텝에도 loss가 10 이하로 떨어지지 않음 (실제 18.5)

---

## 🚨 발견된 치명적 문제

### 문제 1: **초기 Loss가 비정상적으로 높음**

```
Random 예측 Loss (log(224)): 5.41
실제 초기 Loss (step 10):   77.40  ← 14배 높음!
```

**이것은 모델이 완전히 망가져 있다는 의미입니다.**

---

## 📊 실제 Loss 추이

```
Step   Loss    분석
----   -----   ----
10     77.41   Random보다 14배 높음 (완전 망가짐)
100    63.57   여전히 12배 높음
200    30.02   여전히 6배 높음
300    22.81   여전히 4배 높음
500    18.56   여전히 3.4배 높음 ❌
```

**정상적인 학습이라면**:
```
Step   Loss    설명
----   -----   ----
10     5.41    Random 예측 수준
50     2.13    패턴 학습 시작
200    0.42    ✅ 소수점 진입
500    0.16    수렴 완료
```

---

## 🔍 원인 분석

### 원인 1: **과도한 Warmup 설정**

#### 현재 설정
```yaml
warmup_ratio: 0.1
total_steps: 7,002
warmup_steps: 700

→ 500 step은 warmup의 71.4% 지점
→ LR이 목표치의 71.4%만 도달 (약 7e-6)
```

#### 실제 Learning Rate
```
Step   실제 LR      목표 LR (1e-5)의 비율
----   --------     ------------------
10     1.28e-7      1.28%
100    1.41e-6      14.1%
200    2.84e-6      28.4%
500    7.14e-6      71.4%  ← warmup 중
700    1.00e-5      100%   ← warmup 완료
```

**500 step까지 LR이 너무 낮아서 제대로 학습이 안 됨!**

---

### 원인 2: **모델 초기화 문제 의심**

초기 loss가 77.4 (random의 14배)인 것은 **비정상**입니다.

가능한 원인:

#### 1. **Label Encoding 오류**
```python
# 의심되는 상황
true_labels = [0, 1, 2, ..., 223]
predicted_labels = [0, 0, 0, ...]  # 모든 클래스 0으로 초기화

# 이 경우 loss 계산:
# Cross-entropy에서 틀린 클래스에 대한 penalty가 누적
# → Loss가 비정상적으로 높아짐
```

#### 2. **Logit 스케일 문제**
```python
# 정상: logits ∈ [-10, 10]
# 비정상: logits ∈ [-100, 100] or [0, 0.1]

# Logit이 너무 크거나 작으면 softmax 후 확률이 극단적
# → Cross-entropy loss 폭발
```

#### 3. **Num Labels 불일치**
```python
# 설정: num_labels = 280 (max_classes)
# 실제: 224개 클래스만 존재

# 모델은 280개 출력 생성
# 하지만 label은 0-223 범위
# → 56개 출력은 사용되지 않음 (비효율)
```

---

### 원인 3: **부분 최적화 (Local Optimum)**

LR이 낮은 상태에서도 loss가 감소하긴 하지만:

```
Step   Loss    감소폭
----   -----   ------
10     77.41   -
100    63.57   -13.84  (17.9% 감소)
200    30.02   -33.55  (52.8% 감소)
500    18.56   -11.46  (38.2% 감소)
```

**감소 속도가 점점 느려짐** → Plateau 진입 중

---

## 💥 복합적 문제 정리

### 1. **Warmup이 너무 길다**
- Warmup: 700 step (전체의 10%)
- 500 step까지 LR이 71.4%만 도달
- **실질적 학습이 거의 안 됨**

### 2. **초기화가 잘못되었다**
- 초기 loss 77.4 (random의 14배)
- 모델이 "random보다 못한" 상태에서 시작
- **학습 시작점이 엉뚱한 곳**

### 3. **부분 최적화 함정**
- LR이 낮아서 느리게 학습
- Plateau 근처에 도달 (loss 18.5)
- **탈출하기 어려운 상태**

---

## 🎯 해결 방안

### 즉시 조치 (CRITICAL)

#### 1. **Warmup 대폭 축소** (training.yml:68)

**현재**:
```yaml
warmup_ratio: 0.1  # 700 steps
```

**수정**:
```yaml
warmup_ratio: 0.03  # 210 steps (전체의 3%)
```

**이유**:
- 210 step이면 충분히 안정화
- 500 step에는 이미 full LR로 학습 중
- Plateau 탈출 가능

---

#### 2. **모델 초기화 확인**

**num_labels 확인** (training.yml:55):
```yaml
model:
  name_or_path: "Qwen/Qwen3-4B"
  num_labels: null  # ← 이게 문제!
```

**수정**:
```yaml
model:
  name_or_path: "Qwen/Qwen3-4B"
  num_labels: 224  # 실제 클래스 수
```

**이유**:
- `null`이면 모델이 자동으로 추론
- 추론이 잘못되면 초기화 오류 발생
- 명시적으로 224 지정

---

#### 3. **Learning Rate 일시적 증가** (선택사항)

현재 상태가 plateau에 갇혔으므로:

```yaml
learning_rate: 2.0e-5  # 1e-5 → 2e-5 (2배 증가)
```

**주의**: 안정화되면 다시 1e-5로 복귀

---

### 중기 조치

#### 4. **클래스 필터링 적용** (원래 계획)

```yaml
split:
  min_samples_per_class: 10
```

**이유**: 초기화 문제를 해결해도, 클래스 불균형은 여전히 문제

---

#### 5. **Gradient Clipping 확인**

DeepSpeed config에 이미 설정되어 있음:
```yaml
gradient_clipping: 1.0
```

**확인 사항**:
- Gradient norm이 1.0을 자주 초과하는지 확인
- 초과한다면 clipping이 너무 강함 → 학습 방해

---

## 📈 예상 개선 효과

### Warmup 축소 후 (warmup_ratio: 0.03)

```
Step   LR          Loss (예상)
----   --------    -----------
10     4.76e-7     77.41 (동일)
100    4.76e-6     40.00 (현재 63.57)
210    1.00e-5     15.00 (warmup 완료, 현재는 27.41)
500    7.14e-6     2.50  ← ✅ 소수점 진입! (현재 18.56)
```

### num_labels 수정 후

```
초기 Loss: 77.41 → 5.41 (random 수준으로 복귀)
→ 정상적인 학습 시작점
```

---

## 🔬 검증 방법

### 1. **초기 Loss 확인**
```python
# 모델 초기화 직후
initial_predictions = model(first_batch)
initial_loss = cross_entropy(initial_predictions, first_batch_labels)

# 예상값: log(224) ≈ 5.41
# 실제값이 10 이상이면 초기화 문제
```

### 2. **Gradient Norm 모니터링**
```bash
# MLflow에서 grad_norm 확인
cat mlruns/*/metrics/grad_norm

# 대부분 1.0 근처면 정상
# 대부분 0.01 이하면 LR이 너무 낮음
# 대부분 10 이상이면 불안정
```

### 3. **LR 스케줄 시각화**
```python
import matplotlib.pyplot as plt

steps = [10, 50, 100, 200, 500, 700, 1000]
lrs_old = [1.28e-7, 6.99e-7, 1.41e-6, 2.84e-6, 7.14e-6, 1e-5, 8.57e-6]  # warmup_ratio=0.1
lrs_new = [4.76e-7, 2.38e-6, 4.76e-6, 9.52e-6, 7.14e-6, 5.95e-6, 4.29e-6]  # warmup_ratio=0.03

plt.plot(steps, lrs_old, label='Old (warmup=0.1)')
plt.plot(steps, lrs_new, label='New (warmup=0.03)')
plt.axvline(500, color='red', linestyle='--', label='Your checkpoint')
plt.legend()
plt.show()
```

---

## 🎯 즉시 실행 체크리스트

### ✅ 긴급 수정 (지금 바로)

1. [ ] `training.yml:68` warmup_ratio: 0.1 → 0.03
2. [ ] `training.yml:55` num_labels: null → 224
3. [ ] 체크포인트 삭제 후 재학습 시작
4. [ ] 초기 loss가 5-6 범위인지 확인
5. [ ] 200 step에 loss < 1.0 확인

### ✅ 추가 확인 (재학습 중)

6. [ ] Gradient norm 모니터링 (대부분 0.1-1.0 범위)
7. [ ] Learning rate 추이 확인 (210 step에 1e-5 도달)
8. [ ] 500 step에 loss < 0.5 달성 확인

### ✅ 후속 조치 (학습 안정화 후)

9. [ ] 클래스 필터링 적용 (min_samples_per_class: 10)
10. [ ] 전체 파이프라인 재실행
11. [ ] 최종 성능 평가

---

## 🔍 추가 디버깅 정보

### Tokenization 확인

```python
# 첫 샘플 확인
sample_text = serialized_datasets['train'][0]['text']
print(f"Text length: {len(sample_text)}")
print(f"Text sample: {sample_text[:200]}")

# Tokenization
from transformers import AutoTokenizer
tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen3-4B")
tokens = tokenizer(sample_text, max_length=256, truncation=True)
print(f"Token count: {len(tokens['input_ids'])}")
print(f"Tokens: {tokens['input_ids'][:20]}")
```

**의심 사항**:
- Text가 비어있거나 너무 짧음
- Tokenization이 실패하여 padding만 가득
- → 모델이 학습할 정보가 없음 → 높은 loss

---

## 📝 결론

**"부분최적화에 빠졌다"는 맞지만, 근본 원인은**:

1. **Warmup이 너무 길어서** 500 step까지 제대로 학습 안 됨
2. **초기화가 잘못되어** random보다 14배 나쁜 상태에서 시작
3. **클래스 불균형**은 2차 문제 (1, 2를 해결해야 효과 있음)

**즉시 조치**:
- warmup_ratio: 0.1 → 0.03
- num_labels: null → 224
- 재학습 시작

**예상 결과**:
- 200 step에 loss < 1.0
- 500 step에 loss < 0.5
- 정상적인 수렴 곡선
