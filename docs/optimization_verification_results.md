# 최적화 검증 결과

**작성일**: 2025-10-10
**상황**: 12배 속도 차이 수정 후 첫 학습 시도

---

## 🎯 검증 결과

### 초기 학습 메트릭 (Step 10-20)

```
Step 1:  loss=69.53, grad_norm=196.35, lr=1.28e-08
Step 2:  loss=69.25, grad_norm=205.19, lr=2.71e-08
Step 3:  loss=69.33, grad_norm=207.80, lr=4.14e-08
Step 4:  loss=69.20, grad_norm=206.96, lr=5.57e-08
Step 5:  loss=69.80, grad_norm=209.51, lr=7.00e-08
Step 6:  loss=69.71, grad_norm=204.16, lr=8.42e-08
Step 7:  loss=67.83, grad_norm=203.12, lr=9.85e-08
Step 8:  loss=68.18, grad_norm=213.86, lr=1.13e-07
Step 9:  loss=69.08, grad_norm=203.22, lr=1.27e-07
Step 10: loss=69.67, grad_norm=190.84, lr=1.41e-07
...
Step 20: loss=64.19, grad_norm=157.34, lr=2.84e-07
```

---

## 📊 문제 분석

### ❌ 여전히 문제 있는 지표

#### 1. **초기 Loss 여전히 매우 높음**
- **기대값**: 5-6 (random 수준, log(280) ≈ 5.63)
- **실제값**: **69.5**
- **차이**: 12배 높음 (여전히 비정상)

#### 2. **Gradient Norm 여전히 폭발**
- **정상 범위**: 0.1-1.0
- **실제값**: **196-214** (여전히 200배 높음)
- **Gradient Clipping**: 99.5% 잘림 (1.0 / 196 = 0.5%)

#### 3. **Learning Rate가 너무 낮음**
- Warmup 구간 (700 steps): step 20에서 lr = 2.84e-07
- 목표 LR (1e-5)의 **0.003%만 도달**

---

## 🔍 근본 원인 재분석

### 이전 분석의 오류

**이전 가설** (틀림):
- Padding 방식 차이 → 학습 속도 차이 ✅ (맞음)
- 환경 최적화 부족 → 학습 속도 차이 ✅ (맞음)
- 이것들이 gradient explosion 해결 ❌ (틀림!)

**새로운 발견**:
- **Padding 최적화는 속도만 개선**, gradient explosion은 해결 안 됨
- **초기 Loss 69.5는 속도와 무관한 다른 문제**

---

## 💥 진짜 문제: 데이터 문제

### 가설 1: Padding Token 문제

#### 증거
백업 코드 (train.py:249-256):
```python
def tok_fn(batch):
    return tok(
        batch["text"],
        truncation=True,
        max_length=320,
        add_special_tokens=True,
        padding=False  # ← DataCollator가 배치 시 패딩
    )
```

현재 코드 (train/nodes.py:125-133):
```python
def tokenize_function(examples: Dict[str, Any]) -> Dict[str, Any]:
    return tokenizer(
        examples["text"],
        truncation=truncation,
        max_length=max_length,
        add_special_tokens=True,
        padding=False,  # ← 백업과 동일
        return_length=True,
    )
```

**BUT**: 토큰화된 데이터셋 통계
```
train: 2988208 samples, avg text chars: 534.9, max: 620
train: count=2988208, mean=211.5, max=287 (토큰 길이)
```

**문제**:
- `padding=False`로 토크나이징했지만
- **이미 토큰화된 데이터셋에는 padding이 없음**
- DataCollatorWithPadding이 배치 시 패딩하지만, **초기 loss가 높은 것과는 무관**

---

### 가설 2: Classification Head 초기화 (재검토)

#### 이전 테스트 결과
```python
# 단일 샘플 forward pass
Weight std: 0.019893  # 정상 ✅
Single sample loss: 6.2  # 정상 ✅

# 배치 forward pass
ERROR: Cannot handle batch sizes > 1 if no padding token is defined
```

**새로운 의문**:
- 단일 샘플에서는 loss=6.2 (정상)
- 실제 학습에서는 loss=69.5 (비정상)
- **배치 처리 시 문제가 발생**하는 것!

---

### 가설 3: DataCollator 문제

#### 현재 설정 (main_yaml.py:185-188)
```python
collator = DataCollatorWithPadding(
    tokenizer=tokenizer,
    pad_to_multiple_of=8 if training_args.bf16 or training_args.fp16 else None,
)
```

#### 백업 설정 (train.py:269)
```python
collate = DataCollatorWithPadding(tok)
# pad_to_multiple_of 없음
```

**의심**:
- `pad_to_multiple_of=8`: bfloat16 사용 시 8의 배수로 패딩
- 이것이 **과도한 패딩**을 유발하여 loss 폭발?

**검증 필요**:
- DataCollator가 실제로 어떻게 패딩하는지 확인
- Attention mask가 올바른지 확인

---

### 가설 4: 토크나이저 vocab 문제

#### Qwen3-4B 토크나이저 특성
```python
# 백업 코드에서 확인
print(f"PAD token: '{tokenizer.pad_token}' (ID: {tokenizer.pad_token_id})")
print(f"EOS token: '{tokenizer.eos_token}' (ID: {tokenizer.eos_token_id})")

# Qwen3는 기본적으로 PAD와 EOS가 구분되어 있음
if tokenizer.pad_token_id == tokenizer.eos_token_id:
    print("⚠️ 경고: PAD와 EOS 토큰이 동일합니다.")
```

**현재 코드**:
```python
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token
```

**문제 가능성**:
- Qwen3-4B는 원래 PAD 토큰이 있음
- 하지만 위 코드가 **EOS로 덮어씀**
- 이로 인해 **EOS와 PAD가 동일**해져 혼란 발생

---

## 🎯 즉시 확인 사항

### 1. 토크나이저 상태 확인

```python
from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen3-4B", trust_remote_code=True)
print(f"PAD token: {tokenizer.pad_token} (ID: {tokenizer.pad_token_id})")
print(f"EOS token: {tokenizer.eos_token} (ID: {tokenizer.eos_token_id})")
print(f"Vocab size: {tokenizer.vocab_size}")
print(f"Same? {tokenizer.pad_token_id == tokenizer.eos_token_id}")
```

**예상**:
- Qwen3-4B는 PAD와 EOS가 원래 구분됨
- 현재 코드가 잘못 통일시켰을 가능성

---

### 2. 배치 데이터 확인

학습 시작 직후 첫 배치 확인:
```python
# Trainer 내부에서 첫 배치 출력
first_batch = next(iter(train_dataloader))
print(f"input_ids shape: {first_batch['input_ids'].shape}")
print(f"attention_mask shape: {first_batch['attention_mask'].shape}")
print(f"labels shape: {first_batch['labels'].shape}")

# 패딩 확인
print(f"input_ids[0]: {first_batch['input_ids'][0]}")
print(f"attention_mask[0]: {first_batch['attention_mask'][0]}")
```

**확인 사항**:
- input_ids가 과도하게 패딩되었는지
- attention_mask가 올바른지
- labels가 제대로 전달되는지

---

### 3. Model forward pass 디버깅

```python
model.eval()
with torch.no_grad():
    outputs = model(**first_batch)
    logits = outputs.logits
    print(f"Logits shape: {logits.shape}")
    print(f"Logits mean: {logits.mean().item()}")
    print(f"Logits std: {logits.std().item()}")
    print(f"Logits min: {logits.min().item()}")
    print(f"Logits max: {logits.max().item()}")
```

**정상 범위**:
- mean: ~0
- std: 1-10
- min/max: -50 ~ +50

**비정상 (의심)**:
- std > 50 → 초기화 문제
- mean >> 0 → 편향 문제

---

## 🚨 긴급 조치 사항

### 1. 토크나이저 수정 (main_yaml.py)

**현재** (main_yaml.py:165-169):
```python
tokenizer.padding_side = "right"
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.pad_token_id = tokenizer.eos_token_id
    LOGGER.info("Pad token not set; using EOS token %s", tokenizer.pad_token)
```

**수정** (백업 방식):
```python
tokenizer.padding_side = "right"

# Qwen3는 원래 PAD 토큰이 있으므로 조건 확인
if tokenizer.pad_token_id is None:
    LOGGER.warning("⚠️ PAD token not set; using EOS token")
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.pad_token_id = tokenizer.eos_token_id
else:
    LOGGER.info(f"✅ PAD token: {tokenizer.pad_token} (ID: {tokenizer.pad_token_id})")
    LOGGER.info(f"✅ EOS token: {tokenizer.eos_token} (ID: {tokenizer.eos_token_id})")
    if tokenizer.pad_token_id == tokenizer.eos_token_id:
        LOGGER.warning("⚠️ 경고: PAD와 EOS 토큰이 동일합니다. 성능 저하 가능성 있음.")
```

---

### 2. DataCollator 수정 (main_yaml.py:185-188)

**현재**:
```python
collator = DataCollatorWithPadding(
    tokenizer=tokenizer,
    pad_to_multiple_of=8 if training_args.bf16 or training_args.fp16 else None,
)
```

**수정** (백업과 동일):
```python
collator = DataCollatorWithPadding(tokenizer=tokenizer)
# pad_to_multiple_of 제거 (과도한 패딩 방지)
```

---

### 3. 토크나이징 재실행 필요

변경사항:
- train/nodes.py의 tokenize_function은 이미 `padding=False` ✅
- 하지만 **토크나이저 설정**이 달라졌으므로 재실행 필요

---

## 📝 체크리스트

### ✅ 완료된 최적화
- [x] 환경 변수 설정 (MKL, CUDA)
- [x] TF32, Flash Attention 활성화
- [x] DataLoader 최적화 (prefetch, persistent_workers)
- [x] Padding 방식 변경 (False + DataCollator)
- [x] max_length 증가 (320)
- [x] add_special_tokens=True

### ❌ 여전히 문제
- [ ] 초기 loss 69.5 (기대: 5-6)
- [ ] Gradient norm 196 (기대: 0.1-1.0)
- [ ] 학습이 거의 진행 안 됨

### 🔄 다음 작업
- [ ] 토크나이저 PAD/EOS 토큰 확인
- [ ] DataCollator pad_to_multiple_of 제거
- [ ] 배치 데이터 디버깅
- [ ] 모델 forward pass 확인
- [ ] 재학습 후 검증

---

## 🎓 결론

**속도 최적화**: ✅ 성공
- 환경 최적화, DataLoader 최적화 적용 완료
- 학습 속도는 개선되었을 것으로 예상

**Loss/Gradient 문제**: ❌ 미해결
- 초기 loss 69.5 (여전히 12배 높음)
- Gradient norm 196 (여전히 200배 높음)
- **새로운 근본 원인 탐색 필요**:
  1. 토크나이저 PAD/EOS 토큰 통일 문제 (최우선)
  2. DataCollator pad_to_multiple_of 과도한 패딩
  3. 배치 처리 시 데이터 변환 문제
  4. Model config와 토크나이저 불일치

**즉시 조치**:
1. 토크나이저 수정 (PAD/EOS 구분 확인)
2. DataCollator 단순화 (pad_to_multiple_of 제거)
3. 디버깅 코드 추가 (배치 데이터 확인)
4. 재학습 후 검증
