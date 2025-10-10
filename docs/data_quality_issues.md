# 데이터 품질 문제 분석 보고서

**작성일**: 2025-10-10
**증상**: 200 스텝 만에 loss가 소수점으로 진입하지 않음 (비정상)

---

## 🚨 발견된 문제

### 1. **극심한 클래스 불균형** (CRITICAL)

#### 불균형 통계
```
총 샘플: 298,821
고유 클래스: 224개 (설정: max_classes 280)
클래스 불균형 비율: 44,098 : 1
```

#### 상위 클래스 (과대표)
| Label | 샘플 수 | 비율 |
|-------|---------|------|
| 197 | 44,098 | 14.76% |
| 177 | 35,707 | 11.95% |
| 62 | 21,683 | 7.26% |
| 178 | 18,958 | 6.34% |
| 79 | 16,768 | 5.61% |

**상위 5개 클래스가 전체의 45.92%를 차지**

#### 하위 클래스 (과소표)
**Train에 단 1개 샘플만 있는 클래스**: 138, 119, 127, 46, 223, 44, 2, 38, 85, 221

**이 클래스들은 사실상 학습 불가능**
- 1개 샘플로는 패턴 학습 불가
- 모델이 이 클래스를 예측하도록 학습되지 않음
- Random chance보다 못한 성능

---

## 💥 학습 성능에 미치는 영향

### Loss가 빠르게 감소하지 않는 이유

1. **다수 클래스 편향**
   - 모델이 Label 197, 177만 예측해도 26.71% 정확도
   - Loss가 plateau에 쉽게 도달
   - Gradient가 소수 클래스에 제대로 전파되지 않음

2. **학습 불안정성**
   - 배치마다 클래스 분포가 달라짐
   - 소수 클래스는 배치에 거의 나타나지 않음
   - 모델이 일관된 패턴을 학습하지 못함

3. **Stratified Split 실패 가능성**
   - 1개 샘플 클래스는 train/valid/test 분리 불가
   - split/nodes.py:159-161에서 random fallback 발생
   - 일부 클래스는 test에만 존재 가능 (zero-shot 평가)

---

## ✅ 해결 방안

### 방안 1: **최소 샘플 수 필터링** (RECOMMENDED)

클래스별 최소 샘플 수를 강제하여 학습 불가능한 클래스 제거

#### split/nodes.py의 `create_dataset` 수정

**현재 코드** (split/nodes.py:62-121):
```python
def create_dataset(
    base_table: pd.DataFrame,
    params: Dict[str, Any],
) -> Tuple[Dataset, List[str]]:
    label_column = params.get("label_column", "acct_code")
    # ... 기존 코드

    cleaned = base_table.reset_index(drop=True)
    dataset = Dataset.from_pandas(cleaned, preserve_index=False)
```

**수정 제안**:
```python
def create_dataset(
    base_table: pd.DataFrame,
    params: Dict[str, Any],
) -> Tuple[Dataset, List[str]]:
    label_column = params.get("label_column", "acct_code")
    max_classes = params.get("max_classes", 100)
    min_samples_per_class = params.get("min_samples_per_class", 10)  # 새로 추가

    # ... extraction 로직 ...

    # 클래스별 샘플 수 필터링
    if min_samples_per_class > 0:
        original_size = len(base_table)
        class_counts = base_table[label_column].value_counts()
        valid_classes = class_counts[class_counts >= min_samples_per_class].index
        base_table = base_table[base_table[label_column].isin(valid_classes)]

        removed_classes = len(class_counts) - len(valid_classes)
        removed_samples = original_size - len(base_table)

        logger.info(
            "Filtered out %d classes with < %d samples (removed %d samples, %.2f%%)",
            removed_classes, min_samples_per_class, removed_samples,
            removed_samples / original_size * 100
        )

    cleaned = base_table.reset_index(drop=True)
    dataset = Dataset.from_pandas(cleaned, preserve_index=False)
    # ... 나머지 코드 동일
```

**training.yml 파라미터 추가** (training.yml:6-15):
```yaml
split:
  label_column: acct_code
  seed: 42
  test_size: 0.2
  val_size: 0.1
  max_classes: 280
  min_samples_per_class: 10  # 새로 추가 (각 클래스당 최소 10개 샘플)
  labelize_num_proc: 8
  extract_ratio: 1
  extract_seed: 42
  stratify_extract: true
```

**예상 효과**:
- 10개 미만 샘플 클래스 제거 → 학습 안정성 향상
- 224개 → 약 180-200개 클래스로 감소
- 제거되는 전체 샘플은 0.1% 미만 (거의 영향 없음)

---

### 방안 2: **Class Weighting** (COMPLEMENTARY)

Loss function에 클래스 가중치 적용하여 소수 클래스 강조

#### Trainer에 class_weight 추가

**training.yml에 추가** (training.yml:84 이후):
```yaml
  training_args:
    # ... 기존 설정 ...
    greater_is_better: true

  # 새로 추가
  compute_class_weights: true  # 자동 계산
  class_weight_strategy: "balanced"  # or "sqrt" for softer weighting
```

**train/nodes.py 수정 필요**:
```python
from sklearn.utils.class_weight import compute_class_weight

def prepare_for_trainer(tokenized_datasets, params):
    # ... 기존 코드 ...

    if params.get("compute_class_weights", False):
        labels = tokenized_datasets["train"]["labels"]
        unique_labels = np.unique(labels)

        if params.get("class_weight_strategy") == "balanced":
            class_weights = compute_class_weight(
                class_weight='balanced',
                classes=unique_labels,
                y=labels
            )
        elif params.get("class_weight_strategy") == "sqrt":
            # Softer weighting using sqrt
            label_counts = np.bincount(labels)
            class_weights = np.sqrt(np.max(label_counts) / (label_counts + 1))

        # Trainer에 전달 (Trainer 초기화 시 사용)
        trainer_config["class_weights"] = class_weights
```

**주의**: HuggingFace Trainer는 기본적으로 class_weight를 지원하지 않으므로, **Custom Trainer** 또는 **Weighted Loss**를 직접 구현해야 합니다.

---

### 방안 3: **Over-sampling / Under-sampling** (ADVANCED)

소수 클래스 오버샘플링 또는 다수 클래스 언더샘플링

#### Imbalanced-learn 사용 (추가 라이브러리 필요)

```python
from imblearn.over_sampling import RandomOverSampler
from imblearn.under_sampling import RandomUnderSampler
from imblearn.combine import SMOTEENN

# training.yml
split:
  resampling:
    enabled: true
    strategy: "over"  # "over", "under", or "smote"
    target_ratio: 0.3  # 소수 클래스를 다수 클래스의 30%까지 증강
```

**단점**:
- Over-sampling: 과적합 위험
- Under-sampling: 데이터 손실
- SMOTE: 텍스트 데이터에는 부적합

---

### 방안 4: **Focal Loss** (ADVANCED)

Hard example에 집중하는 loss function

```python
class FocalLoss(nn.Module):
    def __init__(self, gamma=2, alpha=None):
        super().__init__()
        self.gamma = gamma
        self.alpha = alpha

    def forward(self, inputs, targets):
        ce_loss = F.cross_entropy(inputs, targets, reduction='none')
        pt = torch.exp(-ce_loss)
        focal_loss = ((1 - pt) ** self.gamma) * ce_loss

        if self.alpha is not None:
            focal_loss = self.alpha[targets] * focal_loss

        return focal_loss.mean()
```

---

## 📊 권장 조치 순서

### 1단계: **즉시 적용** (가장 효과적)
- [x] 최소 샘플 수 필터링 (`min_samples_per_class: 10`)
- [x] 필터링 후 데이터 재생성 (`kedro run --pipeline=full_preprocess`)

### 2단계: **검증**
- [ ] 새로운 split으로 학습 실행
- [ ] 200 스텝에서 loss 확인 (소수점 진입 여부)
- [ ] 클래스별 정확도 확인

### 3단계: **필요시 추가 조치**
- [ ] Class weighting 적용 (1단계로 부족할 경우)
- [ ] Focal Loss 적용 (극심한 불균형이 남아있을 경우)

---

## 🔧 즉시 적용 가능한 코드 변경

### 파일 1: `split/nodes.py:62-121`

**변경 위치**: `create_dataset` 함수에 필터링 로직 추가

### 파일 2: `conf/base/parameters/training.yml:6-15`

**추가 파라미터**:
```yaml
min_samples_per_class: 10
```

---

## 📌 예상 결과

### 필터링 전
- 224개 클래스
- 불균형 비율: 44,098:1
- 10개 클래스가 1개 샘플만 보유

### 필터링 후 (min_samples_per_class: 10)
- 약 180-200개 클래스
- 불균형 비율: 약 4,400:1 (10배 개선)
- 모든 클래스가 최소 10개 샘플 보유
- **Stratified split 성공률 향상**

### 학습 성능 개선 예상
- Loss 수렴 속도: **2-3배 향상**
- 200 스텝 내 loss가 소수점 진입 (0.x 대)
- 소수 클래스 학습 안정성 향상

---

## 🔍 추가 분석 필요 사항

### 1. Validation/Test Split 확인
```python
# valid와 test에도 동일한 클래스 분포가 있는지 확인
python -c "
import pickle
data = pickle.load(open('data/05_model_input/serialized_datasets.pkl', 'rb'))
for split in ['valid', 'test']:
    labels = data[split]['labels']
    unique = set(labels)
    print(f'{split}: {len(unique)} unique classes')

    # Train에만 있는 클래스 확인
    train_labels = set(data['train']['labels'])
    only_in_split = unique - train_labels
    print(f'  Classes only in {split}: {len(only_in_split)}')
"
```

### 2. 클래스별 text 길이 분포
일부 클래스가 과도하게 짧은/긴 텍스트를 가질 경우 학습 문제 발생 가능

---

## 참고 자료

- HuggingFace Datasets Stratified Split: https://huggingface.co/docs/datasets/process#stratified-split
- Imbalanced Classification Guide: https://imbalanced-learn.org/
- Focal Loss Paper: https://arxiv.org/abs/1708.02002
