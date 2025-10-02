# Jupyter Notebook Data Loading Tutorial

주피터 노트북에서 `account-tax` 파이프라인의 데이터를 로드하고 탐색하는 방법을 안내합니다.

## 초기 설정

### 1. 기본 라이브러리 임포트

```python
# 데이터 처리
import pandas as pd
import numpy as np

# 데이터 로드
import pickle
import json
from pathlib import Path

# Kedro 통합
from kedro.framework.startup import bootstrap_project
from kedro.framework.session import KedroSession

# 시각화
import matplotlib.pyplot as plt
import seaborn as sns

# HuggingFace datasets
from datasets import Dataset, DatasetDict

# 화면 표시 설정
pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', 100)
pd.set_option('display.width', None)
```

## Method 1: Kedro Catalog을 통한 로드 (권장)

Kedro의 DataCatalog를 사용하면 모든 데이터셋 타입을 자동으로 처리합니다.

```python
# 프로젝트 부트스트랩
project_path = Path.cwd().parent  # notebooks/에서 실행 시
bootstrap_project(project_path)

# Kedro 세션 생성
with KedroSession.create(project_path=project_path) as session:
    context = session.load_context()
    catalog = context.catalog

    # 데이터셋 로드
    raw_data = catalog.load("raw_account_data")  # Parquet
    standardized_data = catalog.load("standardized_data")  # Parquet
    validated_data = catalog.load("validated_data")  # Parquet
    base_table = catalog.load("base_table")  # Parquet (features 포함)
```

### Catalog에서 사용 가능한 데이터셋

| 데이터셋 이름 | 타입 | 위치 | 설명 |
|--------------|------|----------|-------------|
| `raw_account_data` | Parquet | `data/01_raw/dataset.parquet` | 원본 입력 데이터 |
| `standardized_data` | Parquet | `data/02_intermediate/standardized_data.parquet` | Ingestion 후 |
| `validated_data` | Parquet | `data/03_primary/validated_data.parquet` | Preprocess 후 |
| `base_table` | Parquet | `data/04_feature/base_table.parquet` | Feature engineering 완료 |
| `serialized_datasets` | Pickle | `data/05_model_input/serialized_datasets.pkl` | Text + labels (HF Dataset) |
| `token_length_report` | JSON | `data/08_reporting/token_length_report.json` | 토크나이징 통계 |

## Method 2: 파일 직접 로드

### Parquet 파일 로드

```python
import pandas as pd

# Parquet 파일 로드
df = pd.read_parquet('data/01_raw/dataset.parquet')

# 특정 엔진 지정
df = pd.read_parquet(
    'data/04_feature/base_table.parquet',
    engine='pyarrow'
)

# 빠른 점검
print(f"Shape: {df.shape}")
print(f"Columns: {df.columns.tolist()}")
print(f"Memory: {df.memory_usage(deep=True).sum() / 1024**2:.2f} MB")
df.head()
```

### Pickle 파일 로드 (HuggingFace DatasetDict)

Pickle 파일은 종종 복잡한 중첩 구조를 포함합니다. 탐색 및 추출 방법:

```python
import pickle

# Pickle 파일 로드
with open('data/05_model_input/serialized_datasets.pkl', 'rb') as f:
    data = pickle.load(f)

# 1단계: 타입 확인
print(f"Type: {type(data)}")
print(f"Object: {data}")

# 2단계: 구조 탐색 (키 트리)
def explore_dict_structure(obj, indent=0):
    """딕셔너리 구조를 재귀적으로 탐색합니다."""
    prefix = "  " * indent

    if isinstance(obj, dict):
        for key, value in obj.items():
            print(f"{prefix}{key}: {type(value).__name__}")
            if isinstance(value, (dict, list)):
                explore_dict_structure(value, indent + 1)
    elif isinstance(obj, list) and len(obj) > 0:
        print(f"{prefix}[0]: {type(obj[0]).__name__}")
        if isinstance(obj[0], (dict, list)):
            explore_dict_structure(obj[0], indent + 1)

explore_dict_structure(data)
```

**serialized_datasets.pkl 예시 출력:**
```
<class 'datasets.dataset_dict.DatasetDict'>
DatasetDict({
    train: Dataset({
        features: ['text', 'labels'],
        num_rows: 2538448
    })
    valid: Dataset({
        features: ['text', 'labels'],
        num_rows: 317306
    })
    test: Dataset({
        features: ['text', 'labels'],
        num_rows: 317307
    })
})
```

### 3단계: HuggingFace Dataset에서 DataFrame 추출

```python
# 방법 A: 전체 split을 pandas로 변환
train_df = data['train'].to_pandas()
valid_df = data['valid'].to_pandas()
test_df = data['test'].to_pandas()

# 방법 B: 특정 컬럼 접근
texts = data['train']['text']
labels = data['train']['labels']

# 방법 C: 샘플 인덱스 선택
sample = data['train'].select(range(10))  # 처음 10개 샘플
sample_df = sample.to_pandas()

# 방법 D: 데이터 필터링
filtered = data['train'].filter(lambda x: x['labels'] == 5)
filtered_df = filtered.to_pandas()
```

### JSON 리포트 로드

```python
import json

# Token length report 로드
with open('data/08_reporting/token_length_report.json', 'r') as f:
    report = json.load(f)

# 예쁘게 출력
import pprint
pprint.pprint(report)

# 특정 통계 접근
print(f"Overall mean token length: {report['overall']['mean']}")
print(f"Train split stats: {report['per_split']['train']}")
```

## 일반적인 데이터 탐색 작업

### 1. 기본 통계

```python
# pandas DataFrame용
df.info()
df.describe()
df.dtypes
df.isnull().sum()

# HuggingFace Dataset용
dataset = data['train']
print(f"Num rows: {len(dataset)}")
print(f"Columns: {dataset.column_names}")
print(f"Features: {dataset.features}")
```

### 2. 라벨 분포

```python
# pandas DataFrame에서
label_counts = df['label_column'].value_counts()
print(label_counts)

# HuggingFace Dataset에서
from collections import Counter
labels = data['train']['labels']
label_dist = Counter(labels)
print(label_dist.most_common(10))

# 시각화
pd.Series(label_dist).plot(kind='bar', figsize=(12, 6))
plt.title('Label Distribution')
plt.xlabel('Label')
plt.ylabel('Count')
plt.tight_layout()
plt.show()
```

### 3. 무작위 샘플링

```python
# pandas DataFrame에서
sample_df = df.sample(n=10, random_state=42)

# HuggingFace Dataset에서
import random
random.seed(42)
indices = random.sample(range(len(data['train'])), k=10)
samples = data['train'].select(indices)

# 샘플 표시 (생략 없이 전체 출력)
for i, sample in enumerate(samples):
    print(f"\n{'='*80}")
    print(f"Sample {i+1}")
    print(f"{'='*80}")
    print(f"Text: {sample['text']}")  # 전체 텍스트 출력
    print(f"Label: {sample['labels']}")
```

## 완전한 워크플로우 예시

```python
# 1. Kedro를 통한 데이터 로드
from kedro.framework.startup import bootstrap_project
from kedro.framework.session import KedroSession
from pathlib import Path

project_path = Path.cwd().parent
bootstrap_project(project_path)

with KedroSession.create(project_path=project_path) as session:
    catalog = session.load_context().catalog
    base_table = catalog.load("base_table")

# 2. 기본 탐색
print(f"Shape: {base_table.shape}")
print(f"Columns: {base_table.columns.tolist()}")

# 3. Pickle 데이터셋 로드
import pickle
with open('../data/05_model_input/serialized_datasets.pkl', 'rb') as f:
    datasets = pickle.load(f)

# 4. 구조 탐색
def show_dataset_info(dataset_dict):
    for split_name, dataset in dataset_dict.items():
        print(f"\n{split_name}:")
        print(f"  Rows: {len(dataset)}")
        print(f"  Columns: {dataset.column_names}")
        print(f"  Sample: {dataset[0]}")

show_dataset_info(datasets)

# 5. pandas로 변환하여 분석
train_df = datasets['train'].to_pandas()

# 6. 시각화
import matplotlib.pyplot as plt
import seaborn as sns

fig, axes = plt.subplots(1, 2, figsize=(15, 5))

# 라벨 분포
train_df['labels'].value_counts().plot(kind='bar', ax=axes[0])
axes[0].set_title('Label Distribution')
axes[0].set_xlabel('Label')
axes[0].set_ylabel('Count')

# 텍스트 길이 분포
text_lengths = train_df['text'].str.len()
axes[1].hist(text_lengths, bins=50, edgecolor='black')
axes[1].set_title('Text Length Distribution')
axes[1].set_xlabel('Character Count')
axes[1].set_ylabel('Frequency')

plt.tight_layout()
plt.show()
```

## 팁과 모범 사례

1. **메모리 관리**: HuggingFace Datasets는 메모리 매핑을 사용하므로 대용량 파일도 효율적입니다. `.to_pandas()`는 작은 서브셋에만 사용하세요.

2. **지연 로딩**: Dataset은 지연 로딩됩니다. pandas로 변환하기 전에 `.select()` 또는 `.filter()`를 사용하여 메모리 사용량을 줄이세요.

3. **병렬 처리**: 대용량 parquet 파일 로드 시 `engine='pyarrow'`를 사용하면 성능이 향상됩니다.

4. **캐싱**: Kedro는 세션 동안 로드된 데이터셋을 캐싱하므로 반복된 `catalog.load()` 호출은 빠릅니다.

5. **Pickle 파일 탐색**: DataFrame을 추출하기 전에 항상 `explore_dict_structure()`를 사용하여 중첩된 데이터를 이해하세요.

## 문제 해결

**문제**: 데이터 로드 시 `FileNotFoundError`
- **해결**: 해당 파이프라인 스테이지를 먼저 실행했는지 확인 (`kedro run --pipeline=<stage>`)

**문제**: Pickle 파일에 예상치 못한 구조 포함
- **해결**: `explore_dict_structure()`를 사용하여 전체 키 트리 검사

**문제**: 대용량 Dataset을 pandas로 변환 시 `MemoryError`
- **해결**: `.select(range(n))`으로 먼저 샘플링하거나 청크 단위로 처리

## 추가 참고 자료

- [Kedro Data Catalog Documentation](https://docs.kedro.org/en/stable/data/data_catalog.html)
- [HuggingFace Datasets Documentation](https://huggingface.co/docs/datasets/)
- [Pandas User Guide](https://pandas.pydata.org/docs/user_guide/index.html)
