# Notebook Usage Guide

Follow this checklist whenever you want to explore pipeline outputs inside Jupyter.

## 1. Launch Jupyter from the Kedro project

```bash
cd /home/user/projects/kedro_project/account-tax
kedro jupyter notebook  # or: jupyter lab
```

> Always start Jupyter inside the project root so the Kedro settings and catalog are discovered automatically.

Open `notebooks/explore_project.ipynb` (or create a new notebook in the same folder).

## 2. Import Kedro helpers in a notebook cell

```python
from load_intermediate_outputs import (
    get_catalog,
    load_prepared_datasets,
    load_text_datasets,
    summarise_dataframe,
)
```

These helpers live in `notebooks/load_intermediate_outputs.py` and wrap the Kedro session/ catalog boilerplate for you.

## 3. Materialise split outputs for analysis

```python
prepared = load_prepared_datasets()  # returns {"train": DataFrame, "valid": ..., "test": ...}
textual = load_text_datasets()

# 어떤 split이 준비됐는지 확인
list(prepared.keys())      # ['train', 'valid', 'test']
prepared.keys()            # dict_keys([...]) 그대로 보고 싶다면
```

Each value is a pandas `DataFrame`. If you need the raw lazy loaders instead (for custom IO), pass `materialize=False` to either function.

## 4. Quick sanity checks

```python
summarise_dataframe(prepared["train"])
prepared["train"].head()
prepared["train"].describe(include="all")
prepared["train"].sample(5, random_state=42)

# split별 크기 비교
{split: df.shape for split, df in prepared.items()}

# 특정 라벨 분포 확인
prepared["train"].value_counts("labels").head()
```

`prepared` keeps all original numerical/categorical columns plus `labels` and `split`. `textual` narrows the columns to `acct_code`, `labels`, `text`, `split` so you can feed it directly to tokenisers.

### Mini tutorial: 탐색 패턴 모음

1. **칼럼 목록 살펴보기**
   ```python
   prepared["train"].columns.tolist()
   textual["train"].columns
   ```
2. **결측치 요약**
   ```python
   prepared["train"].isna().mean().sort_values(ascending=False).head(10)
   ```
3. **숫자형 분포 시각화(간단)**
   ```python
   prepared["train"].hist(column=["supply_amount", "vat_amount"], bins=30, figsize=(10, 4))
   ```
4. **텍스트 샘플 미리보기**
   ```python
   textual["train"].loc[::5000, ["labels", "text"]].head()
   ```
5. **카테고리별 그룹 통계**
   ```python
   prepared["train"].groupby("labels")["total_amount"].agg(["count", "mean"]).sort_values("count", ascending=False).head(10)
   ```
6. **딕셔너리 전체 구조 탐색**
   ```python
   def print_dict_structure(mapping, indent=0):
       prefix = " " * indent
       if isinstance(mapping, dict):
           for key, value in mapping.items():
               print(f"{prefix}- {key}: {type(value).__name__}")
               print_dict_structure(value, indent + 2)
       else:
           # 비딕셔너리 최하위 값에 도달하면 종료
           return

   print_dict_structure(prepared)
   ```
   DataFrame과 같이 딕셔너리가 아닌 값에 도달하면 거기서 멈추므로, 중첩된 구조를 한 번에 훑어볼 때 유용합니다. 필요에 따라 `textual`이나 다른 catalog 결과에도 동일하게 적용할 수 있습니다.

필요한 탐색을 마친 후에는 중요한 통찰을 `docs/analysis.md`에 간략히 정리하고, 참고용으로 관련 노트북 셀 번호나 마크다운 링크를 남겨 두세요.

## 5. Load additional datasets on demand

If you need another catalog entry, open a temporary catalog via `get_catalog()` and call `catalog.load("dataset_name")`. Remember to close the session when you are done:

```python
catalog = get_catalog()
try:
    base = catalog.load("base_table")
finally:
    catalog._session.close()
```

## 6. Keep analyses in sync with docs

When you discover insights, record the highlights in `docs/analysis.md` and link back to the notebook cell or file for future reference.
