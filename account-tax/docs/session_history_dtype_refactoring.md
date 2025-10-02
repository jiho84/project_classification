# 세션 기록: 데이터 타입 변환 리팩토링

**날짜**: 2025-10-01
**작업**: 데이터 타입 변환 로직 개선 및 결측값 처리 구현

---

## 요구사항 요약

### 1. normalize_value에 transaction_code 매핑 추가
```python
"transaction_code": {
    "1": "sale",
    "2": "purchase",
}
```

### 2. transform_dtype 로직 개선
- **기존**: 고정된 숫자 컬럼 리스트 사용
- **개선**: 컬럼명 패턴으로 동적 판단
  - 'date' 포함 컬럼 → YYYY-MM-DD 문자열
  - 계산 패턴 컬럼 (year, amount, quantity, price, count) → 숫자 유지
  - 나머지 숫자 컬럼 → str 변환 (float → int → str)

### 3. normalize_missing_values 구현
- 숫자형 컬럼 중 'date' 미포함 → 결측값 0으로 대체
- 날짜 컬럼과 문자열 컬럼은 그대로 유지

### 4. cast_column_types 함수 삭제
- 더 이상 필요 없음 (transform_dtype으로 통합)

---

## 핵심 설계 원칙

### Q1: 날짜 처리 방식
**답변**: 현재 설계 유지하되, 'date' 포함 여부로 확대 적용
- 고정 리스트 사용하지 않음
- 컬럼명에 'date' 포함 여부로 동적 판단

### Q2: 숫자 컬럼 관리 방식
**답변**: 변수 고정 리스트 사용 금지
- 컬럼명 검사로 동적 판단
- 새로운 데이터셋에도 유연하게 대응

### Q3: normalize_missing_values 로직
**답변**: str 아닌 칼럼 중 date 미포함 → 0으로 대체
```python
if pd.api.types.is_numeric_dtype(data[col]):  # 숫자형
    if 'date' not in col.lower():  # date 미포함
        data[col] = data[col].fillna(0)
```

### Q4: year 컬럼 결측값
**답변**: 이전 단계에서 drop 처리됨 (걱정 안해도 됨)

### Q5: serialize_to_text 날짜 변환 로직
**답변**: 유지 필요 (방어적 코딩)
- 결측값 0 대체가 float → int 타입 변경을 의미하지 않음
- 소수점 텍스트화 방지 로직 여전히 필요

---

## 구현 내용

### 1. normalize_value에 transaction_code 매핑 추가 ✅
**파일**: `src/account_tax/pipelines/preprocess/nodes.py`
**위치**: 라인 99-102

```python
"transaction_code": {
    "1": "sale",
    "2": "purchase",
},
```

---

### 2. transform_dtype 패턴 기반 로직으로 수정 ✅
**파일**: `src/account_tax/pipelines/ingestion/nodes.py`
**위치**: 라인 130-199

#### 구현 로직
```python
def transform_dtype(data: pd.DataFrame) -> pd.DataFrame:
    """Transform data types for optimal processing and serialization.

    Strategy:
    1. Numeric columns with 'date' in name → YYYY-MM-DD string format
    2. Numeric columns with calculation patterns → Keep as numeric
    3. Other numeric columns (codes, IDs) → Convert to string
    4. Non-numeric columns → Convert to string
    """

    # 계산용 숫자 패턴
    numeric_keep_patterns = ['year', 'amount', 'quantity', 'price', 'count']

    for col in data.columns:
        if pd.api.types.is_numeric_dtype(data[col]):
            # 숫자형 컬럼
            if 'date' in col.lower():
                # 날짜 → YYYY-MM-DD 문자열
                data[col] = pd.to_datetime(data[col].astype(str), format='%Y%m%d', errors='coerce')
                data[col] = data[col].dt.strftime('%Y-%m-%d')
            elif any(pattern in col.lower() for pattern in numeric_keep_patterns):
                # 계산 컬럼 → 숫자 유지
                pass
            else:
                # 코드/ID 컬럼 → float → int → str
                if pd.api.types.is_float_dtype(data[col]):
                    data[col] = data[col].apply(
                        lambda x: str(int(x)) if pd.notna(x) and x == x else 'nan'
                    )
        else:
            # 비숫자형 → 문자열
            data[col] = data[col].astype(str)

    return data
```

#### 예시
- **Input**:
  - `document_type=13.0` (float32)
  - `supply_amount=1000.0` (float64)
  - `date=20250101` (int)

- **Output**:
  - `document_type="13"` (str)
  - `supply_amount=1000.0` (float64) - 유지
  - `date="2025-01-01"` (str)

---

### 3. normalize_missing_values 구현 ✅
**파일**: `src/account_tax/pipelines/preprocess/nodes.py`
**위치**: 라인 244-284

#### 구현 로직
```python
def normalize_missing_values(
    data: pd.DataFrame,
    placeholders: Dict[str, Any],
) -> pd.DataFrame:
    """Standardise missing values ahead of text serialisation.

    Strategy:
    - Numeric columns (excluding date columns) → Fill NaN with 0
    - Date columns and string columns → Keep as-is
    """
    data = data.copy()
    filled_count = 0

    for col in data.columns:
        # 숫자형 컬럼이면서 date 미포함
        if pd.api.types.is_numeric_dtype(data[col]):
            if 'date' not in col.lower():
                missing_count = data[col].isna().sum()
                if missing_count > 0:
                    data[col] = data[col].fillna(0)
                    logger.info(f"Filled {missing_count} missing values in {col} with 0")
                    filled_count += missing_count

    logger.info(f"Missing value normalization complete: {filled_count} total values filled with 0")
    return data
```

#### 예시
- **Input**:
  - `supply_amount=[1000.0, NaN, 500.0]`
  - `date=["2025-01-01", "NaT", "2025-01-02"]`

- **Output**:
  - `supply_amount=[1000.0, 0.0, 500.0]`
  - `date=["2025-01-01", "NaT", "2025-01-02"]` - 유지

---

### 4. cast_column_types 함수 삭제 ✅
**파일**: `src/account_tax/pipelines/preprocess/nodes.py`
**변경**: 함수 정의 완전히 제거 (라인 167-214)

- 파이프라인에서는 이미 제거되어 있었음
- 함수 정의도 삭제하여 완전히 제거

---

## 변경된 파일 목록

1. **`src/account_tax/pipelines/preprocess/nodes.py`**
   - `normalize_value()`: transaction_code 매핑 추가
   - `normalize_missing_values()`: 결측값 0 대체 로직 구현
   - `cast_column_types()`: 함수 삭제

2. **`src/account_tax/pipelines/ingestion/nodes.py`**
   - `transform_dtype()`: 패턴 기반 동적 타입 변환 로직으로 완전히 재작성

---

## 파이프라인 흐름

### Ingestion Pipeline
```
load_data
  ↓
standardize_columns
  ↓
transform_dtype  ← 패턴 기반 타입 변환
  ↓
extract_metadata
```

### Preprocess Pipeline
```
clean_data
  ↓
filter_data
  ↓
normalize_value  ← transaction_code 매핑 추가
  ↓
validate_data
  ↓
normalize_missing_values  ← 결측값 0 대체
```

---

## 핵심 개선 효과

### 1. 유연성 향상
- 고정 리스트 제거 → 컬럼명 패턴으로 동적 처리
- 새로운 데이터셋에 `discount_amount` 추가되어도 자동으로 숫자 유지
- 새로운 날짜 컬럼 `invoice_date` 추가되어도 자동으로 YYYY-MM-DD 변환

### 2. 매핑 성공
- **이전**: `document_type: "13.0"` (매핑 실패)
- **현재**: `document_type: "13"` (매핑 성공)

### 3. 소수점 제거
- 코드 컬럼에서 ".0" 완전히 제거
- 텍스트 직렬화 시 불필요한 토큰 생성 방지

### 4. 날짜 형식 통일
- 모든 date 컬럼 자동으로 "YYYY-MM-DD" 형식 변환
- 수동 관리 불필요

### 5. 결측값 처리
- 계산용 숫자 컬럼의 NaN 자동으로 0 대체
- 텍스트 직렬화 전 데이터 정제

---

## 기술적 세부사항

### 패턴 매칭 로직
```python
numeric_keep_patterns = ['year', 'amount', 'quantity', 'price', 'count']

# 다음 컬럼들은 자동으로 숫자 유지:
# - year
# - supply_amount, vat_amount, total_amount
# - quantity
# - unit_price
# - journal_key_count
```

### 날짜 변환 로직
```python
if 'date' in col.lower():
    # 다음 컬럼들은 자동으로 YYYY-MM-DD 변환:
    # - date
    # - close_date
    # - vat_change_date
    # - tax_invoice_apply_date
```

### float → int → str 변환
```python
# NaN 처리 포함
data[col] = data[col].apply(
    lambda x: str(int(x)) if pd.notna(x) and x == x else 'nan'
)

# 결과:
# 13.0 → "13"
# NaN → "nan"
```

---

## Ultra Think 분석 결과

### 발견된 문제
초기 로직에서 치명적 결함 발견:

```python
# 잘못된 로직 (초기안)
if pd.api.types.is_numeric_dtype(data[col]):
    if 'date' in col.lower():
        # 날짜
    else:
        # 숫자 유지 ← document_type도 여기로 감! (잘못됨)
```

**문제**: `document_type`, `acct_code` 같은 코드 컬럼이 float32이므로 숫자로 유지됨!

### 해결책
계산용 숫자 패턴으로 명시적 구분:

```python
# 올바른 로직 (최종안)
if pd.api.types.is_numeric_dtype(data[col]):
    if 'date' in col.lower():
        # 날짜 → YYYY-MM-DD
    elif any(pattern in col.lower() for pattern in numeric_keep_patterns):
        # year, supply_amount → 유지
    else:
        # document_type, acct_code → str 변환
```

---

## 테스트 포인트

### 1. transform_dtype 테스트
- [ ] `document_type` → "13" (str)
- [ ] `acct_code` → "71600" (str)
- [ ] `supply_amount` → 1000.0 (float)
- [ ] `date` → "2025-01-01" (str)
- [ ] `close_date` → "2025-12-31" (str)

### 2. normalize_value 테스트
- [ ] `transaction_code: "1"` → "sale"
- [ ] `transaction_code: "2"` → "purchase"
- [ ] `document_type: "13"` → "tax_invoice" (매핑 성공 확인)

### 3. normalize_missing_values 테스트
- [ ] `supply_amount` NaN → 0.0
- [ ] `vat_amount` NaN → 0.0
- [ ] date 컬럼 NaT → 그대로 유지

### 4. 파이프라인 통합 테스트
```bash
kedro run --pipeline=full_preprocess
```

---

## 참고 사항

### 이전 세션에서 이어진 작업
- token_length_report JSON 유니코드 처리
- serialize_to_text에 key_value_separator 추가
- max_classes 280으로 증가 (273개 레이블 수용)

### 남은 작업
- 없음 (모든 요구사항 완료)

### 주의사항
- serialize_to_text의 날짜 변환 로직 유지 (방어적 코딩)
- 결측값 0 대체가 타입을 변경하지 않음 (float 유지)
- 소수점 제거 로직 여전히 필요

---

---

## 최종 개선: datetime64 접근 방식 (간소화)

### 변경 이유
설계자 제안을 받아들여 날짜 처리를 더 간단하고 효율적으로 개선:
- **이전**: 날짜를 문자열로 변환 후 직렬화 시 다시 파싱
- **개선**: 날짜를 datetime64로 저장하여 연산과 직렬화 모두 간편하게

### 변경 내용

#### 1. transform_dtype 간소화 ✅
**파일**: `src/account_tax/pipelines/ingestion/nodes.py`
**변경**: 날짜 컬럼을 datetime64로 저장

```python
if 'date' in col.lower():
    # datetime64로 변환 (연산 편의성)
    data[col] = pd.to_datetime(data[col].astype(str), format='%Y%m%d', errors='coerce')
    logger.info(f"Converted {col} to datetime64")
```

**장점**:
- 날짜 연산 바로 가능 (휴일 판단, 날짜 차이 계산 등)
- 중간 변환 단계 제거
- pandas의 최적화된 datetime 연산 활용

#### 2. serialize_to_text 간소화 ✅
**파일**: `src/account_tax/pipelines/split/nodes.py`
**변경**: datetime64 타입 감지 후 자동 변환

```python
for col in df.columns:
    if pd.api.types.is_datetime64_any_dtype(df[col]):
        # datetime64 → "YYYY-MM-DD" (NaT는 빈 문자열)
        df[col] = df[col].dt.strftime('%Y-%m-%d').fillna('')
```

**장점**:
- 하드코딩된 컬럼명 리스트 불필요
- 타입 기반 자동 감지
- 새로운 날짜 컬럼 자동 대응

### 데이터 흐름 비교

#### 이전 방식
```
date=20250101 (int)
  ↓ transform_dtype
date="2025-01-01" (str)
  ↓ serialize_to_text
date="2025-01-01" (str) - 그대로 사용
```

#### 개선된 방식
```
date=20250101 (int)
  ↓ transform_dtype
date=Timestamp('2025-01-01') (datetime64)
  ↓ [날짜 연산 가능]
  ↓ serialize_to_text
date="2025-01-01" (str) - dt.strftime() 한 번만
```

### 메모리 비교
- **datetime64**: 8 bytes (고정)
- **string**: 가변 (평균 10-12 bytes)
- **차이**: 거의 없음 (실용적으로 무시 가능)

### 연산 편의성
```python
# 날짜 연산 예시 (feature 파이프라인)
df['is_weekend'] = df['date'].dt.dayofweek >= 5
df['month'] = df['date'].dt.month
df['days_since'] = (pd.Timestamp.now() - df['date']).dt.days
```

---

---

## 최종 단순화: Int64 (Nullable Integer) 접근 방식

### 변경 이유
".0" 소수점 문제 발견 → **Nullable Int64**로 근본 해결

**문제 발견**:
```
token_length_report.json에서:
vat_amount:61527.0  ❌
total_amount:676800.0  ❌
counterparty_card_number:0.0  ❌
```

**근본 원인**:
- float64 → string 변환 시 ".0" 포함
- normalize_missing_values에서 `fillna(0)` 해도 여전히 float

**해결책**: Int64 (Nullable Integer) 사용
- NaN 유지 가능
- 정수 타입 유지
- ".0" 문제 완전 해결

---

### 최종 구현

#### 1. transform_dtype (ingestion/nodes.py) - 완전 재작성 ✅

**전략**:
- **date 패턴**: 다양한 형식 → 숫자만 추출 → Int64 (YYYYMMDD)
- **amount 패턴**: round(0) → Int64
- **나머지**: 전부 string

```python
DATE_TOKEN = "date"
AMOUNT_TOKEN = "amount"

def transform_dtype(data: pd.DataFrame) -> pd.DataFrame:
    """Standardise column dtypes once during ingestion."""
    data = data.copy()

    date_cols = [col for col in data.columns if DATE_TOKEN in col.lower()]
    amount_cols = [col for col in data.columns if AMOUNT_TOKEN in col.lower()]

    # 1. 날짜 컬럼 → Int64 (YYYYMMDD)
    for col in date_cols:
        s = data[col].astype("string")
        cleaned = s.str.replace(r"\D", "", regex=True).replace("", pd.NA)

        # 8자리 검증
        bad_mask = cleaned.str.len().notna() & (cleaned.str.len() != 8)
        if bad_mask.any():
            logger.warning(f"{col}: invalid date tokens")
            cleaned = cleaned.mask(bad_mask)

        data[col] = pd.to_numeric(cleaned, errors="coerce").astype("Int64")

    # 2. 금액 컬럼 → Int64 (round)
    for col in amount_cols:
        numeric = pd.to_numeric(data[col], errors="coerce")
        data[col] = numeric.round(0).astype("Int64")

    # 3. 나머지 → string
    protected = set(date_cols) | set(amount_cols)
    for col in data.columns:
        if col not in protected:
            data[col] = data[col].astype("string")

    return data
```

**핵심 개선**:
- ✅ year, quantity, price, count → **string 변환** (계산 불필요)
- ✅ 날짜 다양한 형식 → **YYYYMMDD 통일**
- ✅ 금액 스케일링 소수점 → **round(0)**
- ✅ 패턴 기반 자동 감지

---

#### 2. serialize_to_text (split/nodes.py) - 패턴 기반 ✅

**전략**:
- date 패턴: Int64 → "YYYY-MM-DD"
- amount 패턴: Int64 → string (no ".0")
- 하드코딩 제거, 자동 감지

```python
def serialize_function(examples):
    df = pd.DataFrame({col: examples[col] for col in text_columns})

    # 패턴 기반 자동 감지
    date_cols = [col for col in df.columns if 'date' in col.lower()]
    amount_cols = [col for col in df.columns if 'amount' in col.lower()]

    # 1. 날짜: Int64 → "YYYY-MM-DD"
    for col in date_cols:
        df[col] = (
            pd.to_datetime(df[col].astype("Int64"), format="%Y%m%d", errors="coerce")
              .dt.strftime("%Y-%m-%d")
              .fillna("")
        )

    # 2. 금액: Int64 → string
    for col in amount_cols:
        df[col] = df[col].astype("Int64").astype("string").fillna("")

    # 3. 텍스트 빌드
    if include_column_names:
        texts = df.apply(
            lambda row: separator.join(
                f"{col}{key_value_separator}{row[col]}" for col in text_columns
            ),
            axis=1,
        ).tolist()
    else:
        texts = df.astype("string").agg(separator.join, axis=1).tolist()

    return {"text": texts}
```

**핵심 개선**:
- ✅ 하드코딩 컬럼 리스트 제거
- ✅ 패턴 기반 자동 감지
- ✅ 새 컬럼 자동 대응

---

#### 3. normalize_missing_values (preprocess/nodes.py) - Pass-through ✅

**변경**: Int64가 결측값 처리 → 함수는 pass-through

```python
def normalize_missing_values(data: pd.DataFrame, placeholders: Dict[str, Any]) -> pd.DataFrame:
    """Reserved for future use - currently pass-through."""
    return data
```

**이유**:
- Int64가 `<NA>` 자동 처리
- fillna 불필요
- 미래 확장 위해 함수 유지

---

### 최종 데이터 흐름

```
Raw Data (Ingestion)
  ↓
transform_dtype:
  - date → Int64(20250101)
  - amount → Int64(1000)
  - document_type → "13" (string)
  ↓
Preprocess (normalize_missing_values: pass-through)
  ↓
Split (serialize_to_text):
  - date → "2025-01-01" (string)
  - amount → "1000" (string, no ".0")
  - document_type → "13" (string)
  ↓
Text: "date:2025-01-01| vat_amount:1000| document_type:13"
```

---

### 핵심 개선 효과

1. **".0" 완전 제거**: Int64 → string 변환 시 소수점 없음
2. **2포인트 변환**: ingestion + serialize (명확한 책임 분리)
3. **패턴 기반 자동화**: 새 컬럼 자동 대응
4. **결측값 처리 단순화**: Int64가 `<NA>` 자동 처리
5. **타입 안전성**: Nullable Int로 안전한 연산

---

## 마무리

**작업 완료 시각**: 2025-10-01
**상태**: ✅ Int64 접근 방식으로 완전 단순화 완료
**다음 실행**: `kedro run --pipeline=full_preprocess`로 전체 파이프라인 테스트

### 최종 구조 요약
1. **Ingestion**: transform_dtype에서 **Int64 변환** (date, amount)
2. **Preprocess**: normalize_missing_values **pass-through**
3. **Split**: serialize_to_text에서 **패턴 기반 문자열 변환**

### 변경된 파일
- ✅ `src/account_tax/pipelines/ingestion/nodes.py` (transform_dtype 완전 재작성)
- ✅ `src/account_tax/pipelines/split/nodes.py` (serialize_to_text 패턴 기반)
- ✅ `src/account_tax/pipelines/preprocess/nodes.py` (normalize_missing_values pass-through)

---

## 빠른 참조

### 숫자로 유지되는 컬럼 패턴
- `year`
- `*amount*` (supply_amount, vat_amount, total_amount 등)
- `*quantity*`
- `*price*`
- `*count*`

### 문자열로 변환되는 컬럼 예시
- `document_type`, `party_type_code`, `transaction_code`
- `acct_code`, `party_code`, `desc_id`
- `company_vat_id`, `party_vat_id`
- 기타 모든 코드/ID 컬럼

### YYYY-MM-DD로 변환되는 컬럼
- 컬럼명에 'date' 포함된 모든 숫자형 컬럼
- `date`, `close_date`, `vat_change_date`, `tax_invoice_apply_date`
