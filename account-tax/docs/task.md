# Task Journal

> 모든 작업은 질문 → 계획 → 실행 → 사후관리 순으로 기록합니다.
> 대분류(##), 중분류(###), 소분류(- [ ] 체크박스)를 사용해 현재 위치를 명확히 파악하세요.

## 예시 구조

## 대: 파이프라인 개선
### 중: Split 단계 최적화
- [ ] 소: `labelize_and_cast` 병렬도 검증
- [ ] 소: 병렬 처리 결과를 `docs/history.md`에 기록

### 중: 문서 체계 업데이트
- [ ] 소: `docs/architecture.md` 최신화
- [ ] 소: 관련 회고를 `docs/review.md`에 작성

## 대: 데이터 파이프라인 분석 환경 구축
### 중: 카탈로그 구성 개편
- [x] 소: `prepared_datasets_mlflow`를 `kedro_mlflow.io.artifacts.MlflowArtifactDataset` + `partitioned.ParquetDataset` 조합으로 정의
- [x] 소: `text_datasets_mlflow`를 동일한 방식으로 정의하고 기존 직렬화 아티팩트 항목 정리

### 중: 노트북 헬퍼 작성
- [x] 소: `notebooks/load_intermediate_outputs.ipynb`(또는 .py)에서 Kedro Session을 열어 `prepared_datasets_mlflow`, `text_datasets_mlflow`를 로드하는 샘플 코드 추가
- [x] 소: 노트북에서 pandas 분석 예시(행 수/기본 통계) 포함해 사용법 기록

### 중: 검증 및 문서화
- [x] 소: `kedro run --pipeline split` 실행 후 `data/05_model_input/`와 `data/06_models/` Parquet 파일 및 MLflow 아티팩트 생성 확인
- [x] 소: 검증 결과를 `docs/history.md`에 기록하고, 필요 시 후속 개선 과제를 체크박스로 업데이트

### 중: 데이터 품질 및 토큰 분석 노드 추가
- [x] 소: 결측치 정규화 노드 설계(대표 값, 적용 컬럼 정의)
- [x] 소: 결측치 정규화 노드 구현 및 파이프라인 배치 (Split 이전)
- [x] 소: 토큰 길이/샘플 추출 노드 설계 (분석 대상, 출력 포맷 정의)
- [x] 소: 토큰 길이/샘플 추출 노드 구현 및 파이프라인 배치 (직렬화 이후)
- [x] 소: MLflow에 토큰 통계/샘플 로그 검증 (artifact 및 metric 생성 확인)

## 대: 대기 중 작업
### 중: 필요한 경우 작성하세요.
- [ ] 소: …
