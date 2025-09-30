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

## 대: 파이프라인 구조 개편 (2025-09-29)
> 목표: Data Pipeline(공용)과 Train Pipeline(실험)을 분리하여 재사용성과 실험 유연성 향상

### 중: 설계 분석 및 계획 수립
- [x] 소: 현재 파이프라인 아키텍처 및 파라미터 구조 분석
- [x] 소: 구조 개편 종합 실행 계획 수립
- [x] 소: 작업 우선순위 및 의존관계 정의

### 중: ML 패키지 설치
- [x] 소: PyTorch 및 CUDA 지원 설치
- [x] 소: HuggingFace Transformers, PEFT, Datasets 설치
- [x] 소: DeepSpeed 및 Accelerate 설치
- [x] 소: requirements.txt 업데이트

### 중: 파라미터 구조 개편
- [x] 소: `conf/base/parameters/training.yml` 생성 및 구조 정의
- [x] 소: tokenization, model, optimization, trainer 파라미터 설정
- [x] 소: evaluation 및 diagnostics 파라미터 추가

### 중: 파이프라인 레지스트리 개편
- [x] 소: `pipeline_registry.py`에서 data_pipeline (ingestion+preprocess+feature) 정의
- [x] 소: training_pipeline (split+train) 정의
- [x] 소: e2e (end-to-end) 파이프라인 추가

### 중: 학습 노드 구현
- [x] 소: `tokenize_datasets` 노드 구현 - 텍스트 토큰화
- [x] 소: `load_model` 노드 구현 - Qwen 모델 초기화
- [x] 소: `apply_optimization` 노드 구현 - LoRA 및 최적화 적용
- [x] 소: `prepare_trainer` 노드 구현 - Trainer 설정 및 DeepSpeed 통합
- [x] 소: `train_model` 노드 구현 - 모델 학습 실행
- [x] 소: `evaluate_model` 노드 구현 - 모델 평가 메트릭 계산

### 중: 파이프라인 연결 문제 해결
- [ ] 소: split 파이프라인 출력 (`text_datasets_with_stats`) 확인
- [ ] 소: train 파이프라인 입력 (`text_datasets`) 매핑 수정
- [ ] 소: catalog.yml에서 데이터셋 연결 구성
- [ ] 소: 파이프라인 간 데이터 흐름 검증

### 중: 통합 테스트 및 검증
- [x] 소: Data Pipeline 독립 실행 테스트 (`kedro run --pipeline=data`)
- [x] 소: base_table 출력 검증 및 데이터 품질 확인 (3,302,918 records, 26 features)
- [ ] 소: Split Pipeline 독립 실행 테스트 (`kedro run --pipeline=split`)
- [ ] 소: Train Pipeline 실행 테스트 (`kedro run --pipeline=train`)
- [ ] 소: 모델 학습 및 저장 검증
- [ ] 소: MLflow 실험 추적 및 아티팩트 저장 확인

### 중: 문서화 및 마무리
- [ ] 소: `architecture.md` 업데이트 - 새로운 파이프라인 구조 반영
- [ ] 소: `history.md` 업데이트 - 구조 개편 과정 및 결과 기록
- [ ] 소: `review.md` 작성 - 개선사항 및 향후 과제 정리

## 대: 현재 이슈 및 액션 플랜 (2025-09-29 오후)

### 문제 진단:
1. **파이프라인 연결 단절**: Split 파이프라인이 `text_datasets_with_stats`를 출력하지만, Train 파이프라인은 `text_datasets`를 입력으로 기대
2. **데이터 흐름 불일치**:
   - Split 최종 출력: `text_datasets_with_stats` → `text_datasets_mlflow`
   - Train 첫 입력: `text_datasets` (존재하지 않음)
3. **노드 실행 오류**: `to_hf_and_split` 노드를 찾을 수 없다는 오류 (실제로는 존재함)

### 즉시 필요한 수정사항:
1. **옵션 A - 출력명 변경**: Split pipeline의 `analyze_token_lengths` 노드가 `text_datasets` 대신 `text_datasets_with_stats`로 출력 변경
2. **옵션 B - 카탈로그 별칭**: catalog.yml에서 `text_datasets`를 `text_datasets_with_stats`의 별칭으로 설정
3. **옵션 C - 파이프라인 수정**: analyze_token_lengths를 선택적으로 만들고 기본 출력을 `text_datasets`로 유지

### 권장 해결책:
**옵션 B 채택** - 카탈로그에서 데이터셋 매핑 추가:
```yaml
text_datasets:
  type: MemoryDataset

text_datasets_with_stats:
  type: MemoryDataset

# Training pipeline이 text_datasets를 참조할 수 있도록 별칭 설정
```

### 검증 계획:
1. Split 파이프라인 독립 실행 (`kedro run --pipeline=split`)
2. Training 파이프라인 독립 실행 (`kedro run --pipeline=training`)
3. End-to-end 실행 (`kedro run --pipeline=e2e`)

## 대: 향후 개선 과제 (백로그)
### 중: 추론 파이프라인 구축
- [ ] 소: Inference Pipeline 설계 (모델 로딩 → 토큰화 → 추론 → 후처리)
- [ ] 소: 배치 추론 및 실시간 추론 모드 구현
- [ ] 소: 추론 성능 최적화 및 벤치마킹

### 중: 평가 파이프라인 완성
- [ ] 소: 누락된 evaluation 노드 구현 완료
- [ ] 소: 비즈니스 메트릭 및 세무 영향 분석 통합
- [ ] 소: 자동화된 평가 리포트 생성

### 중: 모델 버전 관리 시스템
- [ ] 소: MLflow Model Registry 통합
- [ ] 소: 모델 버전별 성능 추적 및 비교
- [ ] 소: A/B 테스팅 프레임워크 구축
