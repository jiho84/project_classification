# Review Log

- Document code reviews, retrospectives, and quality assessments.
- Include date, scope, main findings, and follow-up actions.
- Keep entries concise; archive outdated notes instead of duplicating.

## 2025-09-24 · Function Review (Pipelines)

| 함수 | 블록화 | 심플성 | 연결성 | 기능성 | 메모 |
| --- | --- | --- | --- | --- | --- |
| `load_data` | ✅ | ✅ | ✅ | ✅ | MemoryDataset 계약 유지. 빈 데이터 검사 외 복잡 로직 없음. |
| `standardize_columns` | ✅ | ☑️ | ✅ | ✅ | 매핑 dict가 길지만 요구사항 충실. 추가 최적화 필요 없음. |
| `extract_metadata` | ✅ | ✅ | ✅ | ✅ | 스키마/통계 수집에 집중. 외부 의존성 없음. |
| `clean_data` | ✅ | ✅ | ✅ | ✅ | 중복 제거·결측 처리에 집중. Pandas API 활용. |
| `filter_data` | ✅ | ✅ | ✅ | ✅ | 단순 drop; 파라미터 기반 동작 명확. |
| `normalize_value` | ✅ | ☑️ | ✅ | ✅ | 기본 매핑 dict가 길지만 추상화 수준 적절. |
| `validate_data` | ✅ | ✅ | ✅ | ✅ | 금액/날짜 체크 후 DataFrame 반환. |
| `add_holiday_features` | ✅ | ✅ | ✅ | ✅ | `holidays` 라이브러리 활용. 오류 처리 간결. |
| `build_features` | ✅ | ✅ | ✅ | ✅ | 상위 노드를 래핑해 로깅 추가. |
| `select_features` | ✅ | ✅ | ✅ | ✅ | 순서 유지, 누락 컬럼 경고. |
| `prepare_dataset_inputs` | ✅ | ✅ | ✅ | ✅ | 중복/결측 제거만 수행하여 Split 단계가 책임을 이어받도록 정리. |
| `_initialize_label_slots` | ✅ | ✅ | ✅ | ✅ | 더미 슬롯 초기화 전용. |
| `_upsert_labels_into_slots` | ✅ | ✅ | ✅ | ✅ | 슬롯 부족 시 예외 발생시켜 안정성 확보. |
| `make_label2id` / `make_id2label` | ✅ | ✅ | ✅ | ✅ | 간단한 매핑; ClassLabel과 연동. |
| `create_dataset` | ✅ | ✅ | ✅ | ✅ | HF Dataset 변환과 라벨 슬롯 생성 책임 통합. |
| `to_hf_and_split` | ✅ | ✅ | ✅ | ✅ | Stratify 실패 시 fallback 제공; 파라미터 연결 명확. |
| `labelize_and_cast` | ✅ | ✅ | ✅ | ✅ | `num_proc` 지원해 병렬 처리 가능. ClassLabel 캐스팅 처리. |
| `serialize_for_nlp` | ✅ | ☑️ | ✅ | ✅ | batched map + 열 축소. 텍스트 구성 로직이 길지만 목적상 필요. |
| `tokenize_datasets` | ✅ | ☑️ | ✅ | ✅ | HuggingFace 토크나이저 로딩 및 맵핑. 모델별 pad 처리 포함. |
| `prepare_for_trainer` | ✅ | ⚠️ | ✅ | ☑️ | Trainer 준비(토크나이저 재로드, collator, id2label 추출). 기능 충실하지만 코드가 길고 향후 분할 고려 필요. |

**요약**  
- 대부분의 함수가 블록화·심플성·연결성·기능성 기준을 충족.  
- 개선 포인트: `serialize_for_nlp`(텍스트 포맷 로직 분리 검토), `prepare_for_trainer`(서브 함수 분할로 가독성 향상).  
- 후속 조치: `tracking.run.nested=false` 적용 확인, Trainer 준비 로직 분해 검토, 텍스트 포맷 템플릿화 검토.
