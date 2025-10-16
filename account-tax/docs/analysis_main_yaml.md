# main_yaml.py 구조 분석

- **작성일**: 2025-10-15
- **범위**: `src/train/main_yaml.py`의 함수별 책임과 실행 흐름을 노드 스타일에 맞춰 문서화. 코드 수정 없음.

## 함수 책임 (선언 순서)
- `LOGGER = logging.getLogger(__name__)`  
  - 스크립트 전역 로거. 모든 함수가 같은 로깅 채널을 사용하도록 통합.
- `is_rank_zero() -> bool`  
  - 환경 변수 `LOCAL_RANK`가 `"0"`인지 검사해 주 프로세스 여부 판단. 로그 출력 및 저장 작업을 랭크 0으로 제한하는 데 사용.
- `parse_args() -> argparse.Namespace`  
  - DeepSpeed 런처에서 전달하는 `--config_yml`, `--local_rank`, `--rank`, `--world_size` 인자를 파싱.
- `build_training_arguments(args_cfg: Dict[str, Any], deepspeed_cfg: Dict[str, Any] | None) -> TrainingArguments`  
  - 기본 `training_args`를 DeepSpeed 설정과 병합하고, 필요한 경우 배치 크기·그래디언트 누적 값을 덮어씌움. `report_to` 문자열을 리스트로 변환하고 출력 디렉터리를 생성.
- `infer_num_labels(dataset) -> int`  
  - `Dataset`의 `features["labels"]`에서 클래스 수를 추론하거나, 없으면 라벨 고유값 개수로 계산.
- `maybe_apply_lora(model, lora_section: Dict[str, Any]) -> torch.nn.Module`  
  - `lora.enable`이 참일 경우 `LoraConfig`를 구성하고 모델을 LoRA로 감쌈. PEFT 미설치 시 예외 발생.
- `save_metrics(metrics: Dict[str, Any], path: str | None) -> None`  
  - 경로가 주어지면 디렉터리를 만들고 JSON으로 평가 지표를 저장.
- `build_class_weight_tensor(labels, num_labels, alpha, weight_min, weight_max)` *(utils/common.py)*  
  - 훈련 라벨 분포에서 평균 1.0 가중치 텐서를 생성하는 유틸 함수. 역수→지수→클립→정규화 순으로 처리하며, 비활성화 시에는 균등 가중치 텐서를 반환.
- `main() -> None`  
  - 스크립트의 진입점. 아래 세부 흐름과 내부 헬퍼(클로저·내부 클래스)를 모두 포함.

## main() 내부 흐름
1. **설정 로딩 및 시드 고정**  
   - `parse_args()`로 경로 인수를 받아 `yaml.safe_load()`로 YAML 구성 로드. `seed`, `model`, `data`, `training_args`, `deepspeed`, `lora`, `metrics`, `resume`, `loss` 섹션을 지역 변수에 분리. `set_seed()` 호출.
2. **데이터셋 및 토크나이저 준비**  
   - `datasets.load_from_disk`로 토크나이즈된 `DatasetDict` 읽기. 학습·평가 스플릿 선택. 토크나이저 생성 후 패딩 토큰 보정. 라벨 수는 `model.num_labels` 우선, 없으면 `infer_num_labels()` 활용.
3. **모델 생성 및 LoRA 적용**  
   - `AutoModelForSequenceClassification.from_pretrained()`로 기본 모델 로드. 필요 시 `gradient_checkpointing_enable()`, `maybe_apply_lora()` 적용.
4. **훈련 인자 및 데이터 콜레이터 구성**  
   - `build_training_arguments()` 호출, DeepSpeed 배치 값 동기화. `DataCollatorWithPadding`을 dtype에 따라 초기화.
5. **평가 지표 클로저 정의**  
   - `compute_metrics_fn`: argmax 기반 정확도, `f1_weighted`, `f1_macro`를 반환 (`zero_division=0`).
6. **손실 가중치 계산**  
   - `build_class_weight_tensor()`를 항상 호출해 가중치 텐서를 생성. `use_class_weights`가 꺼져 있으면 균등 가중치 텐서를 반환.
7. **콜백 및 Trainer 구성**  
   - 내부 클래스 `GPUMemoryCallback` 정의: 랭크 0과 CUDA 사용 시 GPU 메모리 로그를 기록.  
   - 내부 클래스 `WeightedTrainer` 정의: `Trainer` 상속 후 `compute_loss()`에서 클래스 가중치 적용.  
   - Trainer 생성 시 콜백 목록과 `class_weights_tensor` 전달. MLflow 콜백이 존재하면 `on_train_end`를 덮어써 서브프로세스 행을 방지.
8. **훈련/재시작/평가 실행**  
   - `resume` 설정을 읽어 체크포인트 경로 결정. `trainer.train()` 실행 후 성공 여부 플래그. 평가용 스플릿이 있으면 `trainer.evaluate()`.
9. **모델·토크나이저 저장 및 메트릭 기록**  
   - DeepSpeed 실행 여부에 따라 분기:  
     - DeepSpeed 경로: 모든 랭크가 `save_pretrained()`와 `tokenizer.save_pretrained()` 호출, 전후 배리어 동기화, 랭크 0이 메트릭 저장.  
     - 일반 경로: 월드 프로세스 0이 PEFT 어댑터와 토크나이저 저장, 메트릭 기록.
10. **분산 자원 정리 및 종료 로그**  
    - DeepSpeed 사용 시 프로세스 그룹 파괴 시도. 랭크 0에서 정상 종료 로그 출력.

## 유지보수 관점 이슈
- **비대한 `main()`**: 설정 파싱부터 자원 정리까지 단일 함수에 집중되어 역할 분리가 어렵고 테스트 단위가 지나치게 큼.
- **내부 정의된 클래스/클로저**: `GPUMemoryCallback`, `WeightedTrainer`, `compute_metrics_fn`이 `main()` 내부에 있어 재사용이 제한되고, 별도 테스트 작성이 어려움.
- **손실 가중치 블록의 복잡도**: 통계 보정·클램프·리포팅 루프가 길게 이어져 핵심 학습 흐름 가독성을 저해.
- **암묵적 설정 스키마**: YAML 키 요구사항이 명시적이지 않아 설정 생성 측과의 결합이 높음.
- **미사용 헬퍼 존재**: `_ensure_dir`, `_ensure_dirname`, `compute_accuracy` 등 과거 코드 잔재가 남아 있어 책임 경계를 흐림.
- **교차 관심사 혼재**: 로깅, MLflow 패치, DeepSpeed 동기화, 체크포인트 관리를 한 블록에서 처리하여 추후 확장이 어렵다.

- **현재 구현 흐름**
  1. `build_class_weight_tensor()`가 라벨 분포를 직접 받아 0 카운트를 1.0으로 치환.
  2. `use_class_weights`가 켜져 있으면 역수→지수(alpha)→캡(min/max)→평균 1 정규화 순으로 처리하고, 꺼져 있으면 균등 가중치를 반환.
- **단순화 효과**
  - 반복 루프, dummy 값 조정, 통계 리포트를 모두 제거해 블록 길이를 약 70줄에서 25줄 수준으로 축소.
  - 역수→알파→클립→정규화 순서가 코드에 그대로 드러나 직관적이고 테스트하기 쉬움.
  - 설정 파라미터가 `alpha`, `min`, `max` 세 축으로 단순화돼 후속 조정이 간편.
- **추가 확인 사항**
  - 평균 1 정규화 이후에도 캡 조건이 그대로 유지되는지 모니터링(필요 시 후속 보정 고려).
  - 희귀 클래스가 다수일 때 가중치 분포를 로그로 추적할지 여부 결정.

## 향후 논의 거리
- Kedro 노드 스타일에 맞춰 `load_artifacts`, `prepare_model`, `build_trainer`, `run_training`, `persist_outputs` 등 단계별 함수로 분리할지 여부.
- 손실 가중치 계산을 별도 유틸/모듈로 이동해 독립 테스트와 재사용성을 확보할지 결정.
- 설정 검증(스키마/데이터클래스) 레이어를 추가해 YAML 생성과 스크립트 간 계약을 명확히 할 필요성.
- 내부 콜백·Trainer 서브클래스를 외부 팩토리로 추출하여 다른 파이프라인에서도 동일 패턴을 재사용할지 검토.

---

이 문서는 리팩터링 방향성 논의를 위한 기초 자료이며, 추후 합의된 구조 변경 시 `docs/architecture.md` 갱신이 필요합니다.
