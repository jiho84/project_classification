# Model Training Module Architecture (DeepSpeed + LoRA)

본 문서는 `/backup/src/model_training` 폴더에 포함된 학습 스크립트들의 구조와 입·출력 계약을 설계 관점에서 정리한 것입니다. 현재 코드베이스는 **DeepSpeed 런처**를 기준으로 작성되었으며, LoRA(PEFT) 어댑터와 MLflow 로깅을 지원하도록 구성돼 있습니다.

## 구성 파일 개요
| 파일 | 역할 요약 |
|------|-----------|
| `deepspeed_trainer.py` | 주요 진입점(`main`)과 데이터 로딩·평가·체크포인트 관리 유틸리티를 포함한 통합 학습 스크립트 |
| `train.py` | 초기 버전의 학습 스크립트 (구조는 `deepspeed_trainer.py`와 유사) |
| `utils.py` | Metric, GPU 모니터링, 경로 관리 등 학습 보조 클래스/함수 정의 |
| `debug_utils.py` | 중단 원인 분석, 메모리/시스템 모니터링 등 디버깅 보조 도구 |
| `__init__.py` | 패키지 초기화 (주요 로직 없음) |

## 주요 함수/클래스 정의
### deepspeed_trainer.py
| 이름 | 입력 | 출력 | 설명 |
|------|------|------|------|
| `setup_tokenizer(model_name, trust_remote_code=True)` | 모델 이름(str) | HF `AutoTokenizer` 인스턴스 | PAD/EOS 토큰 검증 및 padding 방향 설정을 수행. rank 0에서만 정보 출력 |
| `auto_setup_environment()` | 없음 | 없음 | MKL·CUDA 관련 환경 변수를 설정하고 Flash Attention 등 최적화 옵션 활성화 |
| `pkl_to_arrow_once(pkl_path, arrow_dir, shard_size="500MB")` | PKL 경로, Arrow 캐시 경로, 샤드 크기 | 없음 | PKL 데이터를 한번만 Arrow 포맷으로 변환하고 체크섬 기반 캐시를 유지. num_classes/metadata도 저장 |
| `build_loaders(cfg)` | Hydra/OmegaConf 설정 객체 | `(train_loader, val_loader, test_loader, num_classes, tokenizer)` | PKL→Arrow→HF Dataset 로딩, 토크나이즈, `DistributedSampler` 적용까지 처리 |
| `evaluate_model_comprehensive(model_engine, eval_dataloader, local_rank=0, metric_tracker=None, prefix="")` | DeepSpeed 엔진, DataLoader, rank, MetricTracker | dict(메트릭 값) | loss/정확도/F1 등 계산하고 rank 0에서만 MLflow 로깅 수행 |
| `evaluate_model(model_engine, eval_dataloader)` | 모델 엔진, DataLoader | dict | 위 함수의 래퍼 (기존 코드 호환용) |
| `save_model_metadata(save_dir, model, tokenizer, config, num_classes)` | 저장 경로, 모델, 토크나이저, 설정, 클래스 수 | 메타데이터 파일 경로(str) | 모델/데이터 관련 설정을 JSON으로 저장 |
| `save_dataset_metadata_once(arrow_dir, checkpoint_root_dir, local_rank=0)` | Arrow 캐시 경로, 체크포인트 루트, rank | 저장 경로(str) 또는 None | 전체 데이터셋 메타데이터를 체크포인트 폴더에 1회 저장 (rank 0 전용) |
| `find_latest_checkpoint(checkpoint_dir, local_rank=0)` | 체크포인트 경로(Path) | `(dir, tag, epoch)` | DeepSpeed `latest` 파일이나 `epoch_*` 폴더를 탐색해 가장 최근 체크포인트 태그 반환 |
| `load_checkpoint_deepspeed(model_engine, scheduler, checkpoint_dir, tag=None, local_rank=0)` | DeepSpeed 엔진, LR 스케줄러, 디렉터리, 태그 | `(client_state, start_epoch, resume_run_id)` | DeepSpeed 내장 `load_checkpoint` 사용. 클래스 수 불일치 시 경고 및 재초기화 |
| `main()` | CLI 인자 | 없음 | Deepspeed 분산 초기화, 데이터 로드, 체크포인트 자동 탐색, MLflow run 구성, 학습 루프 실행 기초 로직 포함 |
| `safe_log_params(params_dict)` *(중첩 함수)* | dict | 없음 | MLflow 파라미터 중복 로깅 예외 처리 (`already logged`) |

### utils.py
| 이름 | 입력 | 출력 | 설명 |
|------|------|------|------|
| `MetricTracker(local_rank=0)` | rank | - | 평가 지표 계산/저장. **mlflow.log_metrics를 rank 0에서만 호출** |
| `MetricTracker.calculate_and_log(predictions, labels, prefix="", step=None)` | 예측/정답 (Tensor/ndarray) | dict | Accuracy/F1/Precision/Recall 계산 후 rank 0에서만 MLflow 로깅 |
| `GPUMonitor(local_rank=0)` | rank | - | GPU 메모리 사용량 조회 및 피크 추적 |
| `GPUMonitor.log_memory(prefix="")` | 접두사 | dict | rank 0에서만 메모리 사용 로그 출력 |
| `RankAwareLogger(rank=0)` | rank | - | rank 0에서만 `info/warning/error/debug` 메시지를 출력하는 래퍼 |
| `PathManager(config, local_rank=0)` | 설정, rank | - | 데이터/캐시/체크포인트 경로를 일괄 관리하고 폴더 생성 |
| `setup_logging()` | 없음 | `logging.Logger` | 로깅 포맷 초기화 및 외부 라이브러리 레벨 조정 |
| `set_seed(seed)` | 정수 | 없음 | random/np/torch 시드 설정 및 CUDNN deterministic 옵션 적용 |
| `calculate_total_parameters(model)` | 모델 | dict | 총 파라미터 수 / 학습 가능 파라미터 수 / 비율 계산 |
| `create_optimizer_grouped_parameters(model, weight_decay=0.01)` | 모델, weight_decay | list | AdamW 스타일 가중치 감쇠 그룹 생성 |
| 기타 JSON/시간/메모리 보조 함수 | - | - | 저장·포매팅 용도 |

### debug_utils.py
| 이름 | 입력 | 출력 | 설명 |
|------|------|------|------|
| `TrainingDebugger(config, logger)` | 설정 dict, logger | - | 학습 중단 원인 분석 도구. 메모리/시스템 지표 추적 및 시그널 핸들러 등록 |
| `track_memory_usage(step, phase)` | 스텝 번호, 단계 | None | GPU/CPU 메모리 사용량 기록. 임계값 초과 시 경고 |
| `track_system_metrics(step)` | 스텝 번호 | None | CPU, 디스크, 네트워크, 프로세스 수 등을 기록 |
| `track_step_timing(step, duration, phase)` | 스텝, 수행시간, 단계 | None | 과도한 스텝 시간 감지 |
| `log_exception(exception, context)` | 예외 객체, 컨텍스트 문자열 | None | 예외 정보와 스택 트레이스를 기록하고 디버그 정보를 저장 |
| `save_debug_info(interruption_type, **kwargs)` | 중단 유형, 추가 정보 | None | 디버그 데이터(JSON) 저장. 시그널/예외 시 호출 |
| `check_environment_issues()` | 없음 | dict | GPU 메모리/디스크 공간/시스템 메모리 부족 여부 확인 |

*(파일 전체에는 메모리 히스토리 직렬화 등 추가 메서드가 포함되어 있으나, 위 표에 핵심 기능만 요약했습니다.)*

## DeepSpeed + MLflow 연동 시 rank==0 로깅 전략
분산 학습 환경(DeepSpeed)에서는 각 프로세스가 동일한 코드를 실행합니다. 중복 로깅을 방지하고 한 번만 메트릭/파라미터를 기록하기 위해 코드 전체에서 **`local_rank` 변수를 통해 rank 0 프로세스만 로그를 남기도록 설계**되어 있습니다.

적용 방식 요약:
- `local_rank = int(os.getenv("LOCAL_RANK", 0))`를 전역으로 선언하고 `torch.cuda.set_device(local_rank)` 수행.
- MLflow 관련 로직은 `if local_rank == 0 and mlflow.active_run():` 등의 조건문으로 감싸 단일 프로세스에서만 실행.
    - 예) `MetricTracker.calculate_and_log` 내부에서 rank 0만 `mlflow.log_metrics` 호출.
    - `RankAwareLogger` 역시 rank 0에서만 표준 출력으로 정보 메시지를 남김.
- 체크포인트 변환, Arrow 캐시 생성 등 부하가 큰 작업은 rank 0에서만 수행하고 `torch.distributed.barrier()`로 동기화.

이 구조 덕분에 DeepSpeed 다중 프로세스 환경에서도 MLflow 메트릭/파라미터/아티팩트가 **한 번**만 기록되며, 로그 중복으로 인한 실험 관리 혼선이 발생하지 않습니다.

## 아키텍처상의 유의점 및 제안
1. **train.py vs deepspeed_trainer.py**: 두 파일이 유사한 기능을 제공하므로, 향후 유지보수를 위해 공통 모듈(`core.py`)로 로직을 추출하는 것을 권장합니다.
2. **파라미터 표준화**: `pyproject.toml` 기반 의존성 관리로 전환 시 MLflow 런 라벨/메트릭 명칭을 config 기반으로 통일하면 실험 비교가 용이합니다.
3. **추론 파이프라인 재사용**: 토크나이저·라벨 맵·LoRA 설정을 inference에도 공유할 수 있도록 config를 계층화하고 `save_model_metadata` 결과를 적극 활용하세요.
4. **분산 로그 검증**: rank 0 체크가 누락된 log/print/파일 생성 코드가 있는지 주기적으로 점검하는 것이 좋습니다.

## 결론
- `backup/src/model_training` 모듈은 **DeepSpeed + LoRA** 학습과 체크포인트 자동 재개, MLflow 로깅까지 포함한 통합 실행 환경을 제공합니다.
- 함수/클래스 별로 입·출력 계약이 명확하게 정리되어 있으며, rank-aware 로깅 전략으로 멀티 프로세스에서도 혼선 없이 추적 가능합니다.
- 위 표와 설명을 토대로, 학습 파이프라인을 Kedro 노드로 감쌀 때 `instantiate → adapt → optimizer → deepspeed → train → evaluate` 단계가 자연스럽게 분리될 수 있습니다.

이 문서는 설계자 관점에서 작성되었으며, 더 깊은 수준의 모듈화나 테스트 전략이 필요할 경우 추가 설계 문서를 갱신해 주시기 바랍니다.
