# MLflow Context 문제 해결 방안 비교

## 방안 2: Subprocess 내에서 MLflow 초기화

### 장점:
- Subprocess가 독립적으로 MLflow 관리
- main_yaml.py에서 명시적 제어 가능

### 단점:
- **두 개의 MLflow run이 생성됨** (Kedro run + Subprocess run)
- Kedro-MLflow의 자동 로깅과 충돌 가능
- 메트릭이 두 run에 분산됨 (일관성 문제)

### 구현:
```python
# main_yaml.py
def main():
    # Subprocess에서 새로운 run 시작
    mlflow.set_tracking_uri(os.environ["MLFLOW_TRACKING_URI"])
    with mlflow.start_run(run_name="training-subprocess"):
        trainer.train()
        # 메트릭이 이 run에 기록됨
```

### 문제점:
- Kedro pipeline run과 training run이 분리됨
- Kedro catalog artifacts와 training artifacts가 다른 run에 저장됨
- **프로젝트의 MLflow 통합 철학과 불일치**

---

## 방안 3: Kedro run_id를 Subprocess에 전달 (🏆 추천)

### 장점:
- **단일 MLflow run 유지** (일관성)
- Kedro-MLflow와 완벽 호환
- 모든 artifacts와 메트릭이 한 곳에 모임
- 프로젝트 아키텍처와 일관성 유지

### 단점:
- 약간의 코드 수정 필요

### 구현:
```python
# nodes.py - launch_training
def launch_training(...):
    # Kedro MLflow run context 가져오기
    if mlflow and mlflow.active_run():
        run_id = mlflow.active_run().info.run_id
        tracking_uri = mlflow.get_tracking_uri()
        experiment_name = mlflow.active_run().info.experiment_id  # or name

    # 환경 변수로 전달
    env = os.environ.copy()
    env["MLFLOW_TRACKING_URI"] = tracking_uri
    env["MLFLOW_RUN_ID"] = run_id
    env["MLFLOW_EXPERIMENT_NAME"] = experiment_name

    subprocess.run(cmd, env=env, ...)
```

```python
# main_yaml.py - setup에서 자동으로 환경 변수 읽음
# MLflowCallback이 MLFLOW_RUN_ID를 감지하면 자동으로 reattach!
# 코드 변경 거의 불필요
```

### 동작 방식:
1. Kedro MlflowHook이 pipeline 시작 시 run 생성
2. nodes.py에서 run_id를 환경 변수로 추출
3. subprocess 실행 시 환경 변수 전달
4. MLflowCallback이 환경 변수 감지 → 기존 run에 reattach
5. 학습 메트릭이 Kedro run에 기록됨
6. nodes.py에서 artifact 업로드 (mlflow.active_run() 사용 가능)

---

## 🎯 결론:

**방안 3이 훨씬 우수합니다:**
- Kedro-MLflow 통합 철학 유지
- 단일 run으로 전체 파이프라인 추적
- 최소한의 코드 변경
- MLflow UI에서 명확한 추적
