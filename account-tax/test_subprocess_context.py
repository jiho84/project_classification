"""Test subprocess MLflow context isolation."""
import os
import subprocess
import sys
import mlflow

# Set tracking URI
mlflow.set_tracking_uri("file:///home/user/projects/kedro_project/account-tax/mlruns")

print("=== Parent Process ===")
print(f"PID: {os.getpid()}")

# Start a run in parent
with mlflow.start_run(run_name="parent-test") as run:
    print(f"Parent run ID: {run.info.run_id}")
    print(f"Parent active run: {mlflow.active_run()}")

    # Create subprocess script
    subprocess_script = """
import mlflow
import os
print("=== Subprocess ===")
print(f"PID: {os.getpid()}")
print(f"Subprocess active run: {mlflow.active_run()}")
print(f"MLFLOW_TRACKING_URI env: {os.environ.get('MLFLOW_TRACKING_URI', 'NOT SET')}")
print(f"MLFLOW_RUN_ID env: {os.environ.get('MLFLOW_RUN_ID', 'NOT SET')}")
"""

    print("\n--- Running subprocess WITHOUT env vars ---")
    subprocess.run([sys.executable, "-c", subprocess_script], check=True)

    print("\n--- Running subprocess WITH env vars ---")
    env = os.environ.copy()
    env["MLFLOW_TRACKING_URI"] = "file:///home/user/projects/kedro_project/account-tax/mlruns"
    env["MLFLOW_RUN_ID"] = run.info.run_id
    subprocess.run([sys.executable, "-c", subprocess_script], env=env, check=True)

print("\n=== Conclusion ===")
print("Subprocess는 부모의 MLflow context를 공유하지 않습니다.")
print("환경 변수로 전달해야 합니다!")
