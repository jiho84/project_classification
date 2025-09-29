"""
Kedro 프로젝트 설정 파일
MLflow 통합 및 동적 경로 설정
"""

from kedro.config import OmegaConfigLoader
from kedro_mlflow.framework.hooks import MlflowHook

# Use OmegaConfigLoader (Kedro 0.19+)
CONFIG_LOADER_CLASS = OmegaConfigLoader

# Configure the loader arguments
CONFIG_LOADER_ARGS = {
    "base_env": "base",
    "default_run_env": "local",
}

# Configuration source path
CONF_SOURCE = "conf"

# Hook 설정 - kedro-mlflow 기본 Hook만 사용
HOOKS = tuple()
