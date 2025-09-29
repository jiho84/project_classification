"""Project settings for Kedro."""

from kedro.config import OmegaConfigLoader
from my_project.hooks import SaveControllerHook, MemoryMonitorHook

CONFIG_LOADER_CLASS = OmegaConfigLoader

CONFIG_LOADER_ARGS = {
    "base_env": "base",
    "default_run_env": "local",
}

# Register hooks
HOOKS = (SaveControllerHook(), MemoryMonitorHook())