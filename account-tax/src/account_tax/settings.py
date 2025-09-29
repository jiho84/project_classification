"""Project settings."""

from kedro.config import OmegaConfigLoader

# Instantiated project hooks (none explicitly; rely on plugin entry points)
HOOKS = ()

# Installed plugins
PLUGINS = []

# Configuration loader
CONFIG_LOADER_CLASS = OmegaConfigLoader

CONFIG_LOADER_ARGS = {
    "base_env": "base",
    "default_run_env": "local",
    "config_patterns": {
        "parameters": ["parameters*", "parameters*/**"],
        "mlflow": ["mlflow*"],
    },
}
