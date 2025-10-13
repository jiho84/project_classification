"""Diagnose Kedro-MLflow hook registration."""
import sys
from pathlib import Path

# Add project to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from kedro.framework.project import pipelines
from kedro.framework.startup import bootstrap_project
from kedro.framework.session import KedroSession

# Bootstrap project
bootstrap_project(Path.cwd())

# Check if hooks are registered
from kedro.framework.project import settings

print("=" * 80)
print("HOOKS registered in settings.py:")
print(f"  HOOKS = {settings.HOOKS}")
print()

# Check if kedro_mlflow is auto-registered
import importlib.metadata as metadata

print("Installed entry points for 'kedro.hooks':")
eps = metadata.entry_points()
if hasattr(eps, 'select'):
    hook_eps = eps.select(group='kedro.hooks')
else:
    hook_eps = eps.get('kedro.hooks', [])

for ep in hook_eps:
    print(f"  - {ep.name}: {ep.value}")
print()

# Try to get session hooks
print("Attempting to create KedroSession...")
try:
    with KedroSession.create() as session:
        context = session.load_context()
        print(f"Session created successfully")
        print(f"Hooks in session: {session._hook_manager.get_plugins()}")
except Exception as e:
    print(f"Error creating session: {e}")
