from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

from kedro.framework.project import configure_project
from kedro.framework.session import KedroSession


class TimingHook:
    def __init__(self) -> None:
        self._starts: dict[str, float] = {}
        self.durations: dict[str, float] = {}

    def before_node_run(self, node, inputs):  # type: ignore[override]
        self._starts[node.name] = time.perf_counter()

    def after_node_run(self, node, inputs, outputs):  # type: ignore[override]
        start = self._starts.pop(node.name, None)
        if start is not None:
            elapsed = time.perf_counter() - start
            self.durations[node.name] = elapsed
            print(f"NODE {node.name} took {elapsed:.3f}s", flush=True)


def profile_pipeline(project_path: Path, pipeline_name: str, env: str) -> dict[str, float]:
    if str(project_path / "src") not in sys.path:
        sys.path.append(str(project_path / "src"))
    configure_project("account_tax")

    timing_hook = TimingHook()
    with KedroSession.create(project_path=project_path, env=env) as session:
        session._hook_manager.register(timing_hook)  # type: ignore[attr-defined]
        try:
            session.run(pipeline_name=pipeline_name)
        finally:
            session._hook_manager.unregister(timing_hook)  # type: ignore[attr-defined]
    return timing_hook.durations


def main() -> None:
    parser = argparse.ArgumentParser(description="Profile Kedro pipeline node durations")
    parser.add_argument("--pipeline", default="__default__", help="Pipeline name")
    parser.add_argument("--env", default="repro", help="Kedro configuration environment")
    parser.add_argument("--output", type=Path, default=None, help="Optional path to save JSON durations")
    args = parser.parse_args()

    durations = profile_pipeline(Path.cwd(), args.pipeline, args.env)
    serialized = json.dumps(
        {name: round(seconds, 3) for name, seconds in durations.items()},
        indent=2,
        ensure_ascii=False,
    )
    print(serialized)
    if args.output:
        args.output.write_text(serialized, encoding="utf-8")


if __name__ == "__main__":
    main()
