"""Helper for loading pipeline intermediates via Kedro catalog.

Usage (inside Jupyter notebook or IPython):

from load_intermediate_outputs import get_catalog
catalog = get_catalog()
prepared = catalog.load("prepared_datasets_mlflow")
train_df = prepared["train"]
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict

import pandas as pd
import sys
from kedro.framework.project import configure_project
from kedro.framework.session import KedroSession

PROJECT_PATH = Path(__file__).resolve().parents[1]
sys.path.append(str(PROJECT_PATH / "src"))
configure_project("account_tax")


def get_catalog():
    """Return a Kedro catalog bound to a temporary session.

    The caller is responsible for closing the session by calling
    `catalog._session.close()` when finished. For simple analysis
    in notebooks, relying on the context manager below is safer.
    """

    session = KedroSession.create(project_path=PROJECT_PATH)
    context = session.load_context()
    catalog = context.catalog
    catalog._session = session  # type: ignore[attr-defined]
    return catalog


def _materialize_partitions(partitions) -> Dict[str, pd.DataFrame]:
    """Resolve PartitionedDataset output (possibly lazy) into pandas DataFrames."""

    resolved: Dict[str, pd.DataFrame] = {}
    for partition_id, loader in partitions.items():
        resolved[partition_id] = loader() if callable(loader) else loader
    return resolved


def load_prepared_datasets(materialize: bool = True) -> Dict[str, pd.DataFrame]:
    """Load `prepared_datasets_mlflow`; optionally materialise partitions to DataFrames."""

    with KedroSession.create(project_path=PROJECT_PATH) as session:
        context = session.load_context()
        partitions = context.catalog.load("prepared_datasets_mlflow")
    return _materialize_partitions(partitions) if materialize else partitions


def load_text_datasets(materialize: bool = True) -> Dict[str, pd.DataFrame]:
    """Load `text_datasets_mlflow`; optionally materialise partitions to DataFrames."""

    with KedroSession.create(project_path=PROJECT_PATH) as session:
        context = session.load_context()
        partitions = context.catalog.load("text_datasets_mlflow")
    return _materialize_partitions(partitions) if materialize else partitions


def summarise_dataframe(df: pd.DataFrame) -> Dict[str, object]:
    """Return a lightweight summary for quick notebook inspection."""

    return {
        "rows": len(df),
        "columns": list(df.columns),
        "preview": df.head(3).to_dict(orient="records"),
    }


if __name__ == "__main__":
    catalog = get_catalog()
    try:
        prepared = _materialize_partitions(catalog.load("prepared_datasets_mlflow"))
        text = _materialize_partitions(catalog.load("text_datasets_mlflow"))

        for name, partitions in {
            "prepared_datasets_mlflow": prepared,
            "text_datasets_mlflow": text,
        }.items():
            print(f"\n{name} partitions: {list(partitions.keys())}")
            sample_key = next(iter(partitions))
            summary = summarise_dataframe(partitions[sample_key])
            print(f"Sample partition '{sample_key}' summary: {summary}")
    finally:
        catalog._session.close()  # type: ignore[attr-defined]
