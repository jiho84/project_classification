"""Lightweight partitioned Parquet dataset compatible with Kedro catalogs."""

from __future__ import annotations

from pathlib import Path
from typing import Dict, Iterable, Mapping, MutableMapping, Optional

import pandas as pd
from kedro.io import AbstractDataset


class PartitionedParquetDataset(AbstractDataset[Dict[str, pd.DataFrame], Dict[str, pd.DataFrame]]):
    """Persist dictionaries of pandas DataFrames into per-partition Parquet files."""

    def __init__(
        self,
        path: str,
        filename_suffix: str = ".parquet",
        load_args: Optional[Mapping[str, object]] = None,
        save_args: Optional[Mapping[str, object]] = None,
    ) -> None:
        self._path = Path(path)
        self._suffix = filename_suffix
        self._load_args = {"engine": "pyarrow", **(load_args or {})}
        default_save = {"engine": "pyarrow", "index": False}
        self._save_args = {**default_save, **(save_args or {})}

    def _load(self) -> Dict[str, pd.DataFrame]:
        if not self._path.exists():
            return {}

        partitions: Dict[str, pd.DataFrame] = {}
        for file_path in sorted(self._path.glob(f"*{self._suffix}")):
            partition_key = file_path.stem
            partitions[partition_key] = pd.read_parquet(file_path, **self._load_args)
        return partitions

    def _save(self, data: MutableMapping[str, pd.DataFrame]) -> None:
        self._path.mkdir(parents=True, exist_ok=True)
        for partition_key, frame in data.items():
            file_path = self._path / f"{partition_key}{self._suffix}"
            frame.to_parquet(file_path, **self._save_args)

    def _exists(self) -> bool:
        if not self._path.exists():
            return False
        return any(self._path.glob(f"*{self._suffix}"))

    def _describe(self) -> Dict[str, object]:
        return {
            "path": str(self._path),
            "filename_suffix": self._suffix,
            "load_args": self._load_args,
            "save_args": self._save_args,
        }


__all__: Iterable[str] = ["PartitionedParquetDataset"]
