from pathlib import Path

import numpy as np

from scapyter.domain.repository.cpa_result_repository import (
    CpaResultRepository,
)
from scapyter.domain.value_object import CpaByteResult


class NpzCpaResultRepository(CpaResultRepository):
    def __init__(self, directory: Path):
        self._directory = directory
        self._directory.mkdir(parents=True, exist_ok=True)

    def _path(self, byte_index: int) -> Path:
        return self._directory / f"byte_{byte_index:02d}.npz"

    def save(self, result: CpaByteResult) -> None:
        np.savez_compressed(
            self._path(result.byte_index),
            key_candidates=np.asarray(
                result.key_candidates,
                dtype=object,
            ),
            corr_matrix=result.corr_matrix,
        )

    def load(self, byte_index: int) -> CpaByteResult:
        path = self._path(byte_index)

        with np.load(path, allow_pickle=True) as data:
            return CpaByteResult(
                byte_index=byte_index,
                key_candidates=data["key_candidates"].tolist(),
                corr_matrix=data["corr_matrix"],
            )

    def load_all(self) -> list[CpaByteResult]:
        results = []

        for path in sorted(self._directory.glob("byte_*.npz")):
            byte_index = int(path.stem.removeprefix("byte_"))
            results.append(self.load(byte_index))

        return results

    def exists(self, byte_index: int) -> bool:
        return self._path(byte_index).exists()
