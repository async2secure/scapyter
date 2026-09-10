from pathlib import Path

import numpy as np

from scapyter.domain.analysis.correlation.value_objects.progressive_cpa_result import (
    ProgressiveCpaResult,
)
from scapyter.domain.repository.progressive_cpa_result_repository import (
    ProgressiveCpaResultRepository,
)
from scapyter.domain.value_object import CpaByteResult


class NpzProgressiveCpaResultRepository(ProgressiveCpaResultRepository):
    def __init__(self, directory: Path):
        self._directory = directory
        self._directory.mkdir(parents=True, exist_ok=True)

    def _byte_directory(self, byte_index: int) -> Path:
        return self._directory / f"byte_{byte_index:02d}"

    def _path(
        self,
        byte_index: int,
        processed_traces: int,
    ) -> Path:
        return self._byte_directory(byte_index) / (f"traces_{processed_traces:08d}.npz")

    def save(self, result: ProgressiveCpaResult) -> None:
        byte_result = result.byte_result

        byte_directory = self._byte_directory(byte_result.byte_index)
        byte_directory.mkdir(parents=True, exist_ok=True)

        np.savez_compressed(
            self._path(
                byte_index=byte_result.byte_index,
                processed_traces=result.processed_traces,
            ),
            processed_traces=result.processed_traces,
            byte_index=byte_result.byte_index,
            key_candidates=np.asarray(
                byte_result.key_candidates,
                dtype=object,
            ),
            corr_matrix=byte_result.corr_matrix,
        )

    def load(
        self,
        byte_index: int,
        processed_traces: int,
    ) -> ProgressiveCpaResult:
        path = self._path(
            byte_index=byte_index,
            processed_traces=processed_traces,
        )

        with np.load(path, allow_pickle=True) as data:
            return ProgressiveCpaResult(
                processed_traces=int(data["processed_traces"]),
                byte_result=CpaByteResult(
                    byte_index=int(data["byte_index"]),
                    key_candidates=data["key_candidates"].tolist(),
                    corr_matrix=data["corr_matrix"],
                ),
            )

    def load_all(
        self,
        byte_index: int,
    ) -> list[ProgressiveCpaResult]:
        byte_directory = self._byte_directory(byte_index)

        if not byte_directory.exists():
            return []

        results = []

        for path in sorted(byte_directory.glob("traces_*.npz")):
            processed_traces = int(path.stem.removeprefix("traces_"))

            results.append(
                self.load(
                    byte_index=byte_index,
                    processed_traces=processed_traces,
                )
            )

        return results

    def exists(
        self,
        byte_index: int,
        processed_traces: int,
    ) -> bool:
        return self._path(
            byte_index=byte_index,
            processed_traces=processed_traces,
        ).exists()
