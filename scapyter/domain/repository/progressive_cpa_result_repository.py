from abc import ABC, abstractmethod
from scapyter.domain.analysis.correlation.value_objects.progressive_cpa_result import (
    ProgressiveCpaResult,
)


class ProgressiveCpaResultRepository(ABC):

    @abstractmethod
    def save(self, result: ProgressiveCpaResult) -> None:
        raise NotImplementedError

    @abstractmethod
    def load(
        self,
        byte_index: int,
        processed_traces: int,
    ) -> ProgressiveCpaResult:
        raise NotImplementedError

    @abstractmethod
    def load_all(
        self,
        byte_index: int,
    ) -> list[ProgressiveCpaResult]:
        raise NotImplementedError

    @abstractmethod
    def exists(
        self,
        byte_index: int,
        processed_traces: int,
    ) -> bool:
        raise NotImplementedError
