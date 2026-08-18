from abc import ABC, abstractmethod

from scapyter.domain.value_object import Batch, Range


class TraceProcessor(ABC):
    """
    A processor transforms traces while preserving metadata.
    """

    @abstractmethod
    def process(self, batch: Batch) -> Batch:
        pass

    def output_shape(self, input_shape: tuple[int, int]) -> tuple[int, int]:
        return input_shape
