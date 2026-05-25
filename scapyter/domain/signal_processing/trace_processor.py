from abc import ABC, abstractmethod

from scapyter.domain.value_object import Batch


class TraceProcessor(ABC):
    """
    A processor transforms traces while preserving metadata.
    """

    @abstractmethod
    def process(self, batch: Batch) -> Batch:
        pass
