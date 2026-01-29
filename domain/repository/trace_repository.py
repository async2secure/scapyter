from abc import ABC

from domain.entities import OutputTraceBatch
from domain.value_object import Range


class TraceRepository(ABC):

    def get_batch(
        self, trace_range: Range, sample_slice: slice = slice(None)
    ) -> OutputTraceBatch:
        raise NotImplementedError
