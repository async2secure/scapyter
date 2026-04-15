from abc import ABC

from domain.value_object import Range, Batch


class TraceRepository(ABC):

    def get_batch(self, trace_range: Range, sample_slice: slice = slice(None)) -> Batch:
        raise NotImplementedError
