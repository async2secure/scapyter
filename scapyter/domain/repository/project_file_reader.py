from abc import ABC

from scapyter.domain.value_object import Range, Batch


class ProjectFileReader(ABC):

    def get_batch(self, trace_range: Range, sample_slice: slice = slice(None)) -> Batch:
        raise NotImplementedError

    def get_single_batch(self, index: int, sample_range: Range | None = None) -> Batch:
        raise NotImplementedError
