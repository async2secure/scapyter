from abc import ABC, abstractmethod

import numpy as np

from scapyter.domain.analysis.correlation.value_objects.trace_statistics import (
    TraceStatistics,
)
from scapyter.domain.value_object import TraceAndModeledLeakage


class Correlation(ABC):

    @abstractmethod
    def update(self, batch: TraceAndModeledLeakage) -> None:
        raise NotImplementedError

    @abstractmethod
    def compute(self, trace_statics: TraceStatistics) -> np.ndarray:
        raise NotImplementedError
