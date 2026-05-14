from abc import ABC, abstractmethod

import numpy as np

from scapyter.domain.value_object import TraceAndModeledLeakage


class Correlation(ABC):

    @abstractmethod
    def update(self, batch: TraceAndModeledLeakage) -> None:
        raise NotImplementedError

    @abstractmethod
    def compute(self) -> np.ndarray:
        raise NotImplementedError
