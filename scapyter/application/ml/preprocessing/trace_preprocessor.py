from typing import Protocol

import numpy as np


class TracePreprocessor(Protocol):

    def partial_fit(self, traces: np.ndarray):
        raise NotImplementedError

    def transform(self, traces: np.ndarray) -> np.ndarray:
        raise NotImplementedError

    def finalize(self):
        raise NotImplementedError
