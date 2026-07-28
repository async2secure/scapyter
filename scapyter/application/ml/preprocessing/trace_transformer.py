from typing import Protocol

import numpy as np


class TraceTransformer(Protocol):

    def fit(self, traces: np.ndarray) -> None:
        raise NotImplementedError

    def transform(self, traces: np.ndarray) -> np.ndarray:
        raise NotImplementedError
