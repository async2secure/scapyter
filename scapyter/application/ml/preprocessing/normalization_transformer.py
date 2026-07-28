import numpy as np

from scapyter.application.ml.preprocessing.trace_transformer import TraceTransformer


class NormalizationTransformer(TraceTransformer):
    def __init__(self, eps: float = 1e-8):
        self._eps = eps
        self._mean = None
        self._std = None

    def fit(self, traces: np.ndarray):
        self._mean = traces.mean(axis=0)
        self._std = traces.std(axis=0)

    def transform(self, traces: np.ndarray) -> np.ndarray:
        if self._mean is None or self._std is None:
            raise RuntimeError("NormalizationTransformer has not been fitted.")

        return (traces - self._mean) / (self._std + self._eps)
