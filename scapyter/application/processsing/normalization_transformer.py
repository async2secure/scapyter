import numpy as np

from scapyter.application.processsing.trace_transformer import TraceTransformer


class NormalizationTransformer(TraceTransformer):
    def __init__(self, eps: float = 1e-8):
        self._eps = eps

    def transform(self, traces: np.ndarray) -> np.ndarray:
        mean = traces.mean(axis=0)
        std = traces.std(axis=0)
        return (traces - mean) / (std + self._eps)
