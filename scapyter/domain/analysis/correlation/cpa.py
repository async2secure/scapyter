from typing import Optional
import numpy as np

from scapyter.domain.analysis.correlation.correlation import Correlation
from scapyter.domain.analysis.correlation.value_objects.trace_statistics import (
    TraceStatistics,
)
from scapyter.domain.value_object import TraceAndModeledLeakage


class CpaCorrelation(Correlation):

    def __init__(self) -> None:
        self._accM: Optional[np.ndarray] = None
        self._accM2: Optional[np.ndarray] = None
        self._accXM: Optional[np.ndarray] = None

    def _initialize_buffers(self, guess_count: int, sample_count: int) -> None:
        self._accM = np.zeros(guess_count, dtype=np.float64)
        self._accM2 = np.zeros(guess_count, dtype=np.float64)
        self._accXM = np.zeros((guess_count, sample_count), dtype=np.float64)

    def update(self, batch: TraceAndModeledLeakage) -> None:
        if self._accXM is None:
            self._initialize_buffers(batch.guess_count, batch.sample_count)

        m = batch.modeled_leakage

        # accumulators
        self._accM += m.sum(axis=0)
        self._accM2 += (m * m).sum(axis=0)

        # dot product (keep as float64 input assumed upstream)
        self._accXM += m.T @ batch.traces.astype(np.float64, copy=False)

    def compute(self, trace_statistics: TraceStatistics) -> np.ndarray:
        mean = trace_statistics.mean
        variance = trace_statistics.variance
        n = trace_statistics.processed_traces

        inv_n = 1.0 / n

        m = self._accM * inv_n
        xm = self._accXM * inv_n

        numerator = xm - m[:, None] * mean[None, :]
        denom_left = self._accM2 * inv_n - m**2

        denom_left = np.maximum(denom_left, 1e-12)
        variance = np.maximum(variance, 1e-12)

        denominator = np.sqrt(denom_left[:, None] * variance[None, :])

        return numerator / denominator
