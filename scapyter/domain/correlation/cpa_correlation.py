from typing import Optional
import numpy as np
from numba import jit

from scapyter.domain.correlation.correlation import Correlation
from scapyter.domain.value_object import TraceAndModeledLeakage


@jit(nopython=True)
def compute_correlation_numba(
    acc_x2, acc_m, acc_m2, acc_xm, last_trace_sum, processed_traces
):
    # Perform all the heavy numerical computation here
    mean = np.nan_to_num(last_trace_sum / processed_traces)
    variance = np.nan_to_num((acc_x2 / processed_traces) - mean**2)

    numerator = (acc_xm / processed_traces) - np.outer(acc_m / processed_traces, mean)
    denominator_inner = acc_m2 / processed_traces - (acc_m / processed_traces) ** 2
    denominator = np.sqrt(np.outer(denominator_inner, variance))

    # Handling divide-by-zero
    mask = variance == 0.0
    numerator[:, mask] = 0.0
    denominator[:, mask] = 1.0

    return np.nan_to_num(numerator / denominator)


class CpaCorrelation(Correlation):
    """
    CPA Engine with Lazy Initialization.
    Accumulators are initialized on the first call to 'update'.
    """

    def __init__(self) -> None:
        # Accumulators initialized to None
        self._accM: Optional[np.ndarray] = None
        self._accM2: Optional[np.ndarray] = None
        self._accXM: Optional[np.ndarray] = None
        self._acc_x2: Optional[np.ndarray] = None
        self._last_trace_sum: Optional[np.ndarray] = None

        self._processed_traces: int = 0
        self._sample_point_count: Optional[int] = None
        self._guess_count: Optional[int] = None

    def _initialize_buffers(self, guess_count: int, sample_count: int) -> None:
        """Internal helper to allocate memory once dimensions are known."""
        self._guess_count = guess_count
        self._sample_point_count = sample_count

        self._accM = np.zeros((guess_count,), dtype=np.double)
        self._accM2 = np.zeros((guess_count,), dtype=np.double)
        self._accXM = np.zeros((guess_count, sample_count), dtype=np.double)
        self._acc_x2 = np.zeros((sample_count,), dtype=np.double)
        # last_trace_sum is initialized during the first sum in update()

    def update(self, batch: TraceAndModeledLeakage) -> None:
        # Lazy initialization check
        if self._accXM is None:
            self._initialize_buffers(batch.guess_count, batch.sample_count)

        # Update accumulators
        self._accM += batch.modeled_leakage.sum(0)
        self._accM2 += (batch.modeled_leakage**2).sum(0)

        # Matrix multiplication for X*M
        # Using astype(np.double) to prevent slow integer dot products
        self._accXM += np.dot(
            batch.modeled_leakage.transpose(), batch.traces.astype(np.double)
        )

        # Update trace sums for mean/variance calculation
        batch_sum = np.sum(batch.traces, axis=0, dtype=np.float64)
        if self._last_trace_sum is None:
            self._last_trace_sum = batch_sum
        else:
            self._last_trace_sum += batch_sum

        self._acc_x2 += np.square(batch.traces, dtype=np.double).sum(0)
        self._processed_traces += batch.trace_count

    def compute(self) -> np.ndarray:
        """
        Returns the Pearson Correlation matrix.
        Raises RuntimeError if called before any data has been processed.
        """
        if self._processed_traces == 0 or self._accXM is None:
            raise RuntimeError(
                "Cannot compute correlation: No traces have been processed. "
                "Ensure update() is called at least once."
            )

        return compute_correlation_numba(
            self._acc_x2,
            self._accM,
            self._accM2,
            self._accXM,
            self._last_trace_sum,
            self._processed_traces,
        )
