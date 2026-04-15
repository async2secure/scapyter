from typing import Dict, List
import numpy as np

from domain.value_object import Range


class _GroupedStreamingStats:
    """
    Internal helper: maintains per-group Welford stats.
    """

    def __init__(self, trace_dim: int):
        self._counts: Dict[int, int] = {}
        self._means: Dict[int, np.ndarray] = {}
        self._m2s: Dict[int, np.ndarray] = {}
        self._trace_dim = trace_dim

    def update_chunk(self, traces: np.ndarray, labels: np.ndarray) -> None:
        """
        traces: (N, D)
        labels: (N,)
        """

        if len(traces) == 0:
            return

        # -----------------------------
        # Vectorized grouping (within chunk)
        # -----------------------------
        keys, inverse = np.unique(labels, return_inverse=True)
        n_groups = len(keys)

        counts = np.bincount(inverse)

        sums = np.zeros((n_groups, self._trace_dim), dtype=np.float64)
        np.add.at(sums, inverse, traces)

        means = sums / counts[:, None]

        # Compute M2 (within chunk)
        centered = traces - means[inverse]
        sq = centered**2

        m2s = np.zeros((n_groups, self._trace_dim), dtype=np.float64)
        np.add.at(m2s, inverse, sq)

        # -----------------------------
        # Merge into global stats
        # -----------------------------
        for i, key in enumerate(keys):
            key = int(key)

            c2 = counts[i]
            m2 = means[i]
            s2 = m2s[i]

            if key not in self._counts:
                self._counts[key] = c2
                self._means[key] = m2
                self._m2s[key] = s2
            else:
                c1 = self._counts[key]
                m1 = self._means[key]
                s1 = self._m2s[key]

                delta = m2 - m1
                total = c1 + c2

                self._means[key] = m1 + delta * (c2 / total)
                self._m2s[key] = s1 + s2 + (delta**2) * (c1 * c2 / total)
                self._counts[key] = total

    # -----------------------------
    # Final stats accessors
    # -----------------------------
    @property
    def mean_list(self) -> List[np.ndarray]:
        return list(self._means.values())

    @property
    def largest_count_key(self) -> int:
        return max(self._counts, key=self._counts.get)

    def variance_of(self, key: int) -> np.ndarray:
        count = self._counts[key]
        if count < 2:
            return np.zeros(self._trace_dim, dtype=np.float64)
        return self._m2s[key] / count


# ---------------------------------
# DROP-IN REPLACEMENT
# ---------------------------------
class ProgressiveSnr:
    """
    Streaming + vectorized hybrid implementation.

    Same API:
        - update(traces=..., hex_array=...)
        - finalize()
    """

    def __init__(self):
        self._trace_dim = None
        self._stats = None

    def update(self, *, traces: np.ndarray, hex_array: np.ndarray) -> None:
        """
        Accepts chunks of data.
        """
        if self._stats is None:
            self._trace_dim = traces.shape[1]
            self._stats = _GroupedStreamingStats(self._trace_dim)

        elif traces.shape[1] != self._trace_dim:
            raise ValueError("Trace dimension mismatch")

        self._stats.update_chunk(traces, hex_array)

    def finalize(self) -> np.ndarray:
        """
        SNR = signal variance / noise variance
        """

        if not self._stats._means:
            return np.array([])

        # Stack group means
        means = np.stack(self._stats.mean_list)

        # Between-group variance
        signal_variance = np.var(means, axis=0)

        # Noise from largest group
        key = self._stats.largest_count_key
        noise_variance = self._stats.variance_of(key)

        return np.nan_to_num(signal_variance / noise_variance)
