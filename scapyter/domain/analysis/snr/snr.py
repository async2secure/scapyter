from typing import Dict, List
import numpy as np


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

        # Sort by label so each group's traces are contiguous
        order = np.argsort(labels)
        labels = labels[order]
        traces = traces[order]

        # Find group boundaries
        keys, idx, counts = np.unique(
            labels,
            return_index=True,
            return_counts=True,
        )

        # Sum each group
        sums = np.add.reduceat(traces, idx)
        means = sums / counts[:, None]

        # Repeat each group's index for every trace in that group
        group_ids = np.repeat(np.arange(len(keys)), counts)

        # Compute squared deviations from the group mean
        centered = traces - means[group_ids]
        sq = centered * centered

        # Sum squared deviations (M2) for each group
        m2s = np.add.reduceat(sq, idx)

        # Merge into global statistics
        for key, c2, m2, s2 in zip(keys, counts, means, m2s):
            key = int(key)

            if key not in self._counts:
                self._counts[key] = int(c2)
                self._means[key] = m2
                self._m2s[key] = s2
            else:
                c1 = self._counts[key]
                m1 = self._means[key]
                s1 = self._m2s[key]

                delta = m2 - m1
                total = c1 + c2

                self._means[key] = m1 + delta * (c2 / total)
                self._m2s[key] = (
                        s1
                        + s2
                        + delta * delta * (c1 * c2 / total)
                )
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
