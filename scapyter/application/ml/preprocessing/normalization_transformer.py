import numpy as np

from scapyter.application.ml.preprocessing.trace_preprocessor import TracePreprocessor


class NormalizationTransformer(TracePreprocessor):
    def __init__(self, eps: float = 1e-8):
        self._eps = eps
        self._mean = None
        self._std = None
        self._M2 = None
        self._count = 0

    def partial_fit(self, traces: np.ndarray):
        # traces shape: (n_traces, n_samples)

        batch_count = traces.shape[0]
        batch_mean = traces.mean(axis=0)
        batch_var = traces.var(axis=0)

        if self._mean is None:
            self._mean = batch_mean
            self._M2 = batch_var * batch_count
            self._count = batch_count
            return

        total_count = self._count + batch_count

        delta = batch_mean - self._mean

        self._mean = self._mean + delta * batch_count / total_count

        self._M2 = (
            self._M2
            + batch_var * batch_count
            + delta**2 * self._count * batch_count / total_count
        )

        self._count = total_count

    def finalize(self):
        self._std = np.sqrt(self._M2 / self._count)

    def fit(self, traces: np.ndarray):
        self._mean = traces.mean(axis=0)
        self._std = traces.std(axis=0)

    def transform(self, traces: np.ndarray) -> np.ndarray:
        if self._mean is None or self._std is None:
            raise RuntimeError("NormalizationTransformer has not been fitted.")

        return (traces - self._mean) / (self._std + self._eps)
