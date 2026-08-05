import numpy as np


class FFTTransformer:
    def __init__(
        self,
        absolute: bool = True,
        normalize: bool = True,
    ):
        self._absolute = absolute
        self._normalize = normalize

    def transform(self, traces: np.ndarray) -> np.ndarray:
        fft = np.fft.rfft(traces, axis=1)

        if self._normalize:
            fft = fft / traces.shape[1]

        return np.abs(fft) if self._absolute else fft
