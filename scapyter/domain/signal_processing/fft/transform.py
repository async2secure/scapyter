import numpy as np
from scipy import fft

from scapyter.domain.signal_processing.fft.window_type import WindowFunctionType


def compute_fft_magnitudes(
    traces: np.ndarray, sampling_count: int, window_type: WindowFunctionType = None
) -> np.ndarray:
    if window_type == WindowFunctionType.HAMMING:
        traces = traces * np.hamming(sampling_count)
    elif window_type == WindowFunctionType.HANNING:
        traces = traces * np.hanning(sampling_count)

    spectrum = fft.rfft(traces, axis=-1)
    return (np.abs(spectrum) / sampling_count).astype(np.float64)
