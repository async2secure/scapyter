import numpy as np
from scipy.fft import rfft

from scapyter.domain.signal_processing.fft import WindowFunctionType


def compute_fft_magnitudes(
    traces: np.ndarray, sampling_count: int, window_type: WindowFunctionType = None
) -> np.ndarray:
    """
    Pure Domain Math:
    Transforms time-domain traces into frequency-domain magnitudes.
    """
    # 1. Transform to Frequency Domain
    # rfft is faster for real-valued signals
    spectrum = rfft(traces, axis=-1) / sampling_count

    # 2. Apply Windowing Strategy
    if window_type == WindowFunctionType.HAMMING:
        spectrum *= np.hamming(spectrum.shape[-1])
    elif window_type == WindowFunctionType.HANNING:
        spectrum *= np.hanning(spectrum.shape[-1])

    # 3. Return Absolute Magnitude (Real numbers for downstream analysis)
    return np.abs(spectrum).astype(np.float64)
