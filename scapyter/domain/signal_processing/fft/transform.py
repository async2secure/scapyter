import numpy as np
from scipy import fft

from scapyter.domain.signal_processing.fft.window_type import WindowFunctionType


def compute_fft(
    traces: np.ndarray,
    sampling_count: int,
    window_type: WindowFunctionType | None = None,
    remove_dc: bool = True,
    remove_dc_bin: bool = True,
) -> np.ndarray:
    """
    Compute FFT magnitude features for side-channel analysis.

    Args:
        traces:
            Shape: (trace_count, sample_count)
            Batch of traces.

        sampling_count:
            Number of samples per trace.

        window_type:
            Optional window function.

        remove_dc:
            Remove per-trace DC offset before FFT.

        remove_dc_bin:
            Remove FFT bin 0 from output.

    Returns:
        FFT magnitude features.
        Shape:
            (trace_count, sampling_count//2 + 1)
            or
            (trace_count, sampling_count//2) if remove_dc_bin=True
    """

    # Ensure floating point
    traces = traces.astype(np.float64)

    # ----------------------------------------
    # Remove DC offset per trace
    # ----------------------------------------
    if remove_dc:
        traces = traces - np.mean(
            traces,
            axis=-1,
            keepdims=True,
        )

    # ----------------------------------------
    # Apply window
    # ----------------------------------------
    if window_type == WindowFunctionType.HAMMING:
        traces = traces * np.hamming(sampling_count)

    elif window_type == WindowFunctionType.HANNING:
        traces = traces * np.hanning(sampling_count)

    # ----------------------------------------
    # FFT
    # ----------------------------------------
    spectrum = fft.rfft(
        traces,
        axis=-1,
    )

    # ----------------------------------------
    # Magnitude
    # ----------------------------------------
    magnitude = np.abs(spectrum)

    # Normalize
    magnitude = magnitude / sampling_count

    # ----------------------------------------
    # Remove DC frequency bin
    # ----------------------------------------
    if remove_dc_bin:
        magnitude = magnitude[:, 1:]

    return magnitude.astype(np.float64)
