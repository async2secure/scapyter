import numpy as np
import pytest
from scipy.fft import rfft

from scapyter.domain.signal_processing.fft.transform import compute_fft_magnitudes
from scapyter.domain.signal_processing.fft.window_type import WindowFunctionType


def test_compute_fft_magnitudes_without_window():
    traces = np.array([[1.0, 2.0, 3.0, 4.0]])
    sampling_count = 4

    result = compute_fft_magnitudes(traces, sampling_count)

    expected = np.abs(rfft(traces, axis=-1) / sampling_count).astype(np.float64)

    np.testing.assert_allclose(result, expected)
    assert result.dtype == np.float64


def test_compute_fft_magnitudes_with_hamming_window():
    traces = np.array([[1.0, 2.0, 3.0, 4.0]])
    sampling_count = 4

    result = compute_fft_magnitudes(
        traces,
        sampling_count,
        window_type=WindowFunctionType.HAMMING,
    )

    spectrum = rfft(traces, axis=-1) / sampling_count
    spectrum *= np.hamming(spectrum.shape[-1])

    expected = np.abs(spectrum).astype(np.float64)

    np.testing.assert_allclose(result, expected)


def test_compute_fft_magnitudes_with_hanning_window():
    traces = np.array([[1.0, 2.0, 3.0, 4.0]])
    sampling_count = 4

    result = compute_fft_magnitudes(
        traces,
        sampling_count,
        window_type=WindowFunctionType.HANNING,
    )

    spectrum = rfft(traces, axis=-1) / sampling_count
    spectrum *= np.hanning(spectrum.shape[-1])

    expected = np.abs(spectrum).astype(np.float64)

    np.testing.assert_allclose(result, expected)


def test_compute_fft_magnitudes_handles_multiple_traces():
    traces = np.array(
        [
            [1.0, 2.0, 3.0, 4.0],
            [4.0, 3.0, 2.0, 1.0],
        ]
    )
    sampling_count = 4

    result = compute_fft_magnitudes(traces, sampling_count)

    expected = np.abs(rfft(traces, axis=-1) / sampling_count).astype(np.float64)

    np.testing.assert_allclose(result, expected)
    assert result.shape == expected.shape


@pytest.mark.parametrize(
    "window_type",
    [
        None,
        WindowFunctionType.HAMMING,
        WindowFunctionType.HANNING,
    ],
)
def test_compute_fft_magnitudes_output_is_non_negative(window_type):
    traces = np.random.rand(3, 8)

    result = compute_fft_magnitudes(
        traces,
        sampling_count=8,
        window_type=window_type,
    )

    assert np.all(result >= 0)
