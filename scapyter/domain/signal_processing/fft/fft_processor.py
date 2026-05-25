from scipy.fft import rfftfreq

from scapyter.domain.signal_processing.fft.transform import (
    compute_fft_magnitudes,
)
from scapyter.domain.signal_processing.fft.window_type import WindowFunctionType
from scapyter.domain.signal_processing.trace_processor import TraceProcessor
from scapyter.domain.value_object import Batch


class FFTProcessor(TraceProcessor):
    def __init__(
        self,
        sampling_rate: float,  # Added: we need this for the x-axis
        window_type: WindowFunctionType | None = None,
    ):
        self.sampling_rate = sampling_rate
        self.window_type = window_type

    def output_shape(self, input_shape):
        trace_count, sample_count = input_shape
        return (trace_count, sample_count // 2 + 1)

    def process(self, batch: Batch) -> Batch:
        sampling_count = batch.traces.shape[-1]

        fft_traces = compute_fft_magnitudes(
            traces=batch.traces,
            sampling_count=sampling_count,
            window_type=self.window_type,
        )

        return Batch(
            indices=batch.indices,
            traces=fft_traces,
            metadata=batch.metadata,
        )
