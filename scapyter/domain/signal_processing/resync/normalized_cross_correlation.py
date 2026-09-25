import numpy as np
from scipy.signal import fftconvolve

from scapyter.domain.signal_processing.trace_processor import TraceProcessor
from scapyter.domain.value_object import Batch, Range


class NCCResyncProcessor(TraceProcessor):

    def __init__(
        self,
        reference_trace: np.ndarray,
        reference_range: Range,
        search_range: Range,
        pad_mode: str = "edge",
    ):
        self.reference_range = reference_range
        self.search_range = search_range
        self.pad_mode = pad_mode

        reference = reference_trace[reference_range.start : reference_range.end].astype(
            np.float32
        )

        # Normalize reference once
        reference -= reference.mean()
        reference /= reference.std() + 1e-12

        self.reference = reference
        self.window_size = len(reference)

    def process(self, batch: Batch) -> Batch:

        aligned = np.empty_like(batch.traces)

        for i, trace in enumerate(batch.traces):

            best_start = self._find_best_match(trace)

            shift = best_start - self.reference_range.start

            aligned[i] = self._shift_trace(trace, shift)

        return Batch(
            indices=batch.indices,
            traces=aligned,
            metadata=batch.metadata,
        )

    def _find_best_match(self, trace):

        search = trace[self.search_range.start : self.search_range.end].astype(
            np.float32
        )

        n = self.window_size

        #
        # FFT cross-correlation
        #
        corr = fftconvolve(
            search,
            self.reference[::-1],
            mode="valid",
        )

        #
        # Sliding mean
        #
        csum = np.concatenate(([0.0], np.cumsum(search)))
        csum2 = np.concatenate(([0.0], np.cumsum(search * search)))

        sum_x = csum[n:] - csum[:-n]
        sum_x2 = csum2[n:] - csum2[:-n]

        mean = sum_x / n

        variance = sum_x2 / n - mean**2
        variance = np.maximum(variance, 1e-12)

        std = np.sqrt(variance)

        #
        # Normalize
        #
        ncc = corr / (std * n)

        best_offset = np.argmax(ncc)

        return self.search_range.start + best_offset

    def _shift_trace(
        self,
        trace: np.ndarray,
        shift: int,
    ):

        if shift == 0:
            return trace.copy()

        out = np.empty_like(trace)

        if shift > 0:

            out[:-shift] = trace[shift:]

            if self.pad_mode == "edge":
                out[-shift:] = trace[-1]
            else:
                out[-shift:] = 0

        else:

            shift = -shift

            out[shift:] = trace[:-shift]

            if self.pad_mode == "edge":
                out[:shift] = trace[0]
            else:
                out[:shift] = 0

        return out
