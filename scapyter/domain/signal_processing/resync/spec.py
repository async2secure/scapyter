import numpy as np

from scipy.signal import spectrogram, correlate, correlation_lags

from scapyter.domain.signal_processing.trace_processor import TraceProcessor
from scapyter.domain.value_object import Batch


class SpectrogramResyncProcessor(TraceProcessor):

    def __init__(
        self,
        reference_trace: np.ndarray,
        fs: float,
        nperseg: int = 1024,
        noverlap: int = 960,
        freq_limit: float | None = None,
    ):

        self.fs = fs
        self.nperseg = nperseg
        self.noverlap = noverlap
        self.freq_limit = freq_limit

        self.reference = reference_trace.astype(np.float32)

        self.reference_energy = self._spectrogram_energy(self.reference)

        self.dt_spec = self.reference_time_step()

    def process(self, batch: Batch) -> Batch:

        aligned = np.empty_like(batch.traces)

        for i, trace in enumerate(batch.traces):

            shift = self._find_shift(trace)

            aligned[i] = self._shift_trace(trace, shift)

        return Batch(
            indices=batch.indices,
            traces=aligned,
            metadata=batch.metadata,
        )

    # --------------------------------------------------------
    # Compute spectrogram energy
    # --------------------------------------------------------

    def _spectrogram_energy(self, trace):

        f, t, S = spectrogram(
            trace,
            fs=self.fs,
            window="hann",
            nperseg=self.nperseg,
            noverlap=self.noverlap,
            scaling="density",
        )

        if self.freq_limit is not None:

            mask = f <= self.freq_limit

            S = S[mask, :]

        energy = np.sum(S, axis=0)

        energy -= np.mean(energy)

        energy /= np.std(energy) + 1e-12

        return energy

    def reference_time_step(self):

        hop = self.nperseg - self.noverlap

        return hop / self.fs

    # --------------------------------------------------------
    # Find spectrogram shift
    # --------------------------------------------------------

    def _find_shift(self, trace):

        target_energy = self._spectrogram_energy(trace)

        corr = correlate(target_energy, self.reference_energy, mode="full")

        lags = correlation_lags(
            len(target_energy), len(self.reference_energy), mode="full"
        )

        best_lag = lags[np.argmax(corr)]

        # spectrogram bins -> seconds

        delay = best_lag * self.dt_spec

        # seconds -> samples

        shift = int(round(delay * self.fs))

        return shift

    # --------------------------------------------------------
    # Shift trace
    # --------------------------------------------------------

    def _shift_trace(self, trace, shift):

        out = np.empty_like(trace)

        if shift > 0:

            out[:-shift] = trace[shift:]

            out[-shift:] = trace[-1]

        elif shift < 0:

            shift = abs(shift)

            out[shift:] = trace[:-shift]

            out[:shift] = trace[0]

        else:

            out[:] = trace

        return out
