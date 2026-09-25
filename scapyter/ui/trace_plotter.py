import matplotlib.pyplot as plt
import numpy as np

from scapyter.domain.value_object import Range


class TracePlotter:
    def __init__(self, repository):
        """
        :param repository: An instance of H5TraceRepository
        """
        self.repo = repository

    def plot_single(self, index, sample_range: Range | None = None, color="blue"):
        """Plots a single trace from the repository."""
        batch = self.repo.get_single_batch(index, sample_range=sample_range)

        plt.figure(figsize=(12, 4))
        plt.plot(batch.trace, color=color, linewidth=0.7)
        plt.title(f"Trace {index}")
        plt.xlabel("Sample Index")
        plt.ylabel("Amplitude")
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.show()

    def plot_overlay(self, trace_range, sample_range: Range | None = None, alpha=0.5):
        """Overlays multiple traces to check for alignment or noise."""
        batch = self.repo.get_batch(trace_range, sample_range=sample_range)

        plt.figure(figsize=(12, 5))
        for i in range(len(batch.trace)):
            plt.plot(batch.trace[i], alpha=alpha, linewidth=0.5)

        plt.title(f"Overlay: Traces {trace_range.start} to {trace_range.end}")
        plt.xlabel("Sample Index")
        plt.ylabel("Amplitude")
        plt.show()

    def plot_statistics(self, trace_range, sample_range=slice(None)):
        """Plots the mean and standard deviation of a batch."""
        batch = self.repo.get_batch(trace_range, sample_slice=sample_range)

        mean_trace = np.mean(batch.trace, axis=0)
        std_trace = np.std(batch.trace, axis=0)

        plt.figure(figsize=(12, 5))
        plt.plot(mean_trace, label="Mean", color="black", linewidth=1)
        plt.fill_between(
            range(len(mean_trace)),
            mean_trace - std_trace,
            mean_trace + std_trace,
            color="red",
            alpha=0.2,
            label="Std Dev",
        )

        plt.title(f"Statistical Analysis (N={len(batch.trace)})")
        plt.legend()
        plt.show()

    def plot_fft_processed(
        self,
        index,
        fs,
        db=True,
    ):
        """
        Plot processed FFT magnitude trace.

        Assumes repository contains:
            abs(rFFT(trace))
            with DC bin already removed if desired.
        """

        batch = self.repo.get_single_batch(index)

        magnitude = batch.traces[0].astype(np.float64)

        if db:
            magnitude = 20 * np.log10(magnitude + 1e-12)

        # FFT output length
        fft_bins = len(magnitude)

        # If FFTProcessor removes DC bin:
        # bins correspond to frequencies 1..Nyquist
        freqs = np.linspace(
            fs / fft_bins,
            fs / 2,
            fft_bins,
        )

        plt.figure(figsize=(12, 4))

        plt.plot(
            freqs / 1e6,
            magnitude,
            linewidth=0.8,
        )

        plt.title(f"FFT Spectrum - Trace {index}")

        plt.xlabel("Frequency (MHz)")

        plt.ylabel("Magnitude (dB)" if db else "Magnitude")

        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.show()
