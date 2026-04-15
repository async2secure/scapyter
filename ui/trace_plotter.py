import matplotlib.pyplot as plt
import numpy as np


class TracePlotter:
    def __init__(self, repository):
        """
        :param repository: An instance of H5TraceRepository
        """
        self.repo = repository

    def plot_single(self, index, sample_range=slice(None), color="blue"):
        """Plots a single trace from the repository."""
        trace = self.repo.get_single_batch(index, sample_slice=sample_range)

        plt.figure(figsize=(12, 4))
        plt.plot(trace.trace, color=color, linewidth=0.7)
        plt.title(f"Trace {index}")
        plt.xlabel("Sample Index")
        plt.ylabel("Amplitude")
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.show()

    def plot_overlay(self, trace_range, sample_range=slice(None), alpha=0.5):
        """Overlays multiple traces to check for alignment or noise."""
        batch = self.repo.get_batch(trace_range, sample_slice=sample_range)

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
