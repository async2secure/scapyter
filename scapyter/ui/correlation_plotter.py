import matplotlib.pyplot as plt
import numpy as np

from scapyter.domain.value_object import CpaByteResult


class CorrelationPlotter:
    def __init__(self, results: list[CpaByteResult]):
        """
        :param results: list of CpaByteResult
        """
        self.results = results

    def plot(self, byte_index: int):
        # find result object
        result = next((r for r in self.results if r.byte_index == byte_index), None)

        if result is None:
            print(f"Error: No results found for Byte {byte_index}")
            return

        corr_matrix = result.corr_matrix
        key_candidates = result.key_candidates

        plt.figure(figsize=(12, 6))

        samples = corr_matrix.shape[1]
        x_axis = np.arange(samples)

        # envelopes
        highest_envelope = np.max(corr_matrix, axis=0)
        lowest_envelope = np.min(corr_matrix, axis=0)

        # best hypothesis index
        best_idx = int(np.argmax(np.max(np.abs(corr_matrix), axis=1)))

        # map to actual key value
        best_key_value = key_candidates.values[best_idx]

        # plot envelopes
        plt.plot(
            x_axis,
            highest_envelope,
            color="red",
            label="Max Envelope",
            linewidth=1,
            alpha=0.7,
        )

        plt.plot(
            x_axis,
            lowest_envelope,
            color="black",
            label="Min Envelope",
            linewidth=1,
            alpha=0.7,
        )

        # plot best candidate trace
        plt.plot(
            x_axis,
            corr_matrix[best_idx],
            color="blue",
            linestyle="--",
            label=f"Best Candidate ({best_key_value:02X})",
            linewidth=1.5,
        )

        plt.title(f"CPA Correlation Analysis: Byte {byte_index:02d}")
        plt.xlabel("Sample Point")
        plt.ylabel("Correlation Coefficient")
        plt.legend(loc="upper right")
        plt.grid(True, alpha=0.2)
        plt.axhline(0, color="black", lw=1, alpha=0.3)
        plt.tight_layout()
        plt.show()
