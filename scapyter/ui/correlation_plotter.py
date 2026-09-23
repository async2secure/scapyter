import matplotlib.pyplot as plt
import numpy as np

from scapyter.domain.value_object import CpaByteResult


class CorrelationPlotter:
    def __init__(self, results: list[CpaByteResult]):
        """
        :param results: list of CpaByteResult
        """
        self.results = results

    def plot_correlation_vs_samples(self, byte_index: int, known_key: int | None = None):
        # Find result object
        result = next(
            (r for r in self.results if r.byte_index == byte_index),
            None,
        )

        if result is None:
            print(f"Error: No results found for Byte {byte_index}")
            return

        corr_matrix = result.corr_matrix
        key_candidates = result.key_candidates

        plt.figure(figsize=(12, 6))

        samples = corr_matrix.shape[1]
        x_axis = np.arange(samples)

        # Envelopes
        highest_envelope = np.max(corr_matrix, axis=0)
        lowest_envelope = np.min(corr_matrix, axis=0)

        # Best hypothesis index
        best_idx = int(np.argmax(np.max(np.abs(corr_matrix), axis=1)))

        # Map best hypothesis index to actual key value
        best_key_value = key_candidates.values[best_idx]

        # Plot envelopes
        plt.plot(
            x_axis,
            highest_envelope,
            color="yellow",
            label="Max Envelope",
            linewidth=1,
            alpha=0.7,
        )

        plt.plot(
            x_axis,
            lowest_envelope,
            color="blue",
            label="Min Envelope",
            linewidth=1,
            alpha=0.7,
        )

        # Plot best candidate trace
        plt.plot(
            x_axis,
            corr_matrix[best_idx],
            color="red",
            linestyle="--",
            label=f"Best Candidate ({best_key_value:02X})",
            linewidth=1.5,
        )

        # Plot known key trace if provided
        if known_key is not None:
            known_indices = np.where(np.asarray(key_candidates.values) == known_key)[0]

            if len(known_indices) == 0:
                print(
                    f"Warning: Known key byte {known_key:02X} "
                    "not found in key candidates"
                )
            else:
                known_idx = int(known_indices[0])

                plt.plot(
                    x_axis,
                    corr_matrix[known_idx],
                    color="green",
                    linestyle="-",
                    label=f"Known Key ({known_key:02X})",
                    linewidth=2,
                )

        plt.title(f"CPA Correlation Analysis: Byte {byte_index:02d}")
        plt.xlabel("Sample Point")
        plt.ylabel("Correlation Coefficient")
        plt.legend(loc="upper right")
        plt.grid(True, alpha=0.2)
        plt.axhline(0, color="black", lw=1, alpha=0.3)
        plt.tight_layout()
        plt.show()

    def plot_correlation_vs_keys(self, byte_index: int, known_key: int | None = None):
        """
        Plots the peak correlation coefficient against each key byte candidate.
        X-axis: Key Byte Values
        Y-axis: Maximum Absolute Correlation Coefficient
        """
        # Find result object
        result = next(
            (r for r in self.results if r.byte_index == byte_index),
            None,
        )

        if result is None:
            print(f"Error: No results found for Byte {byte_index}")
            return

        corr_matrix = result.corr_matrix
        key_candidates = result.key_candidates

        plt.figure(figsize=(12, 6))

        # Calculate max absolute correlation for each key candidate across all sample points
        max_corrs = np.max(np.abs(corr_matrix), axis=1)
        keys = np.asarray(key_candidates.values)

        # Sort by key value for an ordered X-axis
        sort_idx = np.argsort(keys)
        sorted_keys = keys[sort_idx]
        sorted_corrs = max_corrs[sort_idx]

        # Plot all key candidates as a bar chart
        plt.bar(
            sorted_keys,
            sorted_corrs,
            color="skyblue",
            edgecolor="none",
            alpha=0.6,
            width=1.0,
            label="Key Candidates",
        )

        # Highlight best candidate
        best_idx = int(np.argmax(max_corrs))
        best_key = keys[best_idx]
        plt.bar(
            best_key,
            max_corrs[best_idx],
            color="red",
            width=1.0,
            label=f"Best Candidate ({best_key:02X})",
        )

        # Highlight known key if provided
        if known_key is not None:
            if known_key in keys:
                known_idx = np.where(keys == known_key)[0][0]
                plt.bar(
                    known_key,
                    max_corrs[known_idx],
                    color="green",
                    width=1.0,
                    alpha=0.8,
                    label=f"Known Key ({known_key:02X})",
                )
            else:
                print(
                    f"Warning: Known key byte {known_key:02X} "
                    "not found in key candidates"
                )

        plt.title(f"CPA Attack: Peak Correlation vs Key Bytes (Byte {byte_index:02d})")
        plt.xlabel("Key Byte Candidate (Hex)")
        plt.ylabel("Max Absolute Correlation Coefficient")
        plt.legend(loc="upper right")
        plt.grid(True, alpha=0.2, axis="y")

        # Set X-ticks to show hex values neatly if it spans 0-255
        plt.xlim(-1, 256)

        plt.tight_layout()
        plt.show()
