from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np

from scapyter.domain.analysis.correlation.value_objects.progressive_cpa_result import (
    ProgressiveCpaResult,
)


class ProgressiveCpaPlotter:
    """
    Plot progressive CPA results.

    Each ProgressiveCpaResult represents one byte at one trace count.

    Each line represents one key guess. The y-axis is the maximum absolute
    correlation over all samples after each progressive update.

    The correct key (if supplied) is highlighted in red.
    The strongest key at the final iteration is highlighted in blue.
    All remaining guesses are shown in light grey.
    """

    def __init__(self, results: list[ProgressiveCpaResult]):
        if not results:
            raise ValueError("results cannot be empty")

        self._results = results

    def plot_convergence(
        self,
        *,
        byte_index: int,
        correct_key: int | None = None,
        ax: plt.Axes | None = None,
        show: bool = True,
    ) -> None:
        """
        Plot maximum absolute correlation versus processed traces.

        Parameters
        ----------
        byte_index
            AES byte to visualise.

        correct_key
            Optional correct key byte (0-255). If provided it is highlighted.

        ax
            Existing matplotlib axis.

        show
            Call plt.show() automatically.

        Returns
        -------
        matplotlib.axes.Axes
        """

        if ax is None:
            _, ax = plt.subplots(figsize=(10, 6))

        # Only use results belonging to the requested byte.
        byte_results = [
            result
            for result in self._results
            if result.byte_result.byte_index == byte_index
        ]

        if not byte_results:
            raise ValueError(f"No progressive results found for byte {byte_index}")

        # Sort by number of processed traces.
        byte_results.sort(key=lambda result: result.processed_traces)

        processed_traces = [result.processed_traces for result in byte_results]

        first_byte_result = byte_results[0].byte_result

        guesses = list(first_byte_result.key_candidates)

        correlation_history: dict[int, list[float]] = {guess: [] for guess in guesses}

        #
        # Compute:
        #
        # guess ->
        # [
        #   max_corr_after_50_traces,
        #   max_corr_after_100_traces,
        #   ...
        # ]
        #
        for result in byte_results:
            byte_result = result.byte_result

            for row, guess in enumerate(guesses):
                max_corr = np.max(np.abs(byte_result.corr_matrix[row]))

                correlation_history[guess].append(max_corr)

        #
        # Find strongest guess at final iteration.
        #
        best_guess = max(
            correlation_history,
            key=lambda guess: correlation_history[guess][-1],
        )

        #
        # Plot all guesses.
        #
        for guess in guesses:
            history = correlation_history[guess]

            if guess == correct_key:
                ax.plot(
                    processed_traces,
                    history,
                    color="tab:red",
                    linewidth=3,
                    label=f"Correct key (0x{guess:02X})",
                    zorder=3,
                )

            elif guess == best_guess:
                ax.plot(
                    processed_traces,
                    history,
                    color="tab:blue",
                    linewidth=2,
                    label=f"Best guess (0x{guess:02X})",
                    zorder=2,
                )

            else:
                ax.plot(
                    processed_traces,
                    history,
                    color="0.75",
                    linewidth=0.8,
                    alpha=0.45,
                    zorder=1,
                )

        ax.set_title(f"Progressive CPA Convergence (Byte {byte_index})")
        ax.set_xlabel("Processed traces")
        ax.set_ylabel("Maximum |Correlation|")

        ax.grid(True, linestyle="--", alpha=0.3)

        ax.legend()

        plt.tight_layout()

        if show:
            plt.show()

        return ax
