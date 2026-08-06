from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np

from scapyter.domain.analysis.correlation.value_objects.progressive_cpa_result import (
    ProgressiveCpaResult,
)


class ProgressiveCpaPlotter:
    """
    Plot progressive CPA results.

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
    ) -> plt.Axes:
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

        processed_traces = [
            result.processed_traces
            for result in self._results
        ]

        byte_results = [
            next(
                b
                for b in result.byte_results
                if b.byte_index == byte_index
            )
            for result in self._results
        ]

        guesses = list(byte_results[0].key_candidates)

        correlation_history: dict[int, list[float]] = {
            guess: []
            for guess in guesses
        }

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

            for row, guess in enumerate(guesses):

                max_corr = np.max(np.abs(result.corr_matrix[row]))

                correlation_history[guess].append(max_corr)

        #
        # Find strongest guess at final iteration.
        #
        best_guess = max(
            correlation_history,
            key=lambda g: correlation_history[g][-1],
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

        if correct_key is not None or best_guess is not None:
            ax.legend()

        plt.tight_layout()

        if show:
            plt.show()

        return ax