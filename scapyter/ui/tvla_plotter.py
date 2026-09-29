import numpy as np
import matplotlib.pyplot as plt

from scapyter.domain.tvla.value_objects import TvlaResult


class TvlaPlotter:
    def __init__(
        self,
        threshold: float = 4.5,
    ):
        self._threshold = threshold

    def plot(
        self,
        result: TvlaResult,
        *,
        show: bool = True,
        ax: plt.Axes | None = None,
    ) -> plt.Axes:
        if ax is None:
            _, ax = plt.subplots(figsize=(14, 5))

        samples = np.arange(
            result.sample_start,
            result.sample_end,
        )

        ax.plot(
            samples,
            result.t_scores,
            linewidth=0.8,
            label="Welch's t-score",
        )

        ax.axhline(
            self._threshold,
            color="red",
            linestyle="--",
            linewidth=1,
            label=f"+{self._threshold}",
        )

        ax.axhline(
            -self._threshold,
            color="red",
            linestyle="--",
            linewidth=1,
            label=f"-{self._threshold}",
        )

        ax.axhline(
            0,
            color="black",
            linewidth=0.5,
            alpha=0.5,
        )

        ax.set_xlabel("Sample")
        ax.set_ylabel("Welch's t-score")

        ax.set_title(f"TVLA ({result.trace_start}-{result.trace_end} traces)")

        ax.grid(alpha=0.3)
        ax.legend()

        if show:
            plt.show()

        return ax

    def plot_progression(
        self,
        results: list[TvlaResult],
        *,
        show: bool = True,
        ax: plt.Axes | None = None,
    ) -> plt.Axes:
        if not results:
            raise ValueError("Cannot plot an empty TVLA progression.")

        if ax is None:
            _, ax = plt.subplots(figsize=(10, 5))

        trace_counts = np.array([result.trace_end for result in results])

        max_t_scores = np.array([np.max(np.abs(result.t_scores)) for result in results])

        ax.plot(
            trace_counts,
            max_t_scores,
            marker="o",
            linewidth=1.5,
            markersize=4,
            label="Maximum |t|",
        )

        ax.axhline(
            self._threshold,
            color="red",
            linestyle="--",
            linewidth=1,
            label=f"Threshold ({self._threshold})",
        )

        ax.set_xlabel("Number of traces")
        ax.set_ylabel("Maximum |t-score|")
        ax.set_title("TVLA progression")

        ax.grid(alpha=0.3)
        ax.legend()

        if show:
            plt.show()

        return ax
