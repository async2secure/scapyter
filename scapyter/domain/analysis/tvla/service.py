import numpy as np
from tqdm import tqdm

from scapyter.domain.progress_range.progress_range import get_progress_batch
from scapyter.domain.repository.project_file_reader import ProjectFileReader
from scapyter.domain.analysis.tvla.calculator import TvlaCalculator
from scapyter.domain.value_object import Range, RangeParameters


class TvlaService:
    def __init__(
        self,
        project_file_reader: ProjectFileReader,
        range_parameters: RangeParameters,
    ):
        self._project_file_reader = project_file_reader
        self._range_parameters = range_parameters

        sample_count = range_parameters.sample_count

        # Running sums for Welch's T-Test
        self._acc_even = np.zeros(sample_count, dtype=np.double)
        self._acc_even_sq = np.zeros(sample_count, dtype=np.double)

        self._acc_odd = np.zeros(sample_count, dtype=np.double)
        self._acc_odd_sq = np.zeros(sample_count, dtype=np.double)

        self._count_even = 0
        self._count_odd = 0

    def update(
        self,
        trace_range: Range,
        batch_size: int = 50,
    ) -> None:
        """
        Processes a range of traces and adds them to the
        accumulated TVLA state.

        The accumulated state is preserved between calls.
        """
        _, batch_range_list = get_progress_batch(
            batch_size=batch_size,
            progress_steps=trace_range.end - trace_range.start,
            trace_range=trace_range,
        )

        sample_range = self._range_parameters.sample_range

        for batch_range in batch_range_list:
            batch = self._project_file_reader.get_batch(
                batch_range,
                sample_range=Range(
                    sample_range.start,
                    sample_range.end,
                ),
            )

            traces = batch.traces

            # Maintain parity based on the absolute trace index.
            if batch_range.start % 2 == 0:
                even_traces = traces[::2]
                odd_traces = traces[1::2]
            else:
                even_traces = traces[1::2]
                odd_traces = traces[::2]

            # Accumulate sums and squared sums.
            self._acc_even += np.sum(even_traces, axis=0)
            self._acc_even_sq += np.sum(
                np.square(even_traces),
                axis=0,
            )
            self._count_even += len(even_traces)

            self._acc_odd += np.sum(odd_traces, axis=0)
            self._acc_odd_sq += np.sum(
                np.square(odd_traces),
                axis=0,
            )
            self._count_odd += len(odd_traces)

    def get_results(self) -> np.ndarray:
        """
        Calculates the current Welch's T-test from
        the accumulated traces.
        """
        return TvlaCalculator.calculate_welch_t_test(
            self._acc_even,
            self._acc_even_sq,
            self._count_even,
            self._acc_odd,
            self._acc_odd_sq,
            self._count_odd,
        )

    def run_max_t(self) -> float:
        """
        Returns the maximum absolute T-score from
        the currently accumulated traces.
        """
        t_scores = self.get_results()
        return float(np.max(np.abs(t_scores)))

    def run(
        self,
        step_size: int,
        batch_size: int = 50,
    ) -> dict[int, float]:
        """
        Runs TVLA incrementally over the configured trace range.

        For example, with a trace range of 0-500 and a step size
        of 50, TVLA is evaluated at:

            50, 100, 150, ..., 500

        Returns:
            A dictionary mapping the number of accumulated traces
            to the maximum absolute T-score.
        """
        if step_size <= 0:
            raise ValueError("step_size must be greater than 0")

        trace_range = self._range_parameters.trace_range
        current_pos = trace_range.start

        results: dict[int, float] = {}

        steps = range(
            current_pos + step_size,
            trace_range.end + 1,
            step_size,
        )

        for end_val in tqdm(
            steps,
            total=len(steps),
            desc="Running TVLA",
            unit="step",
        ):
            # Only process the new traces since the previous step.
            delta_range = Range(
                current_pos,
                end_val,
            )

            self.update(
                delta_range,
                batch_size=batch_size,
            )

            # Calculate TVLA using everything accumulated so far.
            results[end_val] = self.run_max_t()

            current_pos = end_val

        return results
