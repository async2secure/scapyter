import numpy as np
from tqdm import tqdm

from scapyter.domain.analysis.tvla.calculator import TvlaCalculator
from scapyter.domain.progress_range.progress_range import get_progress_batch
from scapyter.domain.repository.project_file_reader import ProjectFileReader
from scapyter.domain.tvla.value_objects import TvlaResult
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
        self._acc_even = np.zeros(
            sample_count,
            dtype=np.double,
        )
        self._acc_even_sq = np.zeros(
            sample_count,
            dtype=np.double,
        )

        self._acc_odd = np.zeros(
            sample_count,
            dtype=np.double,
        )
        self._acc_odd_sq = np.zeros(
            sample_count,
            dtype=np.double,
        )

        self._count_even = 0
        self._count_odd = 0

    def calculate(
        self,
        batch_size: int = 50,
    ) -> TvlaResult:
        """
        Calculates TVLA over the configured trace range.

        Returns:
            A TvlaResult containing the T-scores and
            the ranges used for the calculation.
        """
        self._reset()

        trace_range = self._range_parameters.trace_range

        self._update(
            trace_range,
            batch_size=batch_size,
        )

        return self._create_result(
            trace_start=trace_range.start,
            trace_end=trace_range.end,
        )

    def calculate_trace_progression(
        self,
        step_size: int,
        batch_size: int = 50,
    ) -> list[TvlaResult]:
        """
        Calculates TVLA incrementally over the configured
        trace range.

        For example, with a trace range of 0-500 and a
        step size of 100, returns results for:

            100, 200, 300, 400, 500 traces.

        Each TvlaResult contains the complete T-score
        waveform calculated using all traces accumulated
        up to that point.

        Returns:
            A list of TvlaResult objects ordered by
            increasing trace count.
        """
        if step_size <= 0:
            raise ValueError("step_size must be greater than 0")

        self._reset()

        trace_range = self._range_parameters.trace_range
        current_pos = trace_range.start

        results: list[TvlaResult] = []

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
            delta_range = Range(
                current_pos,
                end_val,
            )

            self._update(
                delta_range,
                batch_size=batch_size,
            )

            results.append(
                self._create_result(
                    trace_start=trace_range.start,
                    trace_end=end_val,
                )
            )

            current_pos = end_val

        return results

    def _update(
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
            progress_steps=(trace_range.end - trace_range.start),
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
            self._acc_even += np.sum(
                even_traces,
                axis=0,
            )
            self._acc_even_sq += np.sum(
                np.square(even_traces),
                axis=0,
            )
            self._count_even += len(even_traces)

            self._acc_odd += np.sum(
                odd_traces,
                axis=0,
            )
            self._acc_odd_sq += np.sum(
                np.square(odd_traces),
                axis=0,
            )
            self._count_odd += len(odd_traces)

    def _get_t_scores(self) -> np.ndarray:
        """
        Calculates the current Welch's T-test.
        """
        if self._count_even < 2 or self._count_odd < 2:
            raise ValueError(
                "Welch's t-test requires at least two traces "
                "in each group. "
                f"Got even={self._count_even}, "
                f"odd={self._count_odd}."
            )

        return TvlaCalculator.calculate_welch_t_test(
            self._acc_even,
            self._acc_even_sq,
            self._count_even,
            self._acc_odd,
            self._acc_odd_sq,
            self._count_odd,
        )

    def _create_result(
        self,
        trace_start: int,
        trace_end: int,
    ) -> TvlaResult:
        """
        Creates a snapshot of the current TVLA state.
        """
        sample_range = self._range_parameters.sample_range

        return TvlaResult(
            t_scores=self._get_t_scores().copy(),
            trace_start=trace_start,
            trace_end=trace_end,
            sample_start=sample_range.start,
            sample_end=sample_range.end,
        )

    def _reset(self) -> None:
        """
        Resets the accumulated TVLA state.
        """
        self._acc_even.fill(0)
        self._acc_even_sq.fill(0)
        self._acc_odd.fill(0)
        self._acc_odd_sq.fill(0)

        self._count_even = 0
        self._count_odd = 0
