import numpy as np

from scapyter.domain.progress_range.progress_range import get_progress_batch
from scapyter.domain.repository.project_file_reader import ProjectFileReader
from scapyter.domain.analysis.tvla.calculator import TvlaCalculator
from scapyter.domain.value_object import Range, RangeParameters


class TvlaService:
    def __init__(
        self, project_file_reader: ProjectFileReader, range_parameters: RangeParameters
    ):
        self._project_file_reader = project_file_reader
        self._range_parameters = range_parameters
        sample_count = range_parameters.sample_count

        # Running sums for Welch's T-Test
        self._acc_even = np.zeros(sample_count, np.double)
        self._acc_even_sq = np.zeros(sample_count, np.double)
        self._acc_odd = np.zeros(sample_count, np.double)
        self._acc_odd_sq = np.zeros(sample_count, np.double)
        self._count_even = 0
        self._count_odd = 0

    def update(self, trace_range: Range, batch_size: int = 50):
        """
        Processes a specific range of traces and adds them to the existing state.
        """
        # We reuse the logic to get batch ranges for the specific sub-range
        _, batch_range_list = get_progress_batch(
            batch_size=batch_size,
            progress_steps=(trace_range.end - trace_range.start),
            trace_range=trace_range,
        )

        sample_range = self._range_parameters.sample_range

        for batch_range in batch_range_list:
            batch = self._project_file_reader.get_batch(
                batch_range, sample_slice=slice(sample_range.start, sample_range.end)
            )
            traces = batch.traces

            # Maintain the parity logic based on the absolute index in the repo
            if batch_range.start % 2 == 0:
                even_traces = traces[::2]
                odd_traces = traces[1::2]
            else:
                even_traces = traces[1::2]
                odd_traces = traces[::2]

            # Update accumulators
            self._acc_even += np.sum(even_traces, axis=0)
            self._acc_even_sq += np.sum(np.square(even_traces), axis=0)
            self._count_even += len(even_traces)

            self._acc_odd += np.sum(odd_traces, axis=0)
            self._acc_odd_sq += np.sum(np.square(odd_traces), axis=0)
            self._count_odd += len(odd_traces)

    def get_results(self):
        """Calculates the current T-test based on accumulated data."""
        return TvlaCalculator.calculate_welch_t_test(
            self._acc_even,
            self._acc_even_sq,
            self._count_even,
            self._acc_odd,
            self._acc_odd_sq,
            self._count_odd,
        )

    def run_max_t(self) -> float:
        """Returns the max T-score of the current state."""
        t_scores = self.get_results()
        return np.max(np.abs(t_scores))
