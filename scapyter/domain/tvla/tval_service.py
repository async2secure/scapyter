import numpy as np

from scapyter.domain.progress_range.progress_range import get_progress_batch
from scapyter.domain.repository.trace_repository import TraceRepository
from scapyter.domain.tvla.tvla_calculator import TvlaCalculator
from scapyter.domain.value_object import RangeParameters, Range


class TvlaService:

    def __init__(
        self, trace_repository: TraceRepository, range_parameters: RangeParameters
    ):
        self._trace_repository = trace_repository
        self._range_parameters = range_parameters
        sample_count = range_parameters.sample_count

        self._acc_even = np.zeros(sample_count, np.double)
        self._acc_even_sq = np.zeros(sample_count, np.double)
        self._acc_odd = np.zeros(sample_count, np.double)
        self._acc_odd_sq = np.zeros(sample_count, np.double)
        self._count_even = 0
        self._count_odd = 0

    def run(self, batch_size: int = 50):

        progress, batch_range_list = get_progress_batch(
            batch_size=batch_size,
            progress_steps=self._range_parameters.trace_count,
            trace_range=self._range_parameters.trace_range,
        )
        sample_range = self._range_parameters.trace_sample_range
        for batch_range in batch_range_list:

            batch = self._trace_repository.get_batch(
                batch_range, sample_slice=slice(sample_range.start, sample_range.end)
            )

            traces = batch.traces

            if batch_range.start % 2 == 0:
                even_traces = traces[::2]
                odd_traces = traces[1::2]
            else:
                even_traces = traces[1::2]
                odd_traces = traces[::2]

            self._acc_even += np.sum(even_traces, axis=0)
            self._acc_even_sq += np.sum(np.square(even_traces), axis=0)
            self._count_even += len(even_traces)

            self._acc_odd += np.sum(odd_traces, axis=0)
            self._acc_odd_sq += np.sum(np.square(odd_traces), axis=0)
            self._count_odd += len(odd_traces)

        return TvlaCalculator.calculate_welch_t_test(
            self._acc_even,
            self._acc_even_sq,
            self._count_even,
            self._acc_odd,
            self._acc_odd_sq,
            self._count_odd,
        )
