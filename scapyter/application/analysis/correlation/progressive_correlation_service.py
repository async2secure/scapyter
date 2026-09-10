from typing import Iterator

import numpy as np

from scapyter.domain.analysis.correlation.trace_statistics_accumulator import (
    TraceStatisticsAccumulator,
)
from scapyter.domain.analysis.correlation.value_objects.correlation_functions import (
    CorrelationFunction,
)
from scapyter.domain.analysis.correlation.value_objects.progressive_cpa_result import (
    ProgressiveCpaResult,
)
from scapyter.domain.progress_range.progress_range import get_progress_batch
from scapyter.domain.repository.project_file_reader import ProjectFileReader
from scapyter.domain.value_object import (
    TraceAndModeledLeakage,
    CpaByteResult,
    RangeParameters,
    DataSource,
)


class ProgressiveCorrelationService:

    def __init__(
        self,
        range_parameters: RangeParameters,
        project_file_reader: ProjectFileReader,
        data_source: DataSource,
    ):
        self._project_file_reader = project_file_reader
        self._range_parameters = range_parameters
        self._data_source = data_source

    def run(
        self,
        correlation_function: CorrelationFunction,
        batch_size: int = 50,
        progress_steps: int = 100,
    ) -> Iterator[ProgressiveCpaResult]:

        trace_statistics_accumulator = TraceStatisticsAccumulator()

        trace_range = self._range_parameters.trace_range

        progress_markers, batch_ranges = get_progress_batch(
            batch_size=batch_size,
            progress_steps=progress_steps,
            trace_range=trace_range,
        )

        progress_markers = set(progress_markers)

        processed_traces = 0

        for batch_range in batch_ranges:
            batch = self._project_file_reader.get_batch(
                batch_range,
                sample_range=self._range_parameters.sample_range,
            )

            known_data = batch.metadata[self._data_source.value]

            trace_statistics_accumulator.update(batch.traces)

            modeled_leakages = [
                correlation_function.leakage_model.calculate(
                    byte_location=correlation_function.byte_location,
                    known_data=known_data,
                    key_guess=key_guess,
                )
                for key_guess in correlation_function.key_byte_guesses
            ]

            correlation_function.correlation.update(
                TraceAndModeledLeakage(
                    traces=batch.traces,
                    modeled_leakage=np.asarray(modeled_leakages).T,
                )
            )

            processed_traces += batch_range.count

            if processed_traces in progress_markers:
                yield self._build_result(
                    processed_traces,
                    correlation_function,
                    trace_statistics_accumulator,
                )

    @staticmethod
    def _build_result(
        processed_traces: int,
        correlation_function: CorrelationFunction,
        trace_statistics_accumulator: TraceStatisticsAccumulator,
    ) -> ProgressiveCpaResult:

        statistics = trace_statistics_accumulator.compute()

        return ProgressiveCpaResult(
            processed_traces=processed_traces,
            byte_result=CpaByteResult(
                byte_index=correlation_function.byte_location,
                key_candidates=correlation_function.key_byte_guesses,
                corr_matrix=correlation_function.correlation.compute(statistics),
            ),
        )
