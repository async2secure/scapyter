from collections.abc import Iterator

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
    RangeParameters,
    DataSource,
    TraceAndModeledLeakage,
    CpaByteResult,
)


class ProgressiveCorrelationService:

    def __init__(
        self,
        range_parameters: RangeParameters,
        project_file_reader: ProjectFileReader,
        data_source: DataSource,
        correlation_functions: list[CorrelationFunction],
    ):
        self._project_file_reader = project_file_reader
        self._range_parameters = range_parameters
        self._data_source = data_source
        self._correlation_functions = correlation_functions
        self._trace_statistics_accumulator = TraceStatisticsAccumulator()

    def run(
        self,
        batch_size: int = 50,
        progress_steps: int = 100,
    ) -> Iterator[ProgressiveCpaResult]:

        trace_range = self._range_parameters.trace_range

        progress_markers, batch_ranges = get_progress_batch(
            batch_size=batch_size,
            progress_steps=progress_steps,
            trace_range=trace_range,
        )

        progress_markers = set(progress_markers)

        processed_traces = 0

        for batch_range in batch_ranges:

            sample_range = self._range_parameters.sample_range

            batch = self._project_file_reader.get_batch(
                batch_range, sample_range=sample_range
            )

            known_data = batch.metadata[self._data_source.value]

            self._trace_statistics_accumulator.update(batch.traces)

            for func in self._correlation_functions:

                modeled_leakages = []

                for key_guess in func.key_byte_guesses:
                    modeled_leakage = func.leakage_model.calculate(
                        byte_location=func.byte_location,
                        known_data=known_data,
                        key_guess=key_guess,
                    )

                    modeled_leakages.append(modeled_leakage)

                trace_and_modeled_leakage = TraceAndModeledLeakage(
                    traces=batch.traces,
                    modeled_leakage=np.asarray(modeled_leakages).T,
                )

                func.correlation.update(trace_and_modeled_leakage)

            processed_traces += batch_range.count

            if processed_traces in progress_markers:
                yield self._build_result(processed_traces)

    def _build_result(
        self,
        processed_traces: int,
    ) -> ProgressiveCpaResult:

        statistics = self._trace_statistics_accumulator.compute()

        return ProgressiveCpaResult(
            processed_traces=processed_traces,
            byte_results=[
                CpaByteResult(
                    byte_index=func.byte_location,
                    key_candidates=func.key_byte_guesses,
                    corr_matrix=func.correlation.compute(statistics),
                )
                for func in self._correlation_functions
            ],
        )
