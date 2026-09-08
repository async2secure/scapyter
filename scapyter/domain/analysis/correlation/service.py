import numpy as np
from tqdm import tqdm

from scapyter.domain.analysis.correlation.trace_statistics_accumulator import (
    TraceStatisticsAccumulator,
)
from scapyter.domain.analysis.correlation.value_objects.correlation_functions import (
    CorrelationFunction,
)
from scapyter.domain.progress_range.progress_range import get_progress_batch
from scapyter.domain.repository.project_file_reader import ProjectFileReader
from scapyter.domain.value_object import (
    RangeParameters,
    DataSource,
    CpaByteResult,
    TraceAndModeledLeakage,
)


class CorrelationService:
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
    ) -> CpaByteResult:
        trace_statistics_accumulator = TraceStatisticsAccumulator()

        trace_range = self._range_parameters.trace_range

        _, batch_range_list = get_progress_batch(
            batch_size=batch_size,
            progress_steps=trace_range.count,
            trace_range=trace_range,
        )

        for batch_range in tqdm(
            batch_range_list,
            desc=f"Byte {correlation_function.byte_location}",
            unit="batch",
        ):
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

        statistics = trace_statistics_accumulator.compute()

        return CpaByteResult(
            byte_index=correlation_function.byte_location,
            key_candidates=correlation_function.key_byte_guesses,
            corr_matrix=correlation_function.correlation.compute(statistics),
        )
