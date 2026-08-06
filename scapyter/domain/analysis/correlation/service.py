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
    TraceAndModeledLeakage,
    DataSource,
    CpaByteResult,
)


class CorrelationService:
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
        self._trace_statics_accumulator = TraceStatisticsAccumulator()

    def run(self, batch_size=50) -> list[CpaByteResult]:
        trace_range = self._range_parameters.trace_range

        _, batch_range_list = get_progress_batch(
            batch_size=batch_size,
            progress_steps=trace_range.count,
            trace_range=trace_range,
        )

        for batch_range in tqdm(
            batch_range_list,
            desc="Processing batches",
            unit="batch",
        ):
            sample_range = self._range_parameters.sample_range

            batch = self._project_file_reader.get_batch(
                batch_range,
                sample_range=sample_range,
            )

            known_data = batch.metadata[self._data_source.value]

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
                self._trace_statics_accumulator.update(batch.traces)
                func.correlation.update(trace_and_modeled_leakage)
        statics = self._trace_statics_accumulator.compute()
        return [
            CpaByteResult(
                func.byte_location,
                func.key_byte_guesses,
                func.correlation.compute(statics),
            )
            for func in self._correlation_functions
        ]
