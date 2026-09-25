from scapyter.application.trace.trace_batch_service import TraceBatchService
from scapyter.domain.analysis.correlation.trace_statistics_accumulator import (
    TraceStatisticsAccumulator,
)
from scapyter.domain.analysis.correlation.value_objects.correlation_functions import (
    CorrelationFunction,
)
from scapyter.domain.value_object import (
    DataSource,
    CpaByteResult,
    TraceAndModeledLeakage,
)


class MultiCorrelationService:
    def __init__(
        self,
        trace_batch_service: TraceBatchService,
        data_source: DataSource,
    ):
        self._trace_batch_service = trace_batch_service
        self._data_source = data_source

    def run(
        self,
        correlation_functions: list[CorrelationFunction],
        batch_size: int = 50,
    ) -> list[CpaByteResult]:
        trace_statistics_accumulator = TraceStatisticsAccumulator()

        for batch in self._trace_batch_service.batches(batch_size):
            known_data = batch.metadata[self._data_source.value]

            # shared: updated once per batch, not once per byte
            trace_statistics_accumulator.update(batch.traces)

            for correlation_function in correlation_functions:
                modeled_leakages = correlation_function.leakage_model.calculate(
                    known_data
                )
                correlation_function.correlation.update(
                    TraceAndModeledLeakage(
                        traces=batch.traces,
                        modeled_leakage=modeled_leakages,
                    )
                )

        statistics = trace_statistics_accumulator.compute()

        return [
            CpaByteResult(
                byte_index=cf.leakage_model.byte_location,
                key_candidates=cf.leakage_model.key_byte_guesses,
                corr_matrix=cf.correlation.compute(statistics),
            )
            for cf in correlation_functions
        ]
