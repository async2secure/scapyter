import numpy as np
from tqdm import tqdm

from scapyter.domain.analysis.correlation.trace_statistics_accumulator import (
    TraceStatisticsAccumulator,
)
from scapyter.domain.analysis.correlation.value_objects.progressive_cpa_result import (
    ProgressiveCpaResult,
)
from scapyter.domain.progress_range.progress_range import get_progress_batch
from scapyter.domain.repository.project_file_reader import ProjectFileReader
from scapyter.domain.value_object import (
    CpaByteResult,
    TraceAndModeledLeakage,
    DataSource,
    RangeParameters,
)


class SumAbsCorrelationService:
    def __init__(
        self,
        project_file_reader: ProjectFileReader,
        data_source: DataSource,
    ):
        self._project_file_reader = project_file_reader
        self._data_source = data_source

    def run(
        self,
        range_parameters: RangeParameters,
        correlation_functions: list,
        batch_size: int = 50,
        step: int | None = None,
    ) -> CpaByteResult | list[ProgressiveCpaResult]:
        """
        step=None (default): returns the final CpaByteResult using all
            traces in `range_parameters.trace_range`.
        step=<int>: returns a list[ProgressiveCpaResult], one every `step`
            traces (cumulative), plus a final entry covering the full range
            if it doesn't already land exactly on a step boundary.
        """
        if not correlation_functions:
            raise ValueError("correlation_functions must not be empty")

        first_model = correlation_functions[0].leakage_model
        byte_location = first_model.byte_location
        for fn in correlation_functions:
            if fn.leakage_model.byte_location != byte_location:
                raise ValueError(
                    "All correlation functions must attack the same byte_location"
                )

        trace_statistics_accumulator = TraceStatisticsAccumulator()
        trace_range = range_parameters.trace_range

        _, batch_range_list = get_progress_batch(
            batch_size=batch_size,
            progress_steps=trace_range.count,
            trace_range=trace_range,
        )

        def _current_result() -> CpaByteResult:
            statistics = trace_statistics_accumulator.compute()
            combined = sum(
                np.abs(np.nan_to_num(fn.correlation.compute(statistics)))
                for fn in correlation_functions
            )
            return CpaByteResult(
                byte_index=byte_location,
                key_candidates=first_model.key_byte_guesses,
                corr_matrix=combined,
            )

        snapshots: list[ProgressiveCpaResult] = []
        next_step_target = step

        for batch_range in tqdm(
            batch_range_list,
            desc=f"CPA byte {byte_location}",
            unit="batch",
            total=len(batch_range_list),
        ):
            batch = self._project_file_reader.get_batch(
                batch_range,
                sample_range=range_parameters.sample_range,
            )
            known_data = batch.metadata[self._data_source.value]

            trace_statistics_accumulator.update(batch.traces)

            for fn in correlation_functions:
                fn.correlation.update(
                    TraceAndModeledLeakage(
                        traces=batch.traces,
                        modeled_leakage=fn.leakage_model.calculate(known_data),
                    )
                )

            if step is not None:
                while trace_statistics_accumulator.processed_traces >= next_step_target:
                    snapshots.append(
                        ProgressiveCpaResult(
                            processed_traces=next_step_target,
                            byte_result=_current_result(),
                        )
                    )
                    next_step_target += step

        final_result = _current_result()

        if step is None:
            return final_result

        total_traces = trace_statistics_accumulator.processed_traces
        if not snapshots or snapshots[-1].processed_traces != total_traces:
            snapshots.append(
                ProgressiveCpaResult(
                    processed_traces=total_traces, byte_result=final_result
                )
            )

        return snapshots
