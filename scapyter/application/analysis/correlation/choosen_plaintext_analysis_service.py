import numpy as np
from tqdm import tqdm

from scapyter.domain.analysis.correlation.cpa import CpaCorrelation
from scapyter.domain.analysis.correlation.trace_statistics_accumulator import (
    TraceStatisticsAccumulator,
)
from scapyter.domain.analysis.correlation.value_objects.correlation_functions import (
    CorrelationFunction,
)
from scapyter.domain.analysis.correlation.value_objects.progressive_cpa_result import (
    ProgressiveCpaResult,
)
from scapyter.domain.leakage_functions.leakage_functions import Monobit
from scapyter.domain.leakage_model.leakage_model import LeakageModel
from scapyter.domain.progress_range.progress_range import get_progress_batch
from scapyter.domain.repository.project_file_reader import ProjectFileReader
from scapyter.domain.targets.targets import ChosenPlaintextRegisterDeltaByte
from scapyter.domain.value_object import (
    CpaByteResult,
    DataSource,
    KeyByteGuesses,
    RangeParameters,
)
from scapyter.domain.value_object import (
    TraceAndModeledLeakage,
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


"""
ChosenPlaintextRegisterAttack, driven by an injected RangeParameters.

The attack no longer knows how the trace file is laid out: whichever traces
and samples `range_parameters` selects are the ones it correlates. Choosing
the right block for a byte is the caller's job.

Not run against your real classes (LeakageModel, Monobit,
ChosenPlaintextRegisterDeltaByte are not available to me). Adjust the
imports below to your package layout.
"""


class ChosenPlaintextRegisterAttack:
    """Runs the 32-CPA register attack over the traces selected by the
    `range_parameters` passed to each call."""

    def __init__(
        self,
        project_file_reader: ProjectFileReader,
        data_source: DataSource = DataSource.PLAINTEXT,
    ):
        # The service holds no state between runs (accumulators are created
        # inside run()), so one instance can serve every attack_* call.
        self._service = SumAbsCorrelationService(
            project_file_reader=project_file_reader,
            data_source=data_source,
        )

    @staticmethod
    def build_functions(byte_location: int) -> list["CorrelationFunction"]:
        """32 CPAs: 4 output bytes of Delta' x 8 single bits."""
        return [
            CorrelationFunction(
                correlation=CpaCorrelation(),
                leakage_model=LeakageModel(
                    target=ChosenPlaintextRegisterDeltaByte(output_byte),
                    leakage_function=Monobit(bit),
                    byte_location=byte_location,
                    key_byte_guesses=KeyByteGuesses.from_full256_range(),
                ),
            )
            for output_byte in range(4)
            for bit in range(8)
        ]

    # ------------------------------------------------------------------
    # Final-result API
    # ------------------------------------------------------------------

    def attack_byte(
        self,
        range_parameters: RangeParameters,
        byte_location: int,
        batch_size: int = 50,
    ) -> CpaByteResult:
        return self._service.run(
            range_parameters,
            self.build_functions(byte_location),
            batch_size=batch_size,
        )

    def attack(
        self,
        range_parameters: RangeParameters,
        byte_locations: list[int],
        batch_size: int = 50,
    ) -> list[CpaByteResult]:
        return [
            self.attack_byte(range_parameters, b, batch_size) for b in byte_locations
        ]

    # ------------------------------------------------------------------
    # Progressive (MTD) API
    # ------------------------------------------------------------------

    def attack_byte_progressive(
        self,
        range_parameters: RangeParameters,
        byte_location: int,
        step: int,
        batch_size: int = 50,
    ) -> list[ProgressiveCpaResult]:
        """One pass over the selected traces, with a cumulative snapshot every
        `step` traces."""
        if step % batch_size != 0:
            raise ValueError(
                f"step ({step}) must be a multiple of batch_size ({batch_size}) "
                "so snapshots are evenly spaced"
            )
        return self._service.run(
            range_parameters,
            self.build_functions(byte_location),
            batch_size=batch_size,
            step=step,
        )

    def attack_progressive(
        self,
        range_parameters: RangeParameters,
        byte_locations: list[int],
        step: int,
        batch_size: int = 50,
    ) -> list[list[ProgressiveCpaResult]]:
        """One snapshot list per byte, in the order of `byte_locations`
        (mirrors attack()). The byte is identified by
        snapshots[0].byte_result.byte_index."""
        return [
            self.attack_byte_progressive(range_parameters, b, step, batch_size)
            for b in byte_locations
        ]
