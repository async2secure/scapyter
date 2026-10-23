"""
ChosenRegisterAttack: one class for both the chosen-plaintext and chosen-ciphertext 32-CPA
register attack (4 output bytes of Delta' x Monobit(0..7), combined via the sum-|rho| service).
Plaintext vs ciphertext is just a choice of DataSource and default target class -- the attack
logic and result handling are identical either way. Which target to use is fully pluggable via
target_factory, so this also covers deeper targets such as Round2RegisterDeltaByte (which needs
extra fixed state beyond output_byte).

The only public entry points are attack_groups() and attack_groups_progressive(): a single byte
against a single range is just a one-group call.

SHARED READS: within one AttackGroup, every byte_location is attacked from the SAME
range_parameters, so this reads each batch of traces ONCE and updates every byte's 32
correlation functions from that one read, rather than re-reading the same traces once per byte
(which is what calling SumAbsCorrelationService.run() once per byte would do). This is built
directly on the domain-layer primitives (TraceStatisticsAccumulator, CpaCorrelation,
get_progress_batch) rather than on SumAbsCorrelationService, because that service's run() takes
one byte_location's functions per call and so can't itself share a read across bytes.
"""

from typing import Callable

import numpy as np
from tqdm import tqdm

from scapyter.application.analysis.models.attack_group import AttackGroup
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
from scapyter.domain.targets.targets import (
    ChosenCiphertextDeltaByte,
    ChosenPlaintextRegisterDeltaByte,
    Target,
)
from scapyter.domain.value_object import (
    CpaByteResult,
    DataSource,
    KeyByteGuesses,
    TraceAndModeledLeakage,
)


class ChosenRegisterAttack:
    """Runs the 32-CPA register attack over the traces selected by each AttackGroup's
    range_parameters. One instance targets one key (one data_source, one target_factory);
    make a second instance for a different round or direction."""

    def __init__(
        self,
        project_file_reader: ProjectFileReader,
        data_source: DataSource,
        target_factory: Callable[[int], Target],
    ):
        self._project_file_reader = project_file_reader
        self._data_source = data_source
        # target_factory(output_byte) -> Target, called once per output byte (0..3) inside
        # build_functions. For a target that needs extra fixed state (e.g.
        # Round2RegisterDeltaByte's round_key_0), bind it ahead of time:
        #     ChosenRegisterAttack(
        #         trace_repo, DataSource.PLAINTEXT,
        #         target_factory=lambda ob: Round2RegisterDeltaByte(ob, round_key_0=recovered_k0),
        #     )
        self._target_factory = target_factory

    # ------------------------------------------------------------------
    # Convenience constructors for the two common, round-1 cases
    # ------------------------------------------------------------------

    @classmethod
    def chosen_plaintext(
        cls,
        project_file_reader: ProjectFileReader,
        target_factory: Callable[[int], Target] = ChosenPlaintextRegisterDeltaByte,
    ) -> "ChosenRegisterAttack":
        return cls(project_file_reader, DataSource.PLAINTEXT, target_factory)

    @classmethod
    def chosen_ciphertext(
        cls,
        project_file_reader: ProjectFileReader,
        target_factory: Callable[[int], Target] = ChosenCiphertextDeltaByte,
    ) -> "ChosenRegisterAttack":
        return cls(project_file_reader, DataSource.CIPHERTEXT, target_factory)

    # ------------------------------------------------------------------

    def build_functions(self, byte_location: int) -> list["CorrelationFunction"]:
        """32 CPAs: 4 output bytes of Delta' x 8 single bits."""
        return [
            CorrelationFunction(
                correlation=CpaCorrelation(),
                leakage_model=LeakageModel(
                    target=self._target_factory(output_byte),
                    leakage_function=Monobit(bit),
                    byte_location=byte_location,
                    key_byte_guesses=KeyByteGuesses.from_full256_range(),
                ),
            )
            for output_byte in range(4)
            for bit in range(8)
        ]

    @staticmethod
    def _combine(
        functions: list["CorrelationFunction"], statistics, byte_location: int
    ) -> CpaByteResult:
        return CpaByteResult(
            byte_index=byte_location,
            key_candidates=functions[0].leakage_model.key_byte_guesses,
            corr_matrix=sum(
                np.abs(np.nan_to_num(fn.correlation.compute(statistics)))
                for fn in functions
            ),
        )

    def attack_groups(
        self,
        groups: list[AttackGroup],
        batch_size: int = 50,
    ) -> list[CpaByteResult]:
        """Run several (byte_locations, range_parameters) pairings in one call, e.g.:

            cra = ChosenRegisterAttack.chosen_ciphertext(trace_repo)
            results = cra.attack_groups(
                AttackGroup(byte_locations=[0, 4, 8, 12], range_parameters=range_parameters1),
                AttackGroup(byte_locations=[1, 5, 9, 13], range_parameters=range_parameters2),
                batch_size=500,
            )

        A single byte against a single range is just one group:
            cra.attack_groups(AttackGroup(byte_locations=0, range_parameters=rp))

        Within one group, every byte_location's traces are read ONCE (not once per byte) -- see
        the module docstring. Returns a single flat list of CpaByteResult, in the order the
        groups and their byte_locations were given.
        """
        results = []
        for group in groups:
            per_byte_functions = {
                b: self.build_functions(b) for b in group.byte_locations
            }
            all_functions = [fn for fns in per_byte_functions.values() for fn in fns]

            trace_statistics_accumulator = TraceStatisticsAccumulator()
            trace_range = group.range_parameters.trace_range
            _, batch_range_list = get_progress_batch(
                batch_size=batch_size,
                progress_steps=trace_range.count,
                trace_range=trace_range,
            )
            for batch_range in tqdm(
                batch_range_list,
                desc=f"Bytes {group.byte_locations} ({len(all_functions)} CPAs)",
                unit="batch",
            ):
                batch = self._project_file_reader.get_batch(
                    batch_range, sample_range=group.range_parameters.sample_range
                )
                known_data = batch.metadata[self._data_source.value]
                trace_statistics_accumulator.update(batch.traces)
                for fn in all_functions:
                    fn.correlation.update(
                        TraceAndModeledLeakage(
                            traces=batch.traces,
                            modeled_leakage=fn.leakage_model.calculate(known_data),
                        )
                    )

            statistics = trace_statistics_accumulator.compute()
            results.extend(
                self._combine(fns, statistics, b)
                for b, fns in per_byte_functions.items()
            )
        return results

    def attack_groups_progressive(
        self,
        groups: list[AttackGroup],
        step: int,
        batch_size: int = 50,
    ) -> list[list[ProgressiveCpaResult]]:
        """Progressive (MTD) counterpart of attack_groups(): one snapshot list per byte,
        flattened across all groups in the order given. Each snapshot list has a cumulative
        result every `step` traces; the byte is identified by snapshots[0].byte_result.byte_index.
        Shares reads across a group's byte_locations the same way attack_groups() does.
        """
        if step % batch_size != 0:
            raise ValueError(
                f"step ({step}) must be a multiple of batch_size ({batch_size}) "
                "so snapshots are evenly spaced"
            )

        all_snapshots = []
        for group in groups:
            per_byte_functions = {
                b: self.build_functions(b) for b in group.byte_locations
            }
            all_functions = [fn for fns in per_byte_functions.values() for fn in fns]

            trace_statistics_accumulator = TraceStatisticsAccumulator()
            trace_range = group.range_parameters.trace_range
            _, batch_range_list = get_progress_batch(
                batch_size=batch_size,
                progress_steps=trace_range.count,
                trace_range=trace_range,
            )
            snapshots_by_byte = {b: [] for b in group.byte_locations}
            seen = 0
            for batch_range in tqdm(
                batch_range_list,
                desc=f"Bytes {group.byte_locations} ({len(all_functions)} CPAs)",
                unit="batch",
            ):
                batch = self._project_file_reader.get_batch(
                    batch_range, sample_range=group.range_parameters.sample_range
                )
                known_data = batch.metadata[self._data_source.value]
                trace_statistics_accumulator.update(batch.traces)
                for fn in all_functions:
                    fn.correlation.update(
                        TraceAndModeledLeakage(
                            traces=batch.traces,
                            modeled_leakage=fn.leakage_model.calculate(known_data),
                        )
                    )
                seen += batch_range.count
                if seen % step == 0:
                    statistics = trace_statistics_accumulator.compute()
                    for b, fns in per_byte_functions.items():
                        snapshots_by_byte[b].append(
                            ProgressiveCpaResult(
                                processed_traces=seen,
                                byte_result=self._combine(fns, statistics, b),
                            )
                        )
            all_snapshots.extend(snapshots_by_byte[b] for b in group.byte_locations)
        return all_snapshots
