"""
Chosen-ciphertext mirror of ChosenPlaintextRegisterAttack: runs the 32-CPA register attack
(ChosenCiphertextDeltaByte, 4 output bytes x Monobit(0..7), combined via the sum-|rho| service)
over whichever traces the caller's RangeParameters select.
"""

from scapyter.application.analysis.correlation.chosen_analysis.sum_abs_correlation_service import (
    SumAbsCorrelationService,
)
from scapyter.domain.analysis.correlation.cpa import CpaCorrelation
from scapyter.domain.analysis.correlation.value_objects.progressive_cpa_result import (
    ProgressiveCpaResult,
)
from scapyter.domain.leakage_functions.leakage_functions import Monobit
from scapyter.domain.leakage_model.leakage_model import LeakageModel
from scapyter.domain.repository.project_file_reader import ProjectFileReader
from scapyter.domain.targets.targets import ChosenCiphertextDeltaByte
from scapyter.domain.value_object import (
    CpaByteResult,
    DataSource,
    KeyByteGuesses,
    RangeParameters,
)
from scapyter.domain.analysis.correlation.value_objects.correlation_functions import (
    CorrelationFunction,
)


class ChosenCiphertextRegisterAttack:
    """Runs the 32-CPA register attack over the traces selected by the
    `range_parameters` passed to each call."""

    def __init__(
        self,
        project_file_reader: ProjectFileReader,
        data_source: DataSource = DataSource.CIPHERTEXT,
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
                    target=ChosenCiphertextDeltaByte(output_byte),
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
