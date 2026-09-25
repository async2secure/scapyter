from dataclasses import dataclass

from scapyter.domain.value_object import CpaByteResult


@dataclass(frozen=True)
class ProgressiveCpaResult:
    processed_traces: int
    byte_result: CpaByteResult
