from dataclasses import dataclass

from scapyter.domain.value_object import RangeParameters


@dataclass(frozen=True)
class AttackGroup:
    byte_locations: list[int]
    range_parameters: RangeParameters
