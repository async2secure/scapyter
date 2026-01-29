from dataclasses import dataclass
from enum import Enum
from typing import Iterator

import numpy as np


@dataclass(frozen=True)
class KeyGuessList:
    values: list[int]

    def __post_init__(self) -> None:
        """must be within 0 to 255"""
        for v in self.values:
            assert 0 <= v < 256
        """must not have duplication"""
        assert len(set(self.values)) == len(self.values)

    @property
    def to_dict(self) -> dict[str, list[int]]:
        return {"guess_range": self.values}

    @classmethod
    def from_dict(cls, value: dict[str, any]) -> "KeyGuessList":
        return KeyGuessList(value["guess_range"])

    @property
    def number_of_guesses(self) -> int:
        return len(self.values)

    @classmethod
    def from_full256_range(cls) -> "KeyGuessList":
        return cls(list(range(256)))

    def difference(self, other: "KeyGuessList") -> "KeyGuessList":
        """Returns a new GuessRange with the difference of the values."""
        new_values = sorted(set(self.values).difference(other.values))
        return KeyGuessList(new_values)

    def __iter__(self) -> Iterator[int]:
        return iter(self.values)


@dataclass(frozen=True)
class TraceAndModeledLeakage:
    traces: np.ndarray  # Shape: (num_traces, num_samples)
    modeled_leakage: np.ndarray  # Shape: (num_traces, num_guesses)

    def __post_init__(self) -> None:
        # Check for None/Empty
        if self.traces.size == 0 or self.modeled_leakage.size == 0:
            raise ValueError("Traces and Power Model cannot be empty.")

        # Ensure trace counts match (Dimension N)
        if self.traces.shape[0] != self.modeled_leakage.shape[0]:
            raise ValueError(
                f"Batch size mismatch: Traces have {self.traces.shape[0]} rows, "
                f"but Power Model has {self.modeled_leakage.shape[0]} rows."
            )

        # Validate Guess Count (Dimension G)
        num_guesses = self.modeled_leakage.shape[1]
        if not (0 < num_guesses <= 256):
            raise ValueError(
                f"Guess count {num_guesses} is out of valid byte range (1-256)."
            )

    @property
    def trace_count(self) -> int:
        return self.traces.shape[0]

    @property
    def sample_count(self) -> int:
        return self.traces.shape[1]

    @property
    def guess_count(self) -> int:
        return self.modeled_leakage.shape[1]


@dataclass(frozen=True)
class Range:
    start: int
    end: int

    @property
    def as_tuple(self) -> tuple[int, int]:
        return self.start, self.end

    @property
    def to_slice(self) -> slice:
        return slice(self.start, self.end)

    @property
    def count(self) -> int:
        return self.end - self.start

    @property
    def is_valid(self) -> bool:
        if self.start == self.end:
            return False
        elif self.count == 0:
            return False
        elif self.start >= self.end:
            return False
        return True

    @classmethod
    def from_tuple(cls, value: tuple[int, int]) -> "Range":
        return cls(start=value[0], end=value[1])

    @classmethod
    def from_built_in_range(cls, r: range) -> "Range":
        return cls(start=r.start, end=r.stop)

    @classmethod
    def from_zero_start(cls, end: int) -> "Range":
        return cls(start=0, end=end)


@dataclass(frozen=True)
class RangeParameters:
    trace_range: Range
    trace_sample_range: Range
    key_guess_list: KeyGuessList


class DataSource(Enum):
    PLAINTEXT = "plaintext"
    CIPHERTEXT = "ciphertext"
