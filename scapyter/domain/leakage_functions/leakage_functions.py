from abc import ABC, abstractmethod

import numpy as np

from scapyter.domain.leakage_functions.constants.hamming_weight_value import HW


class LeakageFunction(ABC):

    @abstractmethod
    def calculate(self, value: np.ndarray) -> np.ndarray:
        raise NotImplementedError


class Value(LeakageFunction):
    """
    Value leakage model.

    Returns the intermediate value unchanged.
    """

    def calculate(self, value: np.ndarray) -> np.ndarray:
        return value


class HammingWeight(LeakageFunction):
    """
    Hamming-weight leakage model.

    For each intermediate value, returns the number of set bits.
    """

    def calculate(self, value: np.ndarray) -> np.ndarray:
        return HW[value]


class Monobit(LeakageFunction):
    """
    Single-bit leakage model.

    Args:
        bit: Bit position to extract, from 0 to 7.
    """

    def __init__(self, bit: int) -> None:
        if not isinstance(bit, int):
            raise TypeError(f"bit should be an int, not {type(bit).__name__}")

        if bit < 0 or bit > 7:
            raise ValueError(f"bit should be between 0 and 7, not {bit}")

        self.bit = bit

    def calculate(self, value: np.ndarray) -> np.ndarray:
        return (value >> self.bit) & 1


class Weight(LeakageFunction):
    """
    Weighted Hamming-weight leakage model.

    The weight is applied to each bit position before summing.

    For example, with weight=3:

        bit 0 -> 3
        bit 1 -> 3
        ...
        bit 7 -> 3

    This preserves the behavior of the old WeightComputeModel if
    its parameter represents the Hamming-weight multiplier.
    """

    def __init__(self, weight: int) -> None:
        if not isinstance(weight, int):
            raise TypeError(f"weight should be an int, not {type(weight).__name__}")

        if weight < 0:
            raise ValueError(f"weight should be >= 0, not {weight}")

        self.weight = weight

    def calculate(self, value: np.ndarray) -> np.ndarray:
        return HW[value] * self.weight


class ZeroValue(LeakageFunction):
    """
    Zero-value leakage model.

    Returns 0 when the intermediate value equals `compare`,
    otherwise returns 1.

        value == compare  -> 0
        value != compare  -> 1

    This is the equivalent of the old ZeroValueComputeModel.
    """

    def __init__(self, compare: int) -> None:
        if not isinstance(compare, int):
            raise TypeError(f"compare should be an int, not {type(compare).__name__}")

        if compare < 0 or compare > 255:
            raise ValueError(f"compare should be between 0 and 255, not {compare}")

        self.compare = compare

    def calculate(self, value: np.ndarray) -> np.ndarray:
        return np.where(value == self.compare, 0, 1)
