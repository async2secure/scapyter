import numpy as np

from abc import abstractmethod, ABC
from scapyter.domain.leakage.constants.hamming_weight_value import HW
from scapyter.domain.leakage.constants.sbox_values import SBOX, INV_SBOX


class LeakageModel(ABC):

    @abstractmethod
    def calculate(
        self,
        known_data: np.ndarray,
        byte_location: int,
        key_guess: int,
        meta: dict[str, np.ndarray] | None = None,
    ) -> np.ndarray:
        raise NotImplementedError


class InvSboxOutputLeakageModel(LeakageModel):

    def calculate(
        self,
        known_data: np.ndarray,
        byte_location: int,
        key_guess: int,
        meta: dict[str, np.ndarray] | None = None,
    ) -> np.ndarray:
        sliced_data = known_data[:, byte_location]
        state = sliced_data ^ key_guess
        intermediate_values = INV_SBOX[state]
        return HW[intermediate_values]


class SboxOutputLeakageModel(LeakageModel):

    def calculate(
        self,
        known_data: np.ndarray,
        byte_location: int,
        key_guess: int,
        meta: dict[str, np.ndarray] | None = None,
    ) -> np.ndarray:
        plaintext_byte = known_data[:, byte_location]
        state = plaintext_byte ^ key_guess
        intermediate_values = SBOX[state]
        return HW[intermediate_values]


class SboxInputOutputHammingDistanceLeakageModel(LeakageModel):

    def calculate(
        self,
        known_data: np.ndarray,
        byte_location: int,
        key_guess: int,
        meta: dict[str, np.ndarray] | None = None,
    ) -> np.ndarray:
        plaintext_byte = known_data[:, byte_location]

        # PT ^ K
        state = plaintext_byte ^ key_guess

        # SBOX(PT ^ K)
        sbox_out = SBOX[state]

        # (PT ^ K) ^ SBOX(PT ^ K)
        intermediate = state ^ sbox_out

        # Hamming Weight leakage
        return HW[intermediate]
