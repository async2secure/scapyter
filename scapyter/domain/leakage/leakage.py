import numpy as np

from abc import abstractmethod, ABC
from scapyter.domain.leakage.constants.hamming_weight_value import HW
from scapyter.domain.leakage.constants.sbox_values import SBOX, INV_SBOX


def inverse_shift_rows(state: np.ndarray) -> np.ndarray:
    """
    Apply AES inverse ShiftRows.

    Input shape:
        (..., 16)

    AES state:

        0   1   2   3
        4   5   6   7
        8   9  10  11
        12 13  14  15
    """
    state = np.asarray(state)

    if state.shape[-1] != 16:
        raise ValueError(f"Expected last dimension to be 16, got {state.shape}")

    return state[
        ...,
        [
            0,
            13,
            10,
            7,
            4,
            1,
            14,
            11,
            8,
            5,
            2,
            15,
            12,
            9,
            6,
            3,
        ],
    ]


def shift_rows(state: np.ndarray) -> np.ndarray:
    """
    Apply AES ShiftRows.

    Input shape:
        (..., 16)

    AES state:

        0   1   2   3
        4   5   6   7
        8   9  10  11
        12 13  14  15
    """
    state = np.asarray(state)

    if state.shape[-1] != 16:
        raise ValueError(f"Expected last dimension to be 16, got {state.shape}")

    return state[
        ...,
        [
            0,
            5,
            10,
            15,
            4,
            9,
            14,
            3,
            8,
            13,
            2,
            7,
            12,
            1,
            6,
            11,
        ],
    ]


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


class InverseShiftRowInvSboxOutputLeakageModel(LeakageModel):

    def calculate(
        self,
        known_data: np.ndarray,
        byte_location: int,
        key_guess: int,
        meta: dict[str, np.ndarray] | None = None,
    ) -> np.ndarray:

        state = inverse_shift_rows(known_data)

        sliced_data = state[:, byte_location]

        state = sliced_data ^ key_guess

        intermediate_values = INV_SBOX[state]

        return HW[intermediate_values]


import numpy as np


class ShiftRowsInvSboxXorLeakageModel(LeakageModel):

    def calculate(
        self,
        known_data: np.ndarray,
        byte_location: int,
        key_guess: int,
        meta: dict[str, np.ndarray] | None = None,
    ) -> np.ndarray:

        # SR(CT)
        shifted_ct = shift_rows(known_data)

        # Select the target byte of SR(CT)
        sr_byte = shifted_ct[:, byte_location]

        # CT ^ KEY
        ct_key = known_data[:, byte_location] ^ key_guess

        # ISB(CT ^ KEY)
        inv_sbox_out = INV_SBOX[ct_key]

        # SR(CT) ^ ISB(CT ^ KEY)
        intermediate = sr_byte ^ inv_sbox_out

        # HW(SR(CT) ^ ISB(CT ^ KEY))
        return HW[intermediate]


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
