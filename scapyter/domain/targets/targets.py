from abc import abstractmethod, ABC

import numpy as np
from scapyter.domain.targets.constants.sbox_values import SBOX, INV_SBOX


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


class Target(ABC):

    @abstractmethod
    def calculate(
        self,
        known_data: np.ndarray,
        byte_location: int,
        key_guess: int,
        meta: dict[str, np.ndarray] | None = None,
    ) -> np.ndarray:
        raise NotImplementedError


class InvSboxOutput(Target):

    def calculate(
        self,
        known_data: np.ndarray,
        byte_location: int,
        key_guess: int,
        meta: dict[str, np.ndarray] | None = None,
    ) -> np.ndarray:
        sliced_data = known_data[:, byte_location]
        state = sliced_data ^ key_guess
        return INV_SBOX[state]


class InverseShiftRowInvSboxOutput(Target):

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

        return INV_SBOX[state]


class ShiftRowsInvSboxXor(Target):

    def calculate(
        self,
        known_data: np.ndarray,
        byte_location: int,
        key_guess: int,
        meta: dict[str, np.ndarray] | None = None,
    ) -> np.ndarray:
        # SR(CT)
        shifted_ct = shift_rows(known_data)

        # Select the targets byte of SR(CT)
        sr_byte = shifted_ct[:, byte_location]

        # CT ^ KEY
        ct_key = known_data[:, byte_location] ^ key_guess

        # ISB(CT ^ KEY)
        inv_sbox_out = INV_SBOX[ct_key]

        # SR(CT) ^ ISB(CT ^ KEY)
        return sr_byte ^ inv_sbox_out


class SboxOutput(Target):

    def calculate(
        self,
        known_data: np.ndarray,
        byte_location: int,
        key_guess: int,
        meta: dict[str, np.ndarray] | None = None,
    ) -> np.ndarray:
        plaintext_byte = known_data[:, byte_location]
        state = plaintext_byte ^ key_guess
        return SBOX[state]


class SboxInputOutputHammingDistance(Target):

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
        return state ^ sbox_out


def _gmul(a: int, b: int) -> int:
    r = 0
    while b:
        if b & 1:
            r ^= a
        a = ((a << 1) ^ 0x11B) & 0xFF if a & 0x80 else a << 1
        b >>= 1
    return r


_FMUL = {
    c: np.array([_gmul(x, c) for x in range(256)], dtype=np.uint8) for c in (1, 2, 3)
}
_BMUL = {
    c: np.array([_gmul(x, c) for x in range(256)], dtype=np.uint8)
    for c in (9, 11, 13, 14)
}
_IMC = [
    [14, 11, 13, 9],
    [9, 14, 11, 13],
    [13, 9, 14, 11],
    [11, 13, 9, 14],
]  # InvMixColumns
_MC = [[2, 3, 1, 1], [1, 2, 3, 1], [1, 1, 2, 3], [3, 1, 1, 2]]  # MixColumns matrix


class ChosenCiphertextDeltaByte(Target):
    def __init__(self, output_byte: int) -> None:
        if output_byte not in (0, 1, 2, 3):
            raise ValueError(f"output_byte must be 0..3, not {output_byte}")
        self.output_byte = output_byte

    def calculate(self, known_data, byte_location, key_guess, meta=None) -> np.ndarray:
        row = byte_location % 4
        col = (byte_location // 4 + row) % 4
        s = INV_SBOX[known_data[:, byte_location] ^ key_guess]
        return (
            _BMUL[_IMC[self.output_byte][row]][s]
            ^ known_data[:, 4 * col + self.output_byte]
        )


class ChosenPlaintextRegisterDeltaByte(Target):
    def __init__(self, output_byte: int) -> None:
        if output_byte not in (0, 1, 2, 3):
            raise ValueError(f"output_byte must be 0..3, not {output_byte}")
        self.output_byte = output_byte

    def calculate(self, known_data, byte_location, key_guess, meta=None) -> np.ndarray:
        row = byte_location % 4
        col = (byte_location // 4 - row) % 4
        s = SBOX[known_data[:, byte_location] ^ key_guess]
        return (
            _FMUL[_MC[self.output_byte][row]][s]
            ^ known_data[:, 4 * col + self.output_byte]
        )
