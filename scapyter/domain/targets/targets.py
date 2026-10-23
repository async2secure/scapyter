from abc import abstractmethod, ABC

import numpy as np
from scapyter.domain.targets.constants.sbox_values import SBOX, INV_SBOX
from scapyter.domain.targets.utils import (
    inverse_shift_rows,
    shift_rows,
    mix_columns,
    FMUL,
    MC,
    BMUL,
    IMC,
    inverse_mix_columns,
)


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
            BMUL[IMC[self.output_byte][row]][s]
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
            FMUL[MC[self.output_byte][row]][s]
            ^ known_data[:, 4 * col + self.output_byte]
        )


class Round2RegisterDeltaByte(Target):
    def __init__(self, output_byte: int, round_key_0: np.ndarray) -> None:
        if output_byte not in (0, 1, 2, 3):
            raise ValueError(f"output_byte must be 0..3, not {output_byte}")
        self.output_byte = output_byte
        self.round_key_0 = np.asarray(round_key_0, dtype=np.uint8)

    def calculate(self, known_data, byte_location, key_guess, meta=None) -> np.ndarray:
        """byte_location is an R1 BYTE INDEX (0..15), i.e. the k1 byte being guessed -- NOT the
        plaintext byte that varies. Pick the trace block with plaintext_byte_for(byte_location),
        but still pass byte_location (the R1 index) here."""

        known_part = mix_columns(
            shift_rows(SBOX[known_data ^ self.round_key_0])
        )  # R1 without k1
        s = SBOX[known_part[:, byte_location] ^ np.uint8(key_guess)]

        row = byte_location % 4
        col = (byte_location // 4 - row) % 4
        return (
            FMUL[MC[self.output_byte][row]][s]
            ^ known_part[:, 4 * col + self.output_byte]
        )


class Round13RegisterDeltaByte(Target):
    """
    AES-256, chosen ciphertext, K14 known.
    byte_location indexes the equivalent key K13' = IMC(K13), NOT K13.
    """

    def __init__(self, output_byte: int, round_key_14: np.ndarray) -> None:
        if output_byte not in (0, 1, 2, 3):
            raise ValueError(f"output_byte must be 0..3, not {output_byte}")
        self.output_byte = output_byte
        self.round_key_14 = np.asarray(round_key_14, dtype=np.uint8)

    @staticmethod
    def ciphertext_byte_for(byte_location: int) -> int:
        # Row 0 is not moved by ISR, so varying C[4*col] makes exactly
        # column (byte_location // 4) of T vary.
        return 4 * (byte_location // 4)

    def calculate(self, known_data, byte_location, key_guess, meta=None) -> np.ndarray:
        known_part = inverse_mix_columns(
            INV_SBOX[inverse_shift_rows(known_data ^ self.round_key_14)]
        )
        s = INV_SBOX[known_part[:, byte_location] ^ np.uint8(key_guess)]

        row = byte_location % 4
        col = (byte_location // 4 + row) % 4
        return (
            BMUL[IMC[self.output_byte][row]][s]
            ^ known_part[:, 4 * col + self.output_byte]
        )
