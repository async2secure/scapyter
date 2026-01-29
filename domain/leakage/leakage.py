from abc import abstractmethod, ABC

import numpy as np

from domain.leakage.constants.humming_weight_value import HW
from domain.leakage.constants.sbox_values import SBOX


class LeakageModel(ABC):

    @abstractmethod
    def calculate(
        self, known_data: np.ndarray, byte_location: int, key_guess: int
    ) -> np.ndarray:
        raise NotImplementedError


class SboxOutputLeakageModel(LeakageModel):

    def calculate(
        self, known_data: np.ndarray, byte_location: int, key_guess: int
    ) -> np.ndarray:
        plaintext_byte = known_data[:, byte_location]
        state = plaintext_byte ^ key_guess
        intermediate_values = SBOX[state]
        return HW[intermediate_values]
