from dataclasses import dataclass

import numpy as np

from scapyter.domain.leakage_functions.leakage_functions import LeakageFunction
from scapyter.domain.targets.targets import Target
from scapyter.domain.value_object import KeyByteGuesses


@dataclass(frozen=True)
class LeakageModel:
    target: Target
    leakage_function: LeakageFunction
    byte_location: int
    key_byte_guesses: KeyByteGuesses

    def calculate(
        self,
        known_data: np.ndarray,
    ) -> np.ndarray:
        modeled_leakages = [
            self.leakage_function.calculate(
                self.target.calculate(
                    byte_location=self.byte_location,
                    known_data=known_data,
                    key_guess=key_guess,
                )
            )
            for key_guess in self.key_byte_guesses
        ]

        return np.asarray(modeled_leakages).T
