from dataclasses import dataclass

from scapyter.domain.analysis.correlation.correlation import Correlation
from scapyter.domain.leakage.leakage import LeakageModel
from scapyter.domain.value_object import KeyByteGuesses


@dataclass
class CorrelationFunction:
    key_byte_guesses: KeyByteGuesses
    correlation: Correlation
    byte_location: int
    leakage_model: LeakageModel
