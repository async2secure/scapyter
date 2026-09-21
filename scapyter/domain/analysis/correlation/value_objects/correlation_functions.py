from dataclasses import dataclass

from scapyter.domain.analysis.correlation.correlation import Correlation
from scapyter.domain.leakage_model.leakage_model import LeakageModel


@dataclass
class CorrelationFunction:
    correlation: Correlation
    leakage_model: LeakageModel
