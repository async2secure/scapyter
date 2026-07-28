from dataclasses import dataclass
from typing import Sequence

import numpy as np


@dataclass
class TraceDataset:
    traces: np.ndarray
    known_values: np.ndarray


@dataclass(frozen=True)
class EpochMetrics:
    epoch: int
    train_loss: float
    train_accuracy: float
    validation_loss: float | None = None
    validation_accuracy: float | None = None


@dataclass(frozen=True)
class TrainingResult:
    history: Sequence[EpochMetrics]

    @property
    def final(self) -> EpochMetrics:
        return self.history[-1]

    @property
    def loss(self) -> float:
        if self.final.validation_loss is not None:
            return self.final.validation_loss
        return self.final.train_loss

    @property
    def accuracy(self) -> float:
        if self.final.validation_accuracy is not None:
            return self.final.validation_accuracy
        return self.final.train_accuracy

    @property
    def epochs(self) -> int:
        return len(self.history)


@dataclass(frozen=True)
class AttackMetric:
    key_guess: int
    training: TrainingResult

    @property
    def loss(self):
        return self.training.loss

    @property
    def accuracy(self):
        return self.training.accuracy


@dataclass(frozen=True)
class AttackResult:
    metrics: list[AttackMetric]

    @property
    def best_guess(self) -> int:
        return self.metrics[0].key_guess

    def rank_of(self, key_guess: int) -> int:
        return next(
            i for i, metric in enumerate(self.metrics) if metric.key_guess == key_guess
        )
