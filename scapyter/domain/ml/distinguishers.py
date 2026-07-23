from typing import Protocol

from scapyter.domain.leakage.leakage import LeakageModel
from scapyter.domain.ml.value_objects import TrainingResult
from scapyter.domain.value_object import DataSource
from scapyter.infrastructure.ml.stream_dataset import StreamDataset


class NonProfiledDistinguisher(Protocol):
    """Evaluates a single key hypothesis."""

    def evaluate(
        self,
        train_dataset: StreamDataset,
        validation_dataset: StreamDataset | None,
        leakage_model: LeakageModel,
        byte_location: int,
        key_guess: int,
        data_source: DataSource,
    ) -> TrainingResult:
        raise NotImplementedError
