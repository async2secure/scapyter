import torch
from torch.utils.data import Dataset

from scapyter.domain.leakage.leakage import LeakageModel
from scapyter.domain.value_object import DataSource
from scapyter.infrastructure.ml.strategy.loss_strategy import LossStrategy
from scapyter.infrastructure.ml.stream_dataset import StreamDataset


class HypotheticalLabelDataset(Dataset):

    def __init__(
        self,
        base_dataset: StreamDataset,
        leakage_model: LeakageModel,
        key_guess: int,
        byte_location: int,
        data_source: DataSource,
        loss_strategy: LossStrategy,
        num_classes: int,
    ):
        self._base = base_dataset
        self._leakage_model = leakage_model
        self._key_guess = key_guess
        self._byte_location = byte_location
        self._data_source = data_source
        self._loss_strategy = loss_strategy
        self._num_classes = num_classes

    def __len__(self):
        return len(self._base)

    def __getitem__(self, idx):

        trace, metadata = self._base[idx]

        if self._data_source == DataSource.PLAINTEXT:
            known = metadata["plaintext"]

        elif self._data_source == DataSource.CIPHERTEXT:
            known = metadata["ciphertext"]

        else:
            raise ValueError(f"Unsupported data source: {self._data_source}")

        raw_label = self._leakage_model.calculate(
            known,
            key_guess=self._key_guess,
            byte_location=self._byte_location,
        )
        raw_label = torch.as_tensor(raw_label).item()

        label = self._loss_strategy.encode_target(
            raw_label,
            self._num_classes,
        )

        return trace, label
