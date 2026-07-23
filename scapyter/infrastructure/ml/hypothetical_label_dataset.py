import torch
from torch.utils.data import Dataset

from scapyter.domain.leakage.leakage import LeakageModel
from scapyter.domain.value_object import DataSource
from scapyter.infrastructure.ml.stream_dataset import StreamDataset


class HypotheticalLabelDataset(Dataset):
    def __init__(
        self,
        base_dataset: StreamDataset,
        leakage_model: LeakageModel,
        key_guess: int,
        byte_location: int,
        data_source: DataSource,
    ):
        self._base = base_dataset
        self._leakage_model = leakage_model
        self._key_guess = key_guess
        self._byte_location = byte_location
        self._data_source = data_source

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

        label = self._leakage_model.calculate(
            known, key_guess=self._key_guess, byte_location=self._byte_location
        )

        label = torch.as_tensor(
            label,
            dtype=torch.long,
        ).squeeze()

        return trace, label
