import torch
from torch.utils.data import Dataset

from scapyter.domain.repository.project_file_reader import ProjectFileReader
from scapyter.domain.value_object import Range


class StreamDataset(Dataset):

    def __init__(
        self,
        repo: ProjectFileReader,
        indices,
        trace_range: Range,
        sample_range: Range,
        scaler=None,
    ):
        self.repo = repo
        self.indices = indices
        self.scaler = scaler
        self._sample_range = sample_range

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, i):
        idx = self.indices[i]

        batch = self.repo.get_single_batch(idx, sample_range=self._sample_range)
        trace = batch.traces.squeeze()
        # trace = batch.traces
        if self.scaler:
            trace = self.scaler.transform(trace)

        # trace = torch.tensor(trace, dtype=torch.float32).squeeze()

        return torch.as_tensor(trace, dtype=torch.float32), batch.metadata
