from dataclasses import dataclass, field
import numpy as np


@dataclass
class InputTraceBatch:
    traces: np.ndarray  # Shape: (N, Samples)
    # Flexible dictionary to hold any metadata (plaintext, key, nonce, etc.)
    metadata: dict[str, np.ndarray] = field(default_factory=dict)

    def __post_init__(self):
        # Validate that all arrays have the same number of rows (N)
        batch_size = self.traces.shape[0]
        for key, arr in self.metadata.items():
            if arr.shape[0] != batch_size:
                raise ValueError(f"Metadata '{key}' size does not match trace count.")


@dataclass(frozen=True)
class OutputTraceBatch:
    indices: range
    samples: np.ndarray  # Shape: (N, Samples)
    metadata: dict[str, np.ndarray]  # Values are Shape: (N, Bytes)

    def __len__(self):
        return self.samples.shape[0]


@dataclass(frozen=True)
class TraceEntity:
    """A single trace and all its associated metadata."""

    index: int
    samples: np.ndarray
    metadata: dict[str, np.ndarray] = field(default_factory=dict)

    def __getitem__(self, key: str):
        return self.metadata.get(key)

    @property
    def plaintext(self):
        return self.metadata.get("plaintext")

    @property
    def key(self):
        return self.metadata.get("key")
