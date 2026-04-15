from dataclasses import dataclass, field
import numpy as np


@dataclass
class InputBatch:
    traces: np.ndarray  # Shape: (N, Samples)
    # Flexible dictionary to hold any metadata (plaintext, key, nonce, etc.)
    metadata: dict[str, np.ndarray] = field(default_factory=dict)

    def __post_init__(self):
        # Validate that all arrays have the same number of rows (N)
        batch_size = self.traces.shape[0]
        for key, arr in self.metadata.items():
            if arr.shape[0] != batch_size:
                raise ValueError(f"Metadata '{key}' size does not match trace count.")
