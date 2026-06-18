from typing import Protocol
import numpy as np


class TraceTransformer(Protocol):
    def transform(self, traces: np.ndarray) -> np.ndarray:
        """
        Parameters
        ----------
        traces : np.ndarray
            Shape: (num_traces, num_samples)

        Returns
        -------
        np.ndarray
            Transformed traces.
        """
        ...
