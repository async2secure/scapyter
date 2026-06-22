from dataclasses import dataclass

import numpy as np


@dataclass
class TraceStatistics:
    mean: np.ndarray
    variance: np.ndarray
    processed_traces: int
