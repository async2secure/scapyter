from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class TvlaResult:
    t_scores: np.ndarray
    trace_start: int
    trace_end: int
    sample_start: int
    sample_end: int
