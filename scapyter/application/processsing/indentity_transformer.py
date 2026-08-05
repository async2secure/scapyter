import numpy as np

from scapyter.application.processsing.trace_transformer import TraceTransformer


class IdentityTransformer(TraceTransformer):
    def transform(self, traces: np.ndarray) -> np.ndarray:
        return traces
