import numpy as np

from scapyter.application.processsing.trace_transformer import TraceTransformer


class TraceProcessingPipeline:
    def __init__(self, transformers: list[TraceTransformer]):
        self._transformers = transformers

    def transform(self, traces: np.ndarray) -> np.ndarray:
        for transformer in self._transformers:
            traces = transformer.transform(traces)
        return traces
