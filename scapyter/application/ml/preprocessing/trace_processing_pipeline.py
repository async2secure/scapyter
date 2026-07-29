import numpy as np

from scapyter.application.ml.preprocessing.trace_preprocessor import TracePreprocessor


class TraceProcessingPipeline:
    def __init__(self, preprocessors: list[TracePreprocessor]):
        self._preprocessors = preprocessors

    def partial_fit(self, traces: np.ndarray):
        for preprocessor in self._preprocessors:
            preprocessor.partial_fit(traces)

    def finalize(self):
        for preprocessor in self._preprocessors:
            preprocessor.finalize()

    def transform(self, traces: np.ndarray) -> np.ndarray:
        for preprocessor in self._preprocessors:
            traces = preprocessor.transform(traces)

        return traces
