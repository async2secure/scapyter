import numpy as np

from scapyter.domain.analysis.correlation.value_objects.trace_statistics import (
    TraceStatistics,
)


class TraceStatisticsAccumulator:
    def __init__(self):
        self.acc_x2 = None
        self.trace_sum = None
        self.processed_traces = 0

    def update(self, traces):
        batch_sum = traces.sum(axis=0)

        if self.trace_sum is None:
            self.trace_sum = batch_sum
            self.acc_x2 = np.square(traces).sum(axis=0)
        else:
            self.trace_sum += batch_sum
            self.acc_x2 += np.square(traces).sum(axis=0)

        self.processed_traces += traces.shape[0]

    def compute(self) -> TraceStatistics:
        mean = self.trace_sum / self.processed_traces
        variance = self.acc_x2 / self.processed_traces - mean**2
        return TraceStatistics(mean, variance, self.processed_traces)
