import numpy as np

from domain.correlation.correlation import Correlation
from domain.leakage.leakage import LeakageModel
from domain.progress_range.progress_range import get_progress_batch
from domain.repository.trace_repository import TraceRepository
from domain.value_object import (
    RangeParameters,
    TraceAndModeledLeakage,
    DataSource,
    KeyByteGuesses,
)


class CorrelationTask:
    def __init__(
        self,
        byte_location: int,
        range_parameters: RangeParameters,
        leakage_model: LeakageModel,
        correlation: Correlation,
        trace_repository: TraceRepository,
        data_source: DataSource,
        key_byte_guesses: KeyByteGuesses,
    ):
        self._leakage_model = leakage_model
        self._correlation = correlation
        self._trace_repository = trace_repository
        self._range_parameters = range_parameters
        self._byte_location = byte_location
        self._data_source = data_source
        self._key_byte_guesses = key_byte_guesses

    def run(self, batch_size=50):
        trace_range = self._range_parameters.trace_range
        progress_steps = trace_range.count
        progress, batch_range_list = get_progress_batch(
            batch_size=batch_size,
            progress_steps=progress_steps,
            trace_range=trace_range,
        )
        for batch_range in batch_range_list:
            sample_range = self._range_parameters.trace_sample_range
            batch = self._trace_repository.get_batch(
                batch_range, sample_slice=slice(sample_range.start, sample_range.end)
            )
            known_data = batch.metadata[self._data_source.value]
            modeled_leakages = []
            for key_guess in self._key_byte_guesses:
                modeled_leakage = self._leakage_model.calculate(
                    byte_location=self._byte_location,
                    known_data=known_data,
                    key_guess=key_guess,
                )
                modeled_leakages.append(modeled_leakage)

            trace_and_modeled_leakage = TraceAndModeledLeakage(
                traces=batch.samples, modeled_leakage=np.asarray(modeled_leakages).T
            )
            self._correlation.update(trace_and_modeled_leakage)
        return self._correlation.compute()
