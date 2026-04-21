import numpy as np

from scapyter.domain.leakage.leakage import LeakageModel
from scapyter.domain.progress_range.progress_range import get_progress_batch
from scapyter.domain.repository.trace_repository import TraceRepository
from scapyter.domain.snr.snr import ProgressiveSnr
from scapyter.domain.value_object import RangeParameters, DataSource


class SnrTask:
    def __init__(
        self,
        byte_location: int,
        range_parameters: RangeParameters,
        known_key_byte: int,
        leakage_model: LeakageModel,
        trace_repository: TraceRepository,
        data_source: DataSource,
        snr: ProgressiveSnr,
    ) -> None:
        self._byte_location = byte_location
        self._range_parameters = range_parameters
        self._leakage_model = leakage_model
        self._trace_repository = trace_repository
        self._data_source = data_source
        self._snr = snr
        self._known_key_byte = known_key_byte

    def run(self, batch_size: int = 50) -> np.ndarray:
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

            modeled_leakage = self._leakage_model.calculate(
                byte_location=self._byte_location,
                known_data=known_data,
                key_guess=self._known_key_byte,
            )

            self._snr.update(traces=batch.traces, hex_array=np.asarray(modeled_leakage))

        return self._snr.finalize()
