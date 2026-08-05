import numpy as np
from tqdm import tqdm

from scapyter.domain.analysis.snr.snr import ProgressiveSnr
from scapyter.domain.leakage.leakage import LeakageModel
from scapyter.domain.progress_range.progress_range import get_progress_batch
from scapyter.domain.repository.project_file_reader import ProjectFileReader
from scapyter.domain.value_object import RangeParameters, DataSource


class MultiByteSnrService:
    def __init__(
        self,
        range_parameters: RangeParameters,
        known_key_bytes: dict[int, int],
        leakage_model: LeakageModel,
        project_file_reader: ProjectFileReader,
        data_source: DataSource,
    ):
        self._byte_locations = list(known_key_bytes.keys())
        self._range_parameters = range_parameters
        self._known_key_bytes = known_key_bytes
        self._leakage_model = leakage_model
        self._project_file_reader = project_file_reader
        self._data_source = data_source

        self._snrs = {byte: ProgressiveSnr() for byte in self._byte_locations}

    def run(self, batch_size: int = 50):

        trace_range = self._range_parameters.trace_range

        _, batch_range_list = get_progress_batch(
            batch_size=batch_size,
            progress_steps=trace_range.count,
            trace_range=trace_range,
        )

        sample_range = self._range_parameters.sample_range

        for batch_range in tqdm(batch_range_list, desc="SNR"):

            # Read ONCE
            batch = self._project_file_reader.get_batch(
                batch_range,
                sample_slice=slice(
                    sample_range.start,
                    sample_range.end,
                ),
            )

            known_data = batch.metadata[self._data_source.value]

            # Reuse same traces
            for byte in self._byte_locations:
                modeled_leakage = self._leakage_model.calculate(
                    byte_location=byte,
                    known_data=known_data,
                    key_guess=self._known_key_bytes[byte],
                    meta=batch.metadata,
                )

                self._snrs[byte].update(
                    traces=batch.traces,
                    hex_array=np.asarray(modeled_leakage),
                )

        return {byte: snr.finalize() for byte, snr in self._snrs.items()}
