import numpy as np
from tqdm import tqdm

from scapyter.domain.leakage.leakage import LeakageModel
from scapyter.domain.progress_range.progress_range import get_progress_batch
from scapyter.domain.repository.project_file_reader import ProjectFileReader
from scapyter.domain.analysis.snr.snr import ProgressiveSnr
from scapyter.domain.value_object import RangeParameters, DataSource


class SnrService:
    def __init__(
        self,
        range_parameters: RangeParameters,
        leakage_model: LeakageModel,
        project_file_reader: ProjectFileReader,
        data_source: DataSource,
    ) -> None:
        self._range_parameters = range_parameters
        self._leakage_model = leakage_model
        self._project_file_reader = project_file_reader
        self._data_source = data_source


    def run(self, byte_location: int, known_key_byte: int,  batch_size: int = 50, ) -> np.ndarray:
        snr = ProgressiveSnr()
        trace_range = self._range_parameters.trace_range
        progress_steps = trace_range.count
        progress, batch_range_list = get_progress_batch(
            batch_size=batch_size,
            progress_steps=progress_steps,
            trace_range=trace_range,
        )

        for batch_range in tqdm(batch_range_list, desc="SNR"):
            sample_range = self._range_parameters.sample_range

            batch = self._project_file_reader.get_batch(
                batch_range, sample_range=sample_range
            )
            known_data = batch.metadata[self._data_source.value]

            modeled_leakage = self._leakage_model.calculate(
                byte_location=byte_location,
                known_data=known_data,
                key_guess=known_key_byte,
                meta=batch.metadata,
            )

            snr.update(traces=batch.traces, hex_array=np.asarray(modeled_leakage))

        return snr.finalize()
