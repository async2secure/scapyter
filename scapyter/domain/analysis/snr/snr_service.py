import numpy as np
from tqdm import tqdm

from scapyter.domain.leakage_model.leakage_model import LeakageModel
from scapyter.domain.targets.targets import Target
from scapyter.domain.progress_range.progress_range import get_progress_batch
from scapyter.domain.repository.project_file_reader import ProjectFileReader
from scapyter.domain.analysis.snr.snr import ProgressiveSnr
from scapyter.domain.value_object import RangeParameters, DataSource


class SnrService:
    def __init__(
        self,
        range_parameters: RangeParameters,
        project_file_reader: ProjectFileReader,
        data_source: DataSource,
    ) -> None:
        self._range_parameters = range_parameters
        self._project_file_reader = project_file_reader
        self._data_source = data_source

    def run(
        self,
        leakage_model: LeakageModel,
        batch_size: int = 50,
    ) -> np.ndarray:
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
            modeled_leakages = leakage_model.calculate(known_data)
            modeled_leakages = modeled_leakages.T[0]
            snr.update(traces=batch.traces, hex_array=np.asarray(modeled_leakages))

        return snr.finalize()
