from collections.abc import Callable

from tqdm import tqdm

from scapyter.domain.progress_range.progress_range import get_progress_batch
from scapyter.domain.repository.project_file_reader import ProjectFileReader
from scapyter.domain.value_object import RangeParameters


class TraceBatchService:
    def __init__(
        self,
        range_parameters: RangeParameters,
        project_file_reader: ProjectFileReader,
    ):
        self._range_parameters = range_parameters
        self._project_file_reader = project_file_reader

    def batches(self, batch_size: int = 50):
        trace_range = self._range_parameters.trace_range

        _, batch_range_list = get_progress_batch(
            batch_size=batch_size,
            progress_steps=trace_range.count,
            trace_range=trace_range,
        )

        for batch_range in tqdm(
            batch_range_list,
            desc="Processing traces",
            unit="batch",
        ):
            yield self._project_file_reader.get_batch(
                batch_range,
                sample_range=self._range_parameters.sample_range,
            )
