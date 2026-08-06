from scapyter.application.ml.preprocessing.trace_processing_pipeline import (
    TraceProcessingPipeline,
)
from scapyter.domain.repository.project_file_reader import ProjectFileReader
from scapyter.domain.value_object import Range


class ProcessedProjectFileReader(ProjectFileReader):

    def __init__(
        self,
        wrapped: ProjectFileReader,
        pipeline: TraceProcessingPipeline | None = None,
    ):
        self._wrapped = wrapped
        self._pipeline = pipeline

    def fit_processing(
        self,
        trace_range: Range,
        sample_range: Range | None = None,
        chunk_size: int = 1000,
    ):
        """
        Fit preprocessing using training traces only.

        Data is loaded in chunks so the whole dataset
        does not need to fit in memory.
        """

        if self._pipeline is None:
            return

        for start in range(
            trace_range.start,
            trace_range.end,
            chunk_size,
        ):
            end = min(
                start + chunk_size,
                trace_range.end,
            )

            batch = self._wrapped.get_batch(
                trace_range=Range(start, end),
                sample_range=sample_range,
            )

            self._pipeline.partial_fit(batch.traces)

        self._pipeline.finalize()

    def get_single_batch(
        self,
        index: int,
        sample_range: Range | None = None,
    ):
        batch = self._wrapped.get_single_batch(
            index=index,
            sample_range=sample_range,
        )

        if self._pipeline is None:
            return batch

        return batch.copy_with(
            traces=self._pipeline.transform(batch.traces),
        )

    def get_batch(
        self,
        trace_range: Range,
        sample_range: Range | None = None,
    ):
        batch = self._wrapped.get_batch(
            trace_range=trace_range,
            sample_range=sample_range,
        )

        if self._pipeline is None:
            return batch

        return batch.copy_with(
            traces=self._pipeline.transform(batch.traces),
        )
