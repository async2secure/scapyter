from scapyter.application.processsing.trace_processing_pipeline import (
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
        self._wrapped: ProjectFileReader = wrapped
        self._pipeline = pipeline

    def fit(
        self,
        trace_range: Range,
        sample_range: Range | None = None,
    ):
        if self._pipeline is None:
            return

        batch = self._wrapped.get_batch(
            trace_range=trace_range,
            sample_range=sample_range,
        )

        self._pipeline.fit(batch.traces)

    def get_single_batch(self, index: int, sample_range: Range | None = None):
        batch = self._wrapped.get_single_batch(
            index=index,
            sample_range=sample_range,
        )

        if self._pipeline is None:
            return batch

        return batch.copy_with(
            traces=self._pipeline.transform(batch.traces),
        )

    def get_batch(self, trace_range: Range, sample_range: Range | None = None):
        batch = self._wrapped.get_batch(
            trace_range=trace_range,
            sample_range=sample_range,
        )

        if self._pipeline is None:
            return batch

        return batch.copy_with(
            traces=self._pipeline.transform(batch.traces),
        )
