from scapyter.application.processsing.trace_processing_pipeline import (
    TraceProcessingPipeline,
)
from scapyter.domain.repository.project_file_reader import ProjectFileReader


class ProcessedProjectFileReader(ProjectFileReader):
    def __init__(
        self,
        wrapped: ProjectFileReader,
        pipeline: TraceProcessingPipeline | None = None,
    ):
        self._wrapped = wrapped
        self._pipeline = pipeline

    def get_batch(self, *args, **kwargs):
        batch = self._wrapped.get_batch(*args, **kwargs)

        return batch.copy_with(
            traces=self._pipeline.transform(batch.traces),
        )
