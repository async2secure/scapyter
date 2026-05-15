from scapyter.domain.signal_processing.trace_processor import TraceProcessor
from scapyter.domain.value_object import Range, RangeParameters
from scapyter.infrastructure.h5_project_file_reader import H5ProjectFileReader
from scapyter.infrastructure.h5_project_file_writer import H5ProjectFileWriter


class TraceProcessingService:

    def __init__(
        self,
        input_path: str,
        output_path: str,
        processor: TraceProcessor,
        range_parameters: RangeParameters | None = None,
        batch_size: int = 1000,
    ):
        self.input_path = input_path
        self.output_path = output_path
        self.processor = processor
        self.range_parameters = range_parameters
        self.batch_size = batch_size

    def execute(self):

        with H5ProjectFileReader(self.input_path) as reader:

            # Resolve ranges
            if self.range_parameters is None:

                trace_range = Range(
                    start=0,
                    end=reader.trace_count,
                )

                sample_range = Range(
                    start=0,
                    end=reader.sample_count,
                )

            else:
                trace_range = self.range_parameters.trace_range
                sample_range = self.range_parameters.sample_range

            writer = H5ProjectFileWriter(
                file_path=self.output_path,
                total_traces=trace_range.count,
            )

            sample_slice = slice(
                sample_range.start,
                sample_range.end,
            )

            for start in range(
                trace_range.start,
                trace_range.end,
                self.batch_size,
            ):
                end = min(
                    start + self.batch_size,
                    trace_range.end,
                )

                batch = reader.get_batch(
                    trace_range=Range(start, end),
                    sample_slice=sample_slice,
                )

                processed_batch = self.processor.process(batch)

                writer.save_batch(processed_batch)
