from scapyter.domain.signal_processing.trace_processor import TraceProcessor
from scapyter.domain.value_object import Batch


class OddTraceProcessor(TraceProcessor):

    def process(self, batch: Batch) -> Batch:
        return Batch(
            indices=range(
                batch.indices.start + 1,
                batch.indices.stop,
                2,
            ),
            traces=batch.traces[1::2],
            metadata={key: value[1::2] for key, value in batch.metadata.items()},
        )

    def output_shape(self, input_shape):
        trace_count, sample_count = input_shape

        return (
            trace_count // 2,
            sample_count,
        )


import h5py
import numpy as np

from scapyter.application.trace_processing_service import TraceProcessingService


def test_trace_processing_service_keeps_odd_traces(tmp_path):
    input_path = tmp_path / "input.h5"
    output_path = tmp_path / "output.h5"

    traces = np.array(
        [
            [0, 0, 0],  # trace 0
            [1, 1, 1],  # trace 1
            [2, 2, 2],  # trace 2
            [3, 3, 3],  # trace 3
            [4, 4, 4],  # trace 4
            [5, 5, 5],  # trace 5
        ],
        dtype=np.float32,
    )

    plaintext = np.array(
        [
            [0],
            [1],
            [2],
            [3],
            [4],
            [5],
        ],
        dtype=np.uint8,
    )

    # Create real input H5 file
    with h5py.File(input_path, "w") as hf:
        hf.create_dataset("traces", data=traces)

        metadata = hf.create_group("metadata")
        metadata.create_dataset("plaintext", data=plaintext)

    # Run the real service
    service = TraceProcessingService(
        input_path=str(input_path),
        output_path=str(output_path),
        processor=OddTraceProcessor(),
    )

    service.run()

    # Check real output H5 file
    with h5py.File(output_path, "r") as hf:
        result = hf["traces"][:]

    expected = np.array(
        [
            [1, 1, 1],
            [3, 3, 3],
            [5, 5, 5],
        ],
        dtype=np.float32,
    )

    print(result)

    np.testing.assert_array_equal(result, expected)
