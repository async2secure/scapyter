import h5py

from domain.value_object import Batch


class ProjectFileWriter:
    def __init__(self, file_path: str, total_traces: int):
        self.file_path = file_path
        self.total_traces = total_traces
        self._batch_cursor = 0

    def save_batch(self, batch: Batch):
        with h5py.File(self.file_path, "a") as hf:
            # 1. Handle Traces
            if "traces" not in hf:
                sample_count = batch.traces.shape[1]
                # chunk_shape = (self.total_traces, 1000)
                hf.create_dataset(
                    "traces",
                    (self.total_traces, sample_count),
                    dtype="f4",
                    chunks=True,
                )

            # 2. Handle Dynamic Metadata (Plaintext, Ciphertext, Nonce, etc.)
            for meta_key, data in batch.metadata.items():
                # Map domain key to HDF5 path (e.g., 'nonce' -> 'metadata/nonce')
                path = f"metadata/{meta_key}"

                if path not in hf:
                    # Create dataset if first time seeing this metadata
                    width = data.shape[1] if len(data.shape) > 1 else 1
                    hf.create_dataset(path, (self.total_traces, width), dtype="u1")

                # Write the slice
                start = self._batch_cursor
                end = start + len(data)
                hf[path][start:end] = data

            # Write the traces slice
            hf["traces"][
                self._batch_cursor : self._batch_cursor + len(batch.traces)
            ] = batch.traces

            # Update cursor for the next batch
            self._batch_cursor += len(batch.traces)
