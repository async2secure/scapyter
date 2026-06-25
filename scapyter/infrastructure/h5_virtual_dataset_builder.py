import h5py
from pathlib import Path


class H5VirtualDatasetBuilder:
    def __init__(self, source_files: list[str]):
        if not source_files:
            raise ValueError("At least one source file is required.")

        self.source_files = [str(Path(f).resolve()) for f in source_files]
        # self.output_file = output_file

    def build(self, output_file: str):
        self._validate_sources()

        with h5py.File(self.source_files[0], "r") as first:
            trace_dtype = first["traces"].dtype
            sample_count = first["traces"].shape[1]

            metadata_info = {
                name: (
                    ds.shape[1:],
                    ds.dtype,
                )
                for name, ds in first["metadata"].items()
            }

        total_traces = self._total_trace_count()

        with h5py.File(output_file, "w", libver="latest") as out:
            # ---- traces ----
            traces_layout = h5py.VirtualLayout(
                shape=(total_traces, sample_count),
                dtype=trace_dtype,
            )

            offset = 0

            for file_path in self.source_files:
                with h5py.File(file_path, "r") as src:
                    n_traces = src["traces"].shape[0]

                traces_layout[offset : offset + n_traces] = h5py.VirtualSource(
                    file_path,
                    "traces",
                    shape=(n_traces, sample_count),
                )

                offset += n_traces

            out.create_virtual_dataset("traces", traces_layout)

            # ---- metadata ----
            metadata_group = out.create_group("metadata")

            for meta_name, (tail_shape, dtype) in metadata_info.items():
                layout = h5py.VirtualLayout(
                    shape=(total_traces, *tail_shape),
                    dtype=dtype,
                )

                offset = 0

                for file_path in self.source_files:
                    with h5py.File(file_path, "r") as src:
                        ds = src[f"metadata/{meta_name}"]
                        n_traces = ds.shape[0]

                    layout[offset : offset + n_traces] = h5py.VirtualSource(
                        file_path,
                        f"metadata/{meta_name}",
                        shape=(n_traces, *tail_shape),
                    )

                    offset += n_traces

                metadata_group.create_virtual_dataset(
                    meta_name,
                    layout,
                )

    def _total_trace_count(self) -> int:
        total = 0

        for file_path in self.source_files:
            with h5py.File(file_path, "r") as f:
                total += f["traces"].shape[0]

        return total

    def _validate_sources(self):
        with h5py.File(self.source_files[0], "r") as first:
            reference_sample_count = first["traces"].shape[1]

            reference_metadata = {
                name: (
                    ds.shape[1:],
                    ds.dtype,
                )
                for name, ds in first["metadata"].items()
            }

        for file_path in self.source_files[1:]:
            with h5py.File(file_path, "r") as f:
                if f["traces"].shape[1] != reference_sample_count:
                    raise ValueError(f"{file_path}: sample count mismatch.")

                current_metadata = {
                    name: (
                        ds.shape[1:],
                        ds.dtype,
                    )
                    for name, ds in f["metadata"].items()
                }

                if current_metadata != reference_metadata:
                    raise ValueError(f"{file_path}: metadata structure mismatch.")
