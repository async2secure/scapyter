import os
import h5py

from scapyter.domain.repository.project_file_reader import ProjectFileReader
from scapyter.domain.value_object import Range, Batch


class H5ProjectFileReader(ProjectFileReader):
    TAG_ALIASES = {
        "plaintext": ["plaintext", "plain_text"],
        "ciphertext": ["ciphertext", "cipher_text"],
        "key": ["key", "secret_key"],
    }

    def __init__(self, file_path: str):
        self.file_path = file_path
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"Trace file not found: {file_path}")

        self._hf = h5py.File(file_path, "r")
        self._validate_structure()
        self._map = self._build_metadata_map()

    def _validate_structure(self):
        if "traces" not in self._hf or "metadata" not in self._hf:
            self.close()
            raise ValueError(
                "Invalid HDF5 format: Missing 'traces' or 'metadata' groups."
            )

    def _build_metadata_map(self) -> dict[str, h5py.Dataset]:
        """Maps logical names to HDF5 datasets."""
        mapping = {}
        meta_group = self._hf["metadata"]
        available_keys = list(meta_group.keys())

        # Map Aliases
        for logical, aliases in self.TAG_ALIASES.items():
            found = next((a for a in aliases if a in available_keys), None)
            if found:
                mapping[logical] = meta_group[found]

        # Map anything else as-is
        for key in available_keys:
            if key not in mapping.values():
                mapping[key] = meta_group[key]

        return mapping

    def get_single_batch(self, index: int, sample_range: Range | None = None) -> Batch:
        """The core Repository method: Returns a Domain Entity."""
        return self.get_batch(
            trace_range=Range(index, index + 1),
            sample_range=sample_range,
        )

    def get_batch(
        self,
        trace_range: Range,
        sample_range: Range | None = None,
    ) -> Batch:
        """
        High-performance bulk loader.
        Requests data in contiguous blocks to minimize HDF5 I/O overhead.
        """
        # 1. Bulk read traces: One I/O operation instead of N

        if sample_range is None:
            sample_range = Range(0, self.sample_count)

        samples_block = self._hf["traces"][
            trace_range.start : trace_range.end, sample_range.start : sample_range.end
        ]

        # 2. Bulk read all metadata
        metadata_block = {}
        for name, dataset in self._map.items():
            # metadata_block['plaintext'] = dataset[start:end]
            metadata_block[name] = dataset[trace_range.start : trace_range.end]

        return Batch(
            indices=range(trace_range.start, trace_range.end),
            traces=samples_block,
            metadata=metadata_block,
        )

    @property
    def trace_count(self) -> int:
        return self._hf["traces"].shape[0]

    @property
    def sample_count(self) -> int:
        return self._hf["traces"].shape[1]

    def close(self):
        self._hf.close()

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()
