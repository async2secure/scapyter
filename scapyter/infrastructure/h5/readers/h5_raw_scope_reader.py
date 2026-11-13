import numpy as np

from scapyter.domain.value_object import Range, Batch
from scapyter.infrastructure.h5.readers.h5_project_file_reader import H5ProjectFileReader


class H5RawScopeReader(H5ProjectFileReader):
    def __init__(self, file_path: str, as_volts: bool = True):
        super().__init__(file_path)
        a = self._hf.attrs
        self.is_raw = a.get("trace_format", "") == "int16_raw"
        self.gain = float(a.get("vertical_gain", 1.0))
        self.offset = float(a.get("vertical_offset", 0.0))
        self.horiz_interval = float(a.get("horiz_interval", np.nan))
        self.as_volts = as_volts and self.is_raw

    def get_batch(self, trace_range: Range, sample_range: Range | None = None) -> Batch:
        b = super().get_batch(trace_range, sample_range)
        if self.as_volts:
            volts = b.traces.astype(np.float32) * np.float32(self.gain) - np.float32(self.offset)
            return Batch(indices=b.indices, traces=volts, metadata=b.metadata)
        return b
