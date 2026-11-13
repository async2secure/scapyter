import h5py

from scapyter.infrastructure.h5.writers.h5_project_file_writer import H5ProjectFileWriter


class H5RawScopeWriter(H5ProjectFileWriter):
    """Stores int16 ADC codes + scaling attributes."""
    TRACE_DTYPE = "i2"

    def set_scaling(self, gain, offset, horiz_interval):
        with h5py.File(self.file_path, "a") as hf:
            hf.attrs["vertical_gain"] = gain
            hf.attrs["vertical_offset"] = offset
            hf.attrs["horiz_interval"] = horiz_interval
            hf.attrs["adc_bits"] = 12
            hf.attrs["trace_format"] = "int16_raw"   # lets the reader detect it

    def save_batch(self, batch):
        if batch.traces.dtype != "<i2" and batch.traces.dtype != "int16":
            raise TypeError(f"expected int16 codes, got {batch.traces.dtype}")
        super().save_batch(batch)