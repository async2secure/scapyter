import numpy as np

from scapyter.domain.signal_processing.trace_processor import TraceProcessor
from scapyter.domain.value_object import Batch, Range


class SADResyncProcessor(TraceProcessor):
    """
    SAD based trace resynchronization using coarse-to-fine search.

    reference_range:
        The window from the reference trace to match.

    search_range:
        The region in each trace where the reference window is expected.

    coarse_step:
        Step size for initial search.

    fine_radius:
        Number of samples around coarse result to search precisely.
    """

    def __init__(
        self,
        reference_trace: np.ndarray,
        reference_range: Range,
        search_range: Range,
        coarse_step: int = 8,
        fine_radius: int = 16,
        pad_mode: str = "edge",
    ):
        self.reference_range = reference_range
        self.search_range = search_range
        self.coarse_step = coarse_step
        self.fine_radius = fine_radius
        self.pad_mode = pad_mode

        self._reference_window = reference_trace[
            reference_range.start : reference_range.end
        ].astype(np.float32)

        # Downsampled reference for coarse search
        self._reference_coarse = self._reference_window[::coarse_step]

    def process(self, batch: Batch) -> Batch:

        aligned = np.empty_like(batch.traces)

        for i, trace in enumerate(batch.traces):

            best_start = self._find_alignment(trace)

            shift = best_start - self.reference_range.start

            aligned[i] = self._shift_trace(
                trace,
                shift,
            )

        return Batch(
            indices=batch.indices,
            traces=aligned,
            metadata=batch.metadata,
        )

    def _find_alignment(
        self,
        trace: np.ndarray,
    ) -> int:

        window_size = self.reference_range.count

        #
        # 1. Coarse search
        #
        best_sad = np.inf
        best_start = self.search_range.start

        for start in range(
            self.search_range.start,
            self.search_range.end - window_size + 1,
            self.coarse_step,
        ):

            candidate = trace[start : start + window_size : self.coarse_step].astype(
                np.float32
            )

            sad = np.abs(candidate - self._reference_coarse).sum()

            if sad < best_sad:
                best_sad = sad
                best_start = start

        #
        # 2. Fine search around coarse result
        #
        fine_start = max(
            self.search_range.start,
            best_start - self.fine_radius,
        )

        fine_end = min(
            self.search_range.end - window_size,
            best_start + self.fine_radius,
        )

        best_sad = np.inf

        for start in range(
            fine_start,
            fine_end + 1,
        ):

            candidate = trace[start : start + window_size].astype(np.float32)

            sad = np.abs(candidate - self._reference_window).sum()

            if sad < best_sad:
                best_sad = sad
                best_start = start

        return best_start

    def _shift_trace(
        self,
        trace: np.ndarray,
        shift: int,
    ) -> np.ndarray:

        if shift == 0:
            return trace.copy()

        out = np.empty_like(trace)

        if shift > 0:

            out[:-shift] = trace[shift:]

            if self.pad_mode == "edge":
                out[-shift:] = trace[-1]
            else:
                out[-shift:] = 0

        else:

            shift = -shift

            out[shift:] = trace[:-shift]

            if self.pad_mode == "edge":
                out[:shift] = trace[0]
            else:
                out[:shift] = 0

        return out


#
# import numpy as np
#
# from scapyter.domain.signal_processing.trace_processor import TraceProcessor
# from scapyter.domain.value_object import Batch, Range
#
#
# class SADResyncProcessor(TraceProcessor):
#     """
#     Sum of Absolute Differences (SAD) trace resynchronization.
#
#     The processor extracts a reference window from a reference trace.
#     For every input trace, it searches for the window with the minimum
#     SAD inside `search_range`, then shifts the entire trace so that the
#     matched window aligns with the reference window.
#     """
#
#     def __init__(
#         self,
#         reference_trace: np.ndarray,
#         reference_range: Range,
#         search_range: Range,
#         pad_mode: str = "edge",  # "edge" or "zero"
#     ):
#         # self.reference_trace = reference_trace
#         self.reference_range = reference_range
#         self.search_range = search_range
#         self.pad_mode = pad_mode
#
#         # self._reference_window = reference_trace
#
#         self._reference_window = (
#             reference_trace[
#                 self.reference_range.start: self.reference_range.end
#             ].astype(np.float32)
#         )
#
#     # def _initialize(self, single_batch):
#     #     """
#     #     Called once before processing starts.
#     #     """
#     #     # batch = reader.get_single_batch(self.reference_trace)
#     #
#     #     self._reference_window = single_batch.trace[
#     #         self.reference_range.start : self.reference_range.end
#     #     ].astype(np.float32)
#
#     def process(self, batch: Batch) -> Batch:
#
#         if self._reference_window is None:
#             raise RuntimeError(
#                 "Processor has not been initialized. "
#                 "Call initialize(reader) before process()."
#             )
#
#         aligned = np.empty_like(batch.traces)
#
#         window_size = self.reference_range.count
#
#         for i, trace in enumerate(batch.traces):
#
#             best_sad = np.inf
#             best_start = self.search_range.start
#
#             # Slide the reference window across the search region
#             for start in range(
#                 self.search_range.start,
#                 self.search_range.end - window_size + 1,
#             ):
#
#                 candidate = trace[start : start + window_size]
#
#                 sad = np.abs(
#                     candidate.astype(np.float32) - self._reference_window
#                 ).sum()
#
#                 if sad < best_sad:
#                     best_sad = sad
#                     best_start = start
#
#             shift = best_start - self.reference_range.start
#
#             aligned[i] = self._shift_trace(trace, shift)
#
#         return Batch(
#             indices=batch.indices,
#             traces=aligned,
#             metadata=batch.metadata,
#         )
#
#     def _shift_trace(self, trace: np.ndarray, shift: int) -> np.ndarray:
#         """
#         Shift a trace without wraparound.
#
#         Positive shift:
#             matched window is to the right
#             -> move trace left
#
#         Negative shift:
#             matched window is to the left
#             -> move trace right
#         """
#
#         if shift == 0:
#             return trace.copy()
#
#         out = np.empty_like(trace)
#
#         if shift > 0:
#             # shift left
#             out[:-shift] = trace[shift:]
#
#             if self.pad_mode == "edge":
#                 out[-shift:] = trace[-1]
#             else:
#                 out[-shift:] = 0
#
#         else:
#             shift = -shift
#
#             # shift right
#             out[shift:] = trace[:-shift]
#
#             if self.pad_mode == "edge":
#                 out[:shift] = trace[0]
#             else:
#                 out[:shift] = 0
#
#         return out
