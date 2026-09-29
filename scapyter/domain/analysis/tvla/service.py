import numpy as np
from tqdm import tqdm

from scapyter.domain.analysis.tvla.calculator import TvlaCalculator
from scapyter.domain.progress_range.progress_range import get_progress_batch
from scapyter.domain.repository.project_file_reader import ProjectFileReader
from scapyter.domain.tvla.value_objects import TvlaResult
from scapyter.domain.value_object import Range, RangeParameters


def detect_fixed_value(column: np.ndarray, min_fraction: float = 0.1) -> np.ndarray:
    """
    Finds the fixed value in a fixed-vs-random metadata column (e.g. plaintext).

    In a fixed-vs-random set the fixed value repeats in a large share of the
    traces, while each random value appears about once, so the most frequent
    row is the fixed one.

    Raises:
        ValueError: if the column is empty or no value is frequent enough.
    """
    column = np.asarray(column)
    if len(column) == 0:
        raise ValueError("Cannot detect the fixed value in an empty column.")

    values, counts = np.unique(column, axis=0, return_counts=True)
    top = counts.argmax()
    if counts[top] < min_fraction * len(column):
        raise ValueError(
            "No dominant fixed value found "
            f"(most common value covers {counts[top]}/{len(column)} traces). "
            "This may not be a fixed-vs-random data set."
        )
    return values[top]


class TvlaService:
    """
    Fixed-vs-random TVLA. A trace belongs to the fixed group when its
    `group_field` metadata (plaintext by default) equals the fixed value;
    every other trace belongs to the random group.
    """

    def __init__(
        self,
        project_file_reader: ProjectFileReader,
        range_parameters: RangeParameters,
        group_field: str = "plaintext",
        fixed_value: np.ndarray | None = None,
    ):
        self._project_file_reader = project_file_reader
        self._range_parameters = range_parameters
        self._group_field = group_field
        self._fixed_value = None if fixed_value is None else np.asarray(fixed_value)

        sample_count = range_parameters.sample_count

        # Running sums for Welch's t-test. Values are accumulated after
        # subtracting a fixed reference vector (self._shift), which leaves
        # variances and mean differences unchanged but avoids cancellation.
        self._acc_fixed = np.zeros(sample_count, dtype=np.float64)
        self._acc_fixed_sq = np.zeros(sample_count, dtype=np.float64)
        self._acc_random = np.zeros(sample_count, dtype=np.float64)
        self._acc_random_sq = np.zeros(sample_count, dtype=np.float64)

        self._count_fixed = 0
        self._count_random = 0
        self._shift: np.ndarray | None = None

    # ------------------------------------------------------------------ #
    # Public API
    # ------------------------------------------------------------------ #
    def calculate(self, batch_size: int = 50) -> TvlaResult:
        """Calculates TVLA over the configured trace range."""
        self._reset()

        trace_range = self._range_parameters.trace_range
        self.update(trace_range, batch_size=batch_size, show_progress=True)

        return self._create_result(
            trace_start=trace_range.start,
            trace_end=trace_range.end,
        )

    def calculate_trace_progression(
        self,
        step_size: int,
        batch_size: int = 50,
    ) -> list[TvlaResult]:
        """
        Calculates TVLA incrementally over the configured trace range.

        With a range of 0-500 and step_size 100, returns results for
        100, 200, 300, 400, 500. If the range length is not a multiple of
        step_size, a final result at trace_range.end is appended.

        Steps where either group has fewer than two traces are skipped,
        since Welch's t-test is undefined there.
        """
        if step_size <= 0:
            raise ValueError("step_size must be greater than 0")

        self._reset()

        trace_range = self._range_parameters.trace_range
        current_pos = trace_range.start

        end_points = list(
            range(trace_range.start + step_size, trace_range.end + 1, step_size)
        )
        if trace_range.is_valid and (
            not end_points or end_points[-1] != trace_range.end
        ):
            end_points.append(trace_range.end)

        results: list[TvlaResult] = []

        for end_val in tqdm(
            end_points,
            desc="Running Progressive TVLA",
            unit="step",
        ):
            self.update(
                Range(current_pos, end_val),
                batch_size=batch_size,
                show_progress=False,
            )
            current_pos = end_val

            if self._count_fixed < 2 or self._count_random < 2:
                continue

            results.append(
                self._create_result(
                    trace_start=trace_range.start,
                    trace_end=end_val,
                )
            )

        return results

    def update(
        self,
        trace_range: Range,
        batch_size: int = 50,
        show_progress: bool = False,
    ) -> None:
        """
        Adds a range of traces to the accumulated TVLA state.
        State is preserved between calls; ranges may be fed in any order
        because group membership comes from each trace's own metadata.
        """
        _, batch_range_list = get_progress_batch(
            batch_size=batch_size,
            progress_steps=trace_range.end - trace_range.start,
            trace_range=trace_range,
        )

        sample_range = self._range_parameters.sample_range

        batch_iterator = batch_range_list
        if show_progress:
            batch_iterator = tqdm(
                batch_range_list,
                total=len(batch_range_list),
                desc="Calculating TVLA",
                unit="batch",
            )

        for batch_range in batch_iterator:
            fixed_value = self._resolve_fixed_value()

            batch = self._project_file_reader.get_batch(
                batch_range,
                sample_range=sample_range,
            )

            # Cast first: squaring integer scope data would overflow.
            traces = np.asarray(batch.traces).astype(np.float64, copy=False)
            if len(traces) == 0:
                continue

            is_fixed = self._group_mask(batch.metadata, fixed_value, len(traces))

            if self._shift is None:
                self._shift = traces.mean(axis=0)
            traces = traces - self._shift

            fixed_traces = traces[is_fixed]
            random_traces = traces[~is_fixed]

            self._acc_fixed += fixed_traces.sum(axis=0)
            self._acc_fixed_sq += np.square(fixed_traces).sum(axis=0)
            self._count_fixed += len(fixed_traces)

            self._acc_random += random_traces.sum(axis=0)
            self._acc_random_sq += np.square(random_traces).sum(axis=0)
            self._count_random += len(random_traces)

    def get_results(self) -> np.ndarray:
        """
        Current Welch's t-scores (fixed minus random).

        Raises:
            ValueError: if either group has fewer than two traces.
        """
        if self._count_fixed < 2 or self._count_random < 2:
            raise ValueError(
                "Welch's t-test requires at least two traces "
                "in each group. "
                f"Got fixed={self._count_fixed}, "
                f"random={self._count_random}."
            )

        return TvlaCalculator.calculate_welch_t_test(
            self._acc_fixed,
            self._acc_fixed_sq,
            self._count_fixed,
            self._acc_random,
            self._acc_random_sq,
            self._count_random,
        )

    # ------------------------------------------------------------------ #
    # Internals
    # ------------------------------------------------------------------ #
    def _resolve_fixed_value(self) -> np.ndarray:
        """Detects the fixed value once, over the configured trace range."""
        if self._fixed_value is None:
            column = self._project_file_reader.get_metadata(
                self._group_field,
                self._range_parameters.trace_range,
            )
            self._fixed_value = detect_fixed_value(column)
        return self._fixed_value

    def _group_mask(
        self,
        metadata: dict,
        fixed_value: np.ndarray,
        trace_count: int,
    ) -> np.ndarray:
        if self._group_field not in metadata:
            raise ValueError(
                f"Metadata field '{self._group_field}' not found. "
                f"Available: {sorted(metadata)}"
            )

        labels = np.asarray(metadata[self._group_field])
        if len(labels) != trace_count:
            raise ValueError(
                f"Metadata '{self._group_field}' has {len(labels)} rows "
                f"for {trace_count} traces."
            )

        is_fixed = labels == fixed_value
        if is_fixed.ndim > 1:
            is_fixed = is_fixed.reshape(len(labels), -1).all(axis=1)
        return is_fixed

    def _create_result(self, trace_start: int, trace_end: int) -> TvlaResult:
        sample_range = self._range_parameters.sample_range
        return TvlaResult(
            t_scores=self.get_results(),
            trace_start=trace_start,
            trace_end=trace_end,
            sample_start=sample_range.start,
            sample_end=sample_range.end,
        )

    def _reset(self) -> None:
        self._acc_fixed.fill(0)
        self._acc_fixed_sq.fill(0)
        self._acc_random.fill(0)
        self._acc_random_sq.fill(0)
        self._count_fixed = 0
        self._count_random = 0
        self._shift = None