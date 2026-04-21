from scapyter.domain.value_object import Range


def get_progress_batch(
    *, batch_size: int, progress_steps: int, trace_range: Range
) -> tuple[list[int], list[Range]]:
    if not trace_range.is_valid:
        return [], []

    # 1. Calculate milestone points
    # We use trace.start + progress_steps to avoid a split at the very beginning
    progress_markers = list(
        range(trace_range.start + progress_steps, trace_range.end, progress_steps)
    )

    # 2. Gather all unique boundary points
    # Using a set handles overlapping batch boundaries and progress steps automatically
    boundaries = {trace_range.start, trace_range.end}
    boundaries.update(range(trace_range.start, trace_range.end, batch_size))
    boundaries.update(progress_markers)

    # 3. Create the list of Range objects
    sorted_points = sorted(boundaries)
    batch_ranges = [
        Range(sorted_points[i], sorted_points[i + 1])
        for i in range(len(sorted_points) - 1)
    ]

    # Include the final trace.end in the progress markers if that matches your needs
    full_progress_steps = sorted(list(set(progress_markers) | {trace_range.end}))

    return full_progress_steps, batch_ranges
