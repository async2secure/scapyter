from scapyter.domain.tvla.tval_service import TvlaService
from scapyter.domain.value_object import Range


class TvlaIncrementalEvaluator:
    def __init__(self, service: TvlaService):
        self._service = service
        # Extract the start and end from the service's existing parameters
        self._total_range = service._range_parameters.trace_range
        self._current_pos = self._total_range.start

    def run_steps(self, step_size: int) -> dict[int, float]:
        """
        Iterates through the repository using the specified step size.
        Returns a dictionary mapping trace_count to max_t_score.
        """
        results = {}

        # Calculate our stopping points
        # e.g., if range is 0-500 and step is 5 -> 5, 10, 15... 500
        steps = range(
            self._current_pos + step_size, self._total_range.end + 1, step_size
        )

        for end_val in steps:
            # Update only the new slice
            delta_range = Range(self._current_pos, end_val)
            self._service.update(delta_range)

            # Record the result
            score = self._service.run_max_t()
            results[end_val] = score

            # Advance pointer
            self._current_pos = end_val

            print(f"[Stepper] Processed to {end_val} traces | Max T: {score:.4f}")

        return results
