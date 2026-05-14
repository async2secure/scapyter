import numpy as np


class TvlaCalculator:

    @staticmethod
    def calculate_welch_t_test(
        group_a_sum: np.ndarray,
        group_a_sq_sum: np.ndarray,
        count_a: int,
        group_b_sum: np.ndarray,
        group_b_sq_sum: np.ndarray,
        count_b: int,
    ) -> np.ndarray:
        # Calculate Means
        mean_a = group_a_sum / count_a
        mean_b = group_b_sum / count_b

        # Calculate Sample Variances (with Bessel's Correction: N-1)
        # Var = (SumSq - (Sum^2 / N)) / (N - 1)
        var_a = (group_a_sq_sum - (group_a_sum**2 / count_a)) / (count_a - 1)
        var_b = (group_b_sq_sum - (group_b_sum**2 / count_b)) / (count_b - 1)

        # Welch's T-Test Formula
        numerator = mean_a - mean_b
        denominator = np.sqrt((var_a / count_a) + (var_b / count_b))

        return numerator / denominator
