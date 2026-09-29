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
        mean_a = group_a_sum / count_a
        mean_b = group_b_sum / count_b

        # Sample variance (Bessel's correction). Clamp tiny negative values
        # caused by floating point error.
        var_a = np.maximum(
            (group_a_sq_sum - group_a_sum**2 / count_a) / (count_a - 1), 0.0
        )
        var_b = np.maximum(
            (group_b_sq_sum - group_b_sum**2 / count_b) / (count_b - 1), 0.0
        )

        numerator = mean_a - mean_b
        denominator = np.sqrt(var_a / count_a + var_b / count_b)

        # Zero-variance points (constant/saturated samples) get t = 0.
        t = np.zeros_like(denominator)
        np.divide(numerator, denominator, out=t, where=denominator > 0)
        return t