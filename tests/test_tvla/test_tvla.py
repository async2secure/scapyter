import numpy as np

from scapyter.domain.tvla.tvla_calculator import TvlaCalculator


from scipy import stats


def test_compare_with_scipy():
    data_a = np.random.normal(10, 2, 100)
    data_b = np.random.normal(12, 2, 100)

    # Calculate inputs for your function
    sum_a = np.sum(data_a)
    sq_sum_a = np.sum(data_a**2)
    sum_b = np.sum(data_b)
    sq_sum_b = np.sum(data_b**2)

    # Your result
    my_t = TvlaCalculator.calculate_welch_t_test(
        np.array([sum_a]),
        np.array([sq_sum_a]),
        100,
        np.array([sum_b]),
        np.array([sq_sum_b]),
        100,
    )

    # SciPy result (equal_var=False triggers Welch's)
    scipy_t, _ = stats.ttest_ind(data_a, data_b, equal_var=False)

    assert np.isclose(my_t[0], scipy_t)
