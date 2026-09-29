import numpy as np
import pytest
from scipy import stats
from unittest.mock import MagicMock

from scapyter.domain.analysis.tvla.calculator import TvlaCalculator
from scapyter.domain.analysis.tvla.service import TvlaService, detect_fixed_value
from scapyter.domain.value_object import Range, RangeParameters

FIXED_PT = np.arange(16, dtype=np.uint8)


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #
def make_dataset(n=200, samples=5, seed=0, is_fixed=None, leak=0.5):
    """
    Random (non-alternating) fixed/random sequence. Returns
    (traces, plaintexts, is_fixed). Fixed traces get a small leakage.
    """
    rng = np.random.default_rng(seed)
    if is_fixed is None:
        is_fixed = rng.random(n) < 0.5
    is_fixed = np.asarray(is_fixed, dtype=bool)
    n = len(is_fixed)

    plaintexts = rng.integers(0, 256, size=(n, 16), dtype=np.uint8)
    plaintexts[is_fixed] = FIXED_PT

    traces = rng.normal(0.0, 1.0, size=(n, samples))
    traces[is_fixed] += leak
    return traces, plaintexts, is_fixed


def make_service(traces, plaintexts, trace_range=None, **kwargs):
    """
    TvlaService whose repository serves slices of the arrays, like a real
    reader, so batching and metadata alignment are exercised.
    """
    trace_count, sample_count = traces.shape
    params = RangeParameters(
        trace_range=trace_range or Range(0, trace_count),
        sample_range=Range(0, sample_count),
    )

    repo = MagicMock()
    repo.get_batch.side_effect = lambda batch_range, sample_range: MagicMock(
        traces=traces[
            batch_range.start : batch_range.end,
            sample_range.start : sample_range.end,
        ],
        metadata={"plaintext": plaintexts[batch_range.start : batch_range.end]},
    )
    repo.get_metadata.side_effect = (
        lambda name, trace_range: plaintexts[trace_range.start : trace_range.end]
    )
    return TvlaService(repo, params, **kwargs), repo


def expected_t(traces, is_fixed):
    """Reference Welch t-test: fixed minus random."""
    data = traces.astype(np.float64)
    return stats.ttest_ind(data[is_fixed], data[~is_fixed], equal_var=False).statistic


@pytest.fixture
def dataset():
    return make_dataset()


# --------------------------------------------------------------------------- #
# Correctness of the t-scores
# --------------------------------------------------------------------------- #
def test_t_scores_match_scipy(dataset):
    traces, plaintexts, is_fixed = dataset
    service, _ = make_service(traces, plaintexts)

    result = service.calculate(batch_size=200)

    assert np.allclose(result.t_scores, expected_t(traces, is_fixed))


def test_groups_follow_plaintext_not_trace_position(dataset):
    """Regression: the fixed/random sequence is not alternating."""
    traces, plaintexts, is_fixed = dataset
    service, _ = make_service(traces, plaintexts)

    service.calculate(batch_size=200)

    assert service._count_fixed == is_fixed.sum()
    assert service._count_random == (~is_fixed).sum()
    assert service._count_fixed != service._count_random


@pytest.mark.parametrize("batch_size", [1, 3, 7, 50, 200, 1000])
def test_result_is_independent_of_batch_size(dataset, batch_size):
    traces, plaintexts, is_fixed = dataset
    service, _ = make_service(traces, plaintexts)

    result = service.calculate(batch_size=batch_size)

    assert np.allclose(result.t_scores, expected_t(traces, is_fixed))


def test_large_offset_does_not_lose_precision():
    """Regression: sumsq - sum^2/n cancels badly with a big DC offset."""
    offset = 1e7
    traces, plaintexts, is_fixed = make_dataset(n=2000, samples=4, seed=1)
    traces = traces + offset
    service, _ = make_service(traces, plaintexts)

    result = service.calculate(batch_size=500)

    assert np.all(np.isfinite(result.t_scores))
    # t is shift-invariant; an exact shift gives an accurate reference.
    assert np.allclose(
        result.t_scores, expected_t(traces - offset, is_fixed), rtol=1e-9
    )


def test_integer_traces_do_not_overflow():
    """Regression: np.square on int8 overflows before the sum."""
    rng = np.random.default_rng(2)
    _, plaintexts, is_fixed = make_dataset(n=100, samples=3, seed=2)
    traces = rng.integers(100, 127, size=(100, 3), dtype=np.int8)
    service, _ = make_service(traces, plaintexts)

    result = service.calculate(batch_size=100)

    assert np.allclose(result.t_scores, expected_t(traces, is_fixed))


def test_constant_sample_point_gives_zero_not_nan(dataset):
    traces, plaintexts, _ = dataset
    traces = traces.copy()
    traces[:, 1] = 42.0  # zero variance in both groups
    service, _ = make_service(traces, plaintexts)

    result = service.calculate(batch_size=50)

    assert result.t_scores[1] == 0.0
    assert np.all(np.isfinite(result.t_scores))


def test_calculate_twice_gives_same_result(dataset):
    traces, plaintexts, _ = dataset
    service, _ = make_service(traces, plaintexts)

    first = service.calculate(batch_size=50)
    second = service.calculate(batch_size=50)

    assert np.array_equal(first.t_scores, second.t_scores)


def test_result_is_a_snapshot(dataset):
    """Later updates must not mutate a result that was already returned."""
    traces, plaintexts, _ = dataset
    service, _ = make_service(traces, plaintexts)

    result = service.calculate(batch_size=50)
    before = result.t_scores.copy()
    service.update(Range(0, 10))

    assert np.array_equal(result.t_scores, before)


def test_incremental_updates_match_single_pass(dataset):
    traces, plaintexts, is_fixed = dataset
    service, _ = make_service(traces, plaintexts)

    service.update(Range(0, 50))
    service.update(Range(50, 121), batch_size=7)
    service.update(Range(121, 200))

    assert np.allclose(service.get_results(), expected_t(traces, is_fixed))


def test_update_order_does_not_matter(dataset):
    traces, plaintexts, is_fixed = dataset
    service, _ = make_service(traces, plaintexts)

    service.update(Range(100, 200))
    service.update(Range(0, 100))

    assert np.allclose(service.get_results(), expected_t(traces, is_fixed))


# --------------------------------------------------------------------------- #
# Result metadata and repository interaction
# --------------------------------------------------------------------------- #
def test_calculate_result_metadata_and_sample_range(dataset):
    traces, plaintexts, _ = dataset
    service, repo = make_service(traces, plaintexts)

    result = service.calculate(batch_size=200)

    assert result.trace_start == 0
    assert result.trace_end == 200
    assert result.sample_start == 0
    assert result.sample_end == 5

    repo.get_batch.assert_called_once()
    _, kwargs = repo.get_batch.call_args
    assert kwargs["sample_range"] == Range(0, 5)


# --------------------------------------------------------------------------- #
# Fixed value detection
# --------------------------------------------------------------------------- #
def test_fixed_value_is_detected_once_over_configured_range(dataset):
    traces, plaintexts, _ = dataset
    service, repo = make_service(traces, plaintexts)

    service.calculate(batch_size=20)  # many batches

    repo.get_metadata.assert_called_once()
    args, _ = repo.get_metadata.call_args
    assert args == ("plaintext", Range(0, 200))
    assert np.array_equal(service._fixed_value, FIXED_PT)


def test_explicit_fixed_value_skips_detection(dataset):
    traces, plaintexts, is_fixed = dataset
    service, repo = make_service(traces, plaintexts, fixed_value=FIXED_PT)

    result = service.calculate(batch_size=200)

    repo.get_metadata.assert_not_called()
    assert np.allclose(result.t_scores, expected_t(traces, is_fixed))


def test_scalar_metadata_column_is_supported():
    """1-D metadata (one value per trace) works as well as (n, 16) rows."""
    rng = np.random.default_rng(9)
    is_fixed = rng.random(100) < 0.5
    values = rng.integers(0, 2**31, size=100)
    values[is_fixed] = 12345
    traces = rng.normal(size=(100, 3))
    service, _ = make_service(traces, values)

    result = service.calculate(batch_size=30)

    assert np.allclose(result.t_scores, expected_t(traces, is_fixed))


def test_detect_fixed_value_finds_the_mode(dataset):
    _, plaintexts, _ = dataset
    assert np.array_equal(detect_fixed_value(plaintexts), FIXED_PT)


def test_detect_fixed_value_raises_when_nothing_dominates():
    rng = np.random.default_rng(10)
    all_random = rng.integers(0, 256, size=(100, 16), dtype=np.uint8)

    with pytest.raises(ValueError, match="No dominant fixed value"):
        detect_fixed_value(all_random)


def test_detect_fixed_value_raises_on_empty_column():
    with pytest.raises(ValueError, match="empty"):
        detect_fixed_value(np.empty((0, 16), dtype=np.uint8))


def test_missing_metadata_field_raises(dataset):
    traces, plaintexts, _ = dataset
    service, _ = make_service(traces, plaintexts, fixed_value=FIXED_PT)
    service._group_field = "ciphertext"

    with pytest.raises(ValueError, match="ciphertext"):
        service.calculate()


# --------------------------------------------------------------------------- #
# Empty / degenerate input
# --------------------------------------------------------------------------- #
def test_empty_range_raises_value_error():
    repo = MagicMock()
    params = RangeParameters(
        trace_range=Range(0, 0),
        sample_range=Range(0, 10),
    )
    service = TvlaService(repo, params)

    with pytest.raises(
        ValueError, match="Welch's t-test requires at least two traces"
    ):
        service.calculate()

    assert repo.get_batch.called is False
    assert repo.get_metadata.called is False
    assert service._count_fixed == 0
    assert service._count_random == 0


def test_single_trace_per_group_raises():
    traces, plaintexts, _ = make_dataset(is_fixed=[True, False])
    service, _ = make_service(traces, plaintexts, fixed_value=FIXED_PT)

    with pytest.raises(ValueError):
        service.calculate()


# --------------------------------------------------------------------------- #
# Trace progression
# --------------------------------------------------------------------------- #
def test_progression_matches_scipy_at_every_step(dataset):
    traces, plaintexts, is_fixed = dataset
    service, _ = make_service(traces, plaintexts)

    results = service.calculate_trace_progression(step_size=40, batch_size=25)

    assert [r.trace_end for r in results] == [40, 80, 120, 160, 200]
    for result in results:
        end = result.trace_end
        assert result.trace_start == 0
        assert np.allclose(
            result.t_scores, expected_t(traces[:end], is_fixed[:end])
        )


def test_progression_includes_partial_final_step():
    traces, plaintexts, is_fixed = make_dataset(
        seed=5, samples=2, is_fixed=[True, False] * 5
    )
    service, _ = make_service(traces, plaintexts)

    results = service.calculate_trace_progression(step_size=4, batch_size=4)

    assert [r.trace_end for r in results] == [4, 8, 10]
    assert np.allclose(results[-1].t_scores, expected_t(traces, is_fixed))


def test_progression_skips_steps_with_too_few_traces():
    traces, plaintexts, _ = make_dataset(
        seed=6, samples=2, is_fixed=[True, False] * 4
    )
    service, _ = make_service(traces, plaintexts)

    # After 2 traces each group has 1 trace, so that step is skipped.
    results = service.calculate_trace_progression(step_size=2, batch_size=2)

    assert [r.trace_end for r in results] == [4, 6, 8]


def test_progression_skips_until_both_groups_have_two():
    """Group sizes are uneven here: fixed reaches 2 traces only at step 8."""
    is_fixed = [False, False, False, False, False, False, True, True]
    traces, plaintexts, _ = make_dataset(seed=7, samples=2, is_fixed=is_fixed)
    service, _ = make_service(traces, plaintexts, fixed_value=FIXED_PT)

    results = service.calculate_trace_progression(step_size=2, batch_size=2)

    assert [r.trace_end for r in results] == [8]


def test_progression_step_size_one_does_not_raise():
    traces, plaintexts, _ = make_dataset(
        seed=7, samples=2, is_fixed=[True, False] * 3
    )
    service, _ = make_service(traces, plaintexts)

    results = service.calculate_trace_progression(step_size=1)

    assert [r.trace_end for r in results] == [4, 5, 6]


@pytest.mark.parametrize("step_size", [0, -5])
def test_progression_rejects_non_positive_step(dataset, step_size):
    traces, plaintexts, _ = dataset
    service, _ = make_service(traces, plaintexts)

    with pytest.raises(ValueError, match="step_size must be greater than 0"):
        service.calculate_trace_progression(step_size=step_size)


# --------------------------------------------------------------------------- #
# Calculator (unit)
# --------------------------------------------------------------------------- #
def test_calculator_matches_scipy():
    rng = np.random.default_rng(8)
    a = rng.normal(0.0, 1.0, size=(50, 3))
    b = rng.normal(0.3, 2.0, size=(70, 3))

    t = TvlaCalculator.calculate_welch_t_test(
        a.sum(axis=0), np.square(a).sum(axis=0), len(a),
        b.sum(axis=0), np.square(b).sum(axis=0), len(b),
    )

    assert np.allclose(t, stats.ttest_ind(a, b, equal_var=False).statistic)


def test_calculator_zero_variance_returns_zero_even_when_means_differ():
    a = np.full((10, 1), 5.0)
    b = np.full((10, 1), 7.0)

    t = TvlaCalculator.calculate_welch_t_test(
        a.sum(axis=0), np.square(a).sum(axis=0), 10,
        b.sum(axis=0), np.square(b).sum(axis=0), 10,
    )

    assert np.array_equal(t, [0.0])