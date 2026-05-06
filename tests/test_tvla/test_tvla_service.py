import pytest
import numpy as np
from unittest.mock import MagicMock

from scapyter.domain.tvla.tval_service import TvlaService
from scapyter.domain.value_object import Range, RangeParameters


@pytest.fixture
def mock_repo():
    return MagicMock()


@pytest.fixture
def range_params():
    # Setup for 4 traces, each with 3 samples
    return RangeParameters(trace_range=Range(0, 4), trace_sample_range=Range(0, 3))


def test_tvla_service_run_accumulates_correctly(mock_repo, range_params):
    # 1. Arrange: Create dummy traces
    # Trace 0, 2 are "even", Trace 1, 3 are "odd"
    # traces[::2] -> [[1, 1, 1], [3, 3, 3]]
    # traces[1::2] -> [[2, 2, 2], [4, 4, 4]]
    traces = np.array(
        [
            [1.0, 1.0, 1.0],  # Index 0 (Even)
            [2.0, 2.0, 2.0],  # Index 1 (Odd)
            [3.0, 3.0, 3.0],  # Index 2 (Even)
            [4.0, 4.0, 4.0],  # Index 3 (Odd)
        ]
    )

    # Mock the return value of repository.get_batch().traces
    mock_batch = MagicMock()
    mock_batch.traces = traces
    mock_repo.get_batch.return_value = mock_batch

    # 2. Act
    service = TvlaService(mock_repo, range_params)
    # Use batch_size=4 to process everything in one go for simplicity
    results = service.run(batch_size=4)

    # 3. Assert
    # Let's verify internal accumulation state before the final T-test
    # Even sums: 1+3 = 4 for each sample point
    # Odd sums: 2+4 = 6 for each sample point
    assert np.all(service._acc_even == np.array([4.0, 4.0, 4.0]))
    assert np.all(service._acc_odd == np.array([6.0, 6.0, 6.0]))
    assert service._count_even == 2
    assert service._count_odd == 2

    # Verify that the repository was called with correct slices
    mock_repo.get_batch.assert_called_once()
    # Check if sample_slice correctly used the RangeParameters (0, 3)
    args, kwargs = mock_repo.get_batch.call_args
    assert kwargs["sample_slice"] == slice(0, 3)


def test_tvla_service_empty_range_handling(mock_repo):
    """Test behavior when trace range is invalid/empty."""
    params = RangeParameters(Range(0, 0), Range(0, 10))
    service = TvlaService(mock_repo, params)

    # Depending on how TvlaCalculator handles 0 count,
    # this might return NaNs or zeros.
    results = service.run()

    assert mock_repo.get_batch.called is False
    assert np.all(np.isnan(results))


def test_tvla_service_aggregates_even_odd_correctly(mock_repo):
    # Setup: 4 traces, 2 samples each
    # Even indices (0, 2): [1, 1] and [3, 3] -> Sum: [4, 4], SqSum: [1+9, 1+9] = [10, 10]
    # Odd indices (1, 3):  [2, 2] and [4, 4] -> Sum: [6, 6], SqSum: [4+16, 4+16] = [20, 20]
    traces = np.array(
        [[1.0, 1.0], [2.0, 2.0], [3.0, 3.0], [4.0, 4.0]]  # 0  # 1  # 2  # 3
    )

    params = RangeParameters(trace_range=Range(0, 4), trace_sample_range=Range(0, 2))

    # Mock the repository to return our traces
    mock_batch = MagicMock()
    mock_batch.traces = traces
    mock_repo.get_batch.return_value = mock_batch

    service = TvlaService(mock_repo, params)
    service.run(batch_size=4)

    # Assert Sums
    assert np.array_equal(service._acc_even, [4.0, 4.0])
    assert np.array_equal(service._acc_odd, [6.0, 6.0])

    # Assert Square Sums
    assert np.array_equal(service._acc_even_sq, [10.0, 10.0])
    assert np.array_equal(service._acc_odd_sq, [20.0, 20.0])

    # Assert Counts
    assert service._count_even == 2
    assert service._count_odd == 2


def test_tvla_service_with_odd_batch_size(mock_repo):
    """
    Ensures that traces are correctly categorized as even/odd based on
    their global index, even if the batch size is odd.
    """
    # 1. Setup Range: 6 traces total, batch size 3
    # Batch 1: Indices 0, 1, 2
    # Batch 2: Indices 3, 4, 5
    params = RangeParameters(trace_range=Range(0, 6), trace_sample_range=Range(0, 1))

    # We create unique values for each trace to track where they go
    # Trace Index:  0    1    2    3    4    5
    # Value:       [0]  [1]  [2]  [3]  [4]  [5]
    batch_1_traces = np.array([[0.0], [1.0], [2.0]])
    batch_2_traces = np.array([[3.0], [4.0], [5.0]])

    # Mock the repo to return batch 1 then batch 2
    mock_repo.get_batch.side_effect = [
        MagicMock(traces=batch_1_traces),
        MagicMock(traces=batch_2_traces),
    ]

    service = TvlaService(mock_repo, params)

    # 2. Run the service with batch_size=3
    service.run(batch_size=3)

    # 3. Assertions
    # Global Even Indices: 0, 2, 4 -> Sum should be 0 + 2 + 4 = 6
    # Global Odd Indices:  1, 3, 5 -> Sum should be 1 + 3 + 5 = 9

    expected_even_sum = 6.0
    expected_odd_sum = 9.0

    assert (
        service._count_even == 3
    ), f"Expected 3 even traces, got {service._count_even}"
    assert service._count_odd == 3, f"Expected 3 odd traces, got {service._count_odd}"

    assert np.isclose(
        service._acc_even[0], expected_even_sum
    ), f"Even accumulation failed. Expected {expected_even_sum}, got {service._acc_even[0]}"

    assert np.isclose(
        service._acc_odd[0], expected_odd_sum
    ), f"Odd accumulation failed. Expected {expected_odd_sum}, got {service._acc_odd[0]}"


def test_tvla_service_works_regardless_of_starting_group(mock_repo):
    # Use 4 traces total, 1 sample each
    params = RangeParameters(Range(0, 4), Range(0, 1))

    # Case 1: Trace 0,2 = [10, 12], Trace 1,3 = [20, 22]
    traces_a = np.array([[10.0], [20.0], [12.0], [22.0]])
    mock_repo.get_batch.side_effect = [MagicMock(traces=traces_a)]
    service_a = TvlaService(mock_repo, params)
    t_val_a = service_a.run()

    # Case 2: Swapped (Trace 0,2 = [20, 22], Trace 1,3 = [10, 12])
    traces_b = np.array([[20.0], [10.0], [22.0], [12.0]])
    mock_repo.get_batch.side_effect = [MagicMock(traces=traces_b)]
    service_b = TvlaService(mock_repo, params)
    t_val_b = service_b.run()

    # Assertions
    # equal_nan=True is needed because if math fails, both should fail the same way
    # but with N=2, these should now be real numbers.
    assert np.isclose(np.abs(t_val_a), np.abs(t_val_b))
    assert np.isclose(t_val_a, -t_val_b)
