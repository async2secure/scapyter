import pytest
import numpy as np
from unittest.mock import MagicMock

from scapyter.domain.analysis.tvla import TvlaService
from scapyter.domain.value_object import Range, RangeParameters


@pytest.fixture
def mock_repo():
    return MagicMock()


@pytest.fixture
def range_params():
    # Setup for 4 traces total, each with 3 samples
    return RangeParameters(trace_range=Range(0, 4), trace_sample_range=Range(0, 3))


def test_tvla_service_update_accumulates_correctly(mock_repo, range_params):
    # 1. Arrange
    traces = np.array(
        [
            [1.0, 1.0, 1.0],  # Index 0 (Even)
            [2.0, 2.0, 2.0],  # Index 1 (Odd)
            [3.0, 3.0, 3.0],  # Index 2 (Even)
            [4.0, 4.0, 4.0],  # Index 3 (Odd)
        ]
    )

    mock_batch = MagicMock()
    mock_batch.traces = traces
    mock_repo.get_batch.return_value = mock_batch

    service = TvlaService(mock_repo, range_params)

    # 2. Act: Use the new update() method for the specific range
    service.update(Range(0, 4), batch_size=4)
    service.get_results()  # Triggers calculation if needed

    # 3. Assert
    assert np.all(service._acc_even == np.array([4.0, 4.0, 4.0]))
    assert np.all(service._acc_odd == np.array([6.0, 6.0, 6.0]))
    assert service._count_even == 2
    assert service._count_odd == 2

    mock_repo.get_batch.assert_called_once()
    _, kwargs = mock_repo.get_batch.call_args
    assert kwargs["sample_slice"] == slice(0, 3)


def test_tvla_service_empty_range_handling(mock_repo):
    """Test behavior when trace range is invalid/empty."""
    params = RangeParameters(Range(0, 0), Range(0, 10))
    service = TvlaService(mock_repo, params)

    # Update with empty range
    service.update(Range(0, 0))
    results = service.get_results()

    assert mock_repo.get_batch.called is False
    # Welch's T-test with 0 counts returns NaN
    assert np.all(np.isnan(results))


def test_tvla_service_aggregates_even_odd_correctly(mock_repo):
    traces = np.array([[1.0, 1.0], [2.0, 2.0], [3.0, 3.0], [4.0, 4.0]])

    params = RangeParameters(trace_range=Range(0, 4), trace_sample_range=Range(0, 2))
    mock_batch = MagicMock()
    mock_batch.traces = traces
    mock_repo.get_batch.return_value = mock_batch

    service = TvlaService(mock_repo, params)
    service.update(Range(0, 4), batch_size=4)

    assert np.array_equal(service._acc_even, [4.0, 4.0])
    assert np.array_equal(service._acc_odd, [6.0, 6.0])
    assert np.array_equal(service._acc_even_sq, [10.0, 10.0])
    assert np.array_equal(service._acc_odd_sq, [20.0, 20.0])


def test_tvla_service_incremental_updates(mock_repo):
    """Verifies that multiple calls to update() correctly accumulate state."""
    params = RangeParameters(trace_range=Range(0, 4), trace_sample_range=Range(0, 1))

    # First update: 2 traces
    batch_1 = MagicMock(traces=np.array([[1.0], [2.0]]))
    # Second update: 2 traces
    batch_2 = MagicMock(traces=np.array([[3.0], [4.0]]))

    mock_repo.get_batch.side_effect = [batch_1, batch_2]

    service = TvlaService(mock_repo, params)

    # Step 1: Process traces 0 and 1
    service.update(Range(0, 2))
    assert service._count_even == 1
    assert service._count_odd == 1

    # Step 2: Process traces 2 and 3
    service.update(Range(2, 4))
    assert service._count_even == 2
    assert service._count_odd == 2

    # Total sums: Even (1+3=4), Odd (2+4=6)
    assert service._acc_even[0] == 4.0
    assert service._acc_odd[0] == 6.0


def test_tvla_service_with_odd_batch_size(mock_repo):
    params = RangeParameters(trace_range=Range(0, 6), trace_sample_range=Range(0, 1))

    batch_1_traces = np.array([[0.0], [1.0], [2.0]])
    batch_2_traces = np.array([[3.0], [4.0], [5.0]])

    mock_repo.get_batch.side_effect = [
        MagicMock(traces=batch_1_traces),
        MagicMock(traces=batch_2_traces),
    ]

    service = TvlaService(mock_repo, params)
    service.update(Range(0, 6), batch_size=3)

    # Even Indices: 0, 2, 4 (Sum 6) | Odd Indices: 1, 3, 5 (Sum 9)
    assert service._count_even == 3
    assert service._count_odd == 3
    assert np.isclose(service._acc_even[0], 6.0)
    assert np.isclose(service._acc_odd[0], 9.0)
