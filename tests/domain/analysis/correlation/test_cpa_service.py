from unittest.mock import MagicMock

import numpy as np
from scipy.stats import pearsonr

from scapyter.domain.analysis.correlation.cpa import CpaCorrelation
from scapyter.domain.analysis.correlation.service import CorrelationService
from scapyter.domain.analysis.correlation.value_objects.correlation_functions import (
    CorrelationFunction,
)
from scapyter.domain.leakage.leakage import SboxOutputLeakageModel
from scapyter.domain.repository.project_file_reader import ProjectFileReader
from scapyter.domain.value_object import (
    DataSource,
    KeyByteGuesses,
    Range,
    RangeParameters,
)


def test_correlation_service_matches_scipy_pearson():
    traces = np.array(
        [
            [1.0, 2.0, 3.0],
            [2.0, 4.0, 2.0],
            [3.0, 1.0, 5.0],
            [4.0, 8.0, 4.0],
            [5.0, 3.0, 6.0],
            [6.0, 7.0, 7.0],
        ]
    )

    known_data = np.array(
        [
            [10],
            [20],
            [30],
            [40],
            [50],
            [60],
        ],
        dtype=np.uint8,
    )

    batch = MagicMock()
    batch.traces = traces
    batch.metadata = {
        DataSource.PLAINTEXT.value: known_data,
    }

    mock_reader = MagicMock(spec=ProjectFileReader)
    mock_reader.get_batch.return_value = batch

    key_guess = 0x00

    correlation_function = CorrelationFunction(
        correlation=CpaCorrelation(),
        byte_location=0,
        leakage_model=SboxOutputLeakageModel(),
        key_byte_guesses=KeyByteGuesses([key_guess]),
    )

    params = RangeParameters(
        trace_range=Range(0, 6),
        sample_range=Range(0, 3),
    )

    service = CorrelationService(
        range_parameters=params,
        project_file_reader=mock_reader,
        data_source=DataSource.PLAINTEXT,
        correlation_functions=[correlation_function],
    )

    results = service.run(batch_size=6)

    actual = results[0].corr_matrix

    leakage = correlation_function.leakage_model.calculate(
        byte_location=correlation_function.byte_location,
        known_data=known_data,
        key_guess=key_guess,
    )

    expected = np.array(
        [
            pearsonr(traces[:, sample], leakage).statistic
            for sample in range(traces.shape[1])
        ]
    )

    np.testing.assert_allclose(
        actual[0],
        expected,
        rtol=1e-10,
        atol=1e-12,
    )
