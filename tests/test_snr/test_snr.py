import numpy as np

from scapyter.domain.snr.snr import ProgressiveSnr


# -----------------------------
# Reference fully vectorized SNR
# -----------------------------
def reference_snr(traces: np.ndarray, labels: np.ndarray) -> np.ndarray:
    keys, inverse = np.unique(labels, return_inverse=True)
    n_groups = len(keys)
    N, D = traces.shape

    counts = np.bincount(inverse)
    sums = np.zeros((n_groups, D))
    np.add.at(sums, inverse, traces)
    means = sums / counts[:, None]

    # Signal variance
    signal_variance = np.var(means, axis=0)

    # Noise variance
    centered = traces - means[inverse]
    sq = centered**2
    sq_sums = np.zeros((n_groups, D))
    np.add.at(sq_sums, inverse, sq)
    variances = sq_sums / counts[:, None]

    largest = np.argmax(counts)
    noise_variance = variances[largest]

    return np.nan_to_num(signal_variance / noise_variance)


# -----------------------------
# Smoke test
# -----------------------------
def test_smoke():
    N, D = 100, 50
    traces = np.random.randn(N, D)
    labels = np.random.randint(0, 4, size=N)

    psnr = ProgressiveSnr()
    psnr.update(traces=traces, hex_array=labels)
    out = psnr.finalize()

    assert out.shape == (D,)
    assert np.all(np.isfinite(out))


# -----------------------------
# Correctness vs reference
# -----------------------------
def test_correctness():
    np.random.seed(0)
    N, D = 1000, 200
    traces = np.random.randn(N, D)
    labels = np.random.randint(0, 8, size=N)

    ref = reference_snr(traces, labels)

    psnr = ProgressiveSnr()
    psnr.update(traces=traces, hex_array=labels)
    out = psnr.finalize()

    np.testing.assert_allclose(out, ref, rtol=1e-6, atol=1e-8)


# -----------------------------
# Streaming equivalence
# -----------------------------
def test_streaming_equivalence():
    np.random.seed(0)
    N, D = 5000, 100
    traces = np.random.randn(N, D)
    labels = np.random.randint(0, 16, size=N)

    # reference full batch
    ref = reference_snr(traces, labels)

    # streaming in chunks
    psnr = ProgressiveSnr()
    chunk_size = 500
    for i in range(0, N, chunk_size):
        psnr.update(
            traces=traces[i : i + chunk_size], hex_array=labels[i : i + chunk_size]
        )

    out = psnr.finalize()
    np.testing.assert_allclose(out, ref, rtol=1e-6, atol=1e-8)


# -----------------------------
# Edge case: single group
# -----------------------------
def test_single_group():
    N, D = 100, 20
    traces = np.random.randn(N, D)
    labels = np.zeros(N)  # all same

    psnr = ProgressiveSnr()
    psnr.update(traces=traces, hex_array=labels)
    out = psnr.finalize()

    # signal variance should be ~0
    assert np.allclose(out, 0)


# -----------------------------
# Edge case: very small counts
# -----------------------------
def test_small_counts():
    traces = np.random.randn(3, 10)
    labels = np.array([0, 1, 1])

    psnr = ProgressiveSnr()
    psnr.update(traces=traces, hex_array=labels)
    out = psnr.finalize()
    assert out.shape == (10,)
    assert np.all(np.isfinite(out))
