import numpy as np


def inverse_shift_rows(state: np.ndarray) -> np.ndarray:
    """
    Apply AES inverse ShiftRows.

    Input shape:
        (..., 16)

    AES state:

        0   1   2   3
        4   5   6   7
        8   9  10  11
        12 13  14  15
    """
    state = np.asarray(state)

    if state.shape[-1] != 16:
        raise ValueError(f"Expected last dimension to be 16, got {state.shape}")

    return state[
        ...,
        [
            0,
            13,
            10,
            7,
            4,
            1,
            14,
            11,
            8,
            5,
            2,
            15,
            12,
            9,
            6,
            3,
        ],
    ]


def shift_rows(state: np.ndarray) -> np.ndarray:
    """
    Apply AES ShiftRows.

    Input shape:
        (..., 16)

    AES state:

        0   1   2   3
        4   5   6   7
        8   9  10  11
        12 13  14  15
    """
    state = np.asarray(state)

    if state.shape[-1] != 16:
        raise ValueError(f"Expected last dimension to be 16, got {state.shape}")

    return state[
        ...,
        [
            0,
            5,
            10,
            15,
            4,
            9,
            14,
            3,
            8,
            13,
            2,
            7,
            12,
            1,
            6,
            11,
        ],
    ]


def _gmul(a: int, b: int) -> int:
    r = 0
    while b:
        if b & 1:
            r ^= a
        a = ((a << 1) ^ 0x11B) & 0xFF if a & 0x80 else a << 1
        b >>= 1
    return r


FMUL = {
    c: np.array([_gmul(x, c) for x in range(256)], dtype=np.uint8) for c in (1, 2, 3)
}
BMUL = {
    c: np.array([_gmul(x, c) for x in range(256)], dtype=np.uint8)
    for c in (9, 11, 13, 14)
}
IMC = [
    [14, 11, 13, 9],
    [9, 14, 11, 13],
    [13, 9, 14, 11],
    [11, 13, 9, 14],
]  # InvMixColumns

MC = [
    [2, 3, 1, 1],
    [1, 2, 3, 1],
    [1, 1, 2, 3],
    [3, 1, 1, 2],
]  # MixColumns matrix


def _mix_columns_generic(state: np.ndarray, matrix, tables) -> np.ndarray:
    """
    Apply a 4x4 GF(2^8) matrix to every column of an AES state.

    state:  (..., 16) uint8, index = 4*col + row
    matrix: 4x4 list of coefficients (MC or IMC)
    tables: dict coefficient -> 256-entry lookup table (FMUL or BMUL)
    """
    state = np.asarray(state, dtype=np.uint8)
    if state.shape[-1] != 16:
        raise ValueError(f"Expected last dimension to be 16, got {state.shape}")

    out = np.empty_like(state)
    for c in range(4):
        for i in range(4):
            acc = np.zeros(state.shape[:-1], dtype=np.uint8)
            for j in range(4):
                acc ^= tables[matrix[i][j]][state[..., 4 * c + j]]
            out[..., 4 * c + i] = acc
    return out


def mix_columns(state: np.ndarray) -> np.ndarray:
    """AES MixColumns on (..., 16) states."""
    return _mix_columns_generic(state, MC, FMUL)


def inverse_mix_columns(state: np.ndarray) -> np.ndarray:
    """AES InvMixColumns on (..., 16) states."""
    return _mix_columns_generic(state, IMC, BMUL)
