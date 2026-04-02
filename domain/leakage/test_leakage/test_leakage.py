import numpy as np

from domain.leakage.leakage import SboxOutputLeakageModel


def test_sbox_leakage_calculation():
    """
    Test a known AES S-box mapping:
    Input: 0x00, Key Guess: 0x12
    Step 1: 0x00 ^ 0x12 = 0x12
    Step 2: Sbox(0x12) = 0xC9 (decimal 201)
    Step 3: HW(0xC9) -> HW(11001001 binary) = 4
    """
    # Setup: 1 trace, 1 byte (value 0x00)
    plaintexts = np.array([[0x00]], dtype=np.uint8)
    model = SboxOutputLeakageModel()

    # Calculate for byte_location 0, key guess 0x12
    result = model.calculate(byte_location=0, key_guess=0x12, known_data=plaintexts)

    assert result[0] == 4
    assert isinstance(result, np.ndarray)


def test_vectorization_multiple_traces():
    """
    Test that the model processes multiple traces simultaneously.
    """
    # Two traces, targeting byte index 1
    # Trace 0: byte 1 is 0xAB
    # Trace 1: byte 1 is 0xFF
    plaintexts = np.array([[0x00, 0xAB], [0x00, 0xFF]], dtype=np.uint8)

    model = SboxOutputLeakageModel()
    guess = 0x00  # XORing with 0 keeps values same

    # Sbox(0xAB) = 0x62 -> HW(01100010) = 3
    # Sbox(0xFF) = 0x16 -> HW(00010110) = 3

    results = model.calculate(byte_location=1, key_guess=guess, known_data=plaintexts)

    np.testing.assert_array_equal(results, [3, 3])


def test_wrong_guess_produces_different_leakage():
    """
    Ensure different key guesses produce different power models.
    """
    plaintexts = np.array([[0x42]], dtype=np.uint8)
    model = SboxOutputLeakageModel()

    leakage_guess_1 = model.calculate(byte_location= 0, key_guess=0x01,known_data=plaintexts)
    leakage_guess_2 = model.calculate(byte_location=0, key_guess=0x02, known_data=plaintexts)

    assert leakage_guess_1 != leakage_guess_2
