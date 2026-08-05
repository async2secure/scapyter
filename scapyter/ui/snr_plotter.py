import matplotlib.pyplot as plt
import numpy as np


class SnrPlotter:
    def __init__(self, data: dict[int, np.ndarray]):
        self._data = data

    def plot_byte(self, byte_location: int):
        snr = self._data[byte_location]

        peak = np.argmax(snr)

        plt.figure(figsize=(12, 4))

        plt.plot(
            snr,
            color="red",
        )

        plt.scatter(
            peak,
            snr[peak],
            color="blue",
        )

        plt.title(
            f"Byte {byte_location} | " f"Peak sample={peak}, " f"SNR={snr[peak]:.3f}"
        )

        plt.xlabel("Samples")
        plt.ylabel("SNR")
        plt.grid(alpha=0.3)

        plt.show()
