import matplotlib.pyplot as plt
import numpy as np


class SnrPlotter:
    def __init__(self, data: np.ndarray):
        """
        :param snr_array: 1D numpy array of SNR values per sample
        """
        self.key_byte = data[0]
        self.snr = data[1]
        self.N = len(self.snr)

    def plot(self):
        plt.figure(figsize=(10, 4))
        plt.plot(self.snr, linewidth=1, color="red")

        plt.xlabel("Samples")
        plt.ylabel("SNR")

        plt.tight_layout()
        plt.show()
