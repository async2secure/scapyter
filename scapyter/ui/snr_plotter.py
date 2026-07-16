import matplotlib.pyplot as plt
import numpy as np


class SnrPlotter:
    def __init__(self, data: np.ndarray):
        self._data = data

    def plot(self):
        plt.figure(figsize=(10, 4))
        plt.plot(self._data, linewidth=1, color="red")

        plt.xlabel("Samples")
        plt.ylabel("SNR")

        plt.tight_layout()
        plt.show()
