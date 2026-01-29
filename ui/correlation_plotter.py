import matplotlib.pyplot as plt
import numpy as np


class CorrelationPlotter:
    def __init__(
        self,
        results: list[tuple[int, np.ndarray]],
        correct_keys: list[tuple[int, int]] = None,
    ):
        """
        :param results: list of (byte_index, correlation_matrix)
        :param correct_keys: list of (byte_index, key_value)
        """
        self.results = results
        # Convert correct_keys list to a dictionary for O(1) lookup: {byte_index: key_value}
        self.correct_keys_dict = dict(correct_keys) if correct_keys else {}

    def plot_cpa_results(self, byte_index: int):
        # Locate the matrix for the requested byte
        corr_matrix = next(
            (res for idx, res in self.results if idx == byte_index), None
        )

        if corr_matrix is None:
            print(f"Error: No results found for Byte {byte_index}")
            return

        plt.figure(figsize=(12, 6))
        samples = corr_matrix.shape[1]
        x_axis = np.arange(samples)

        # 1. Calculate Envelopes (Global max/min at every sample point)
        highest_envelope = np.max(corr_matrix, axis=0)
        lowest_envelope = np.min(corr_matrix, axis=0)

        # 2. Determine which specific key to plot as the "Result" line
        correct_k = self.correct_keys_dict.get(byte_index)

        if correct_k is not None:
            # Plot the known correct key
            target_key = correct_k
            label_prefix = f"Correct Key ({target_key:02X})"
        else:
            # Fallback: Find the key with the highest absolute correlation peak
            target_key = np.argmax(np.max(np.abs(corr_matrix), axis=1))
            label_prefix = f"Best Candidate ({target_key:02X})"

        # --- Plotting ---
        # Line 1: Highest correlation envelope
        plt.plot(
            x_axis,
            highest_envelope,
            color="red",
            label="Max Envelope",
            linewidth=1,
            alpha=0.7,
        )

        # Line 2: Lowest correlation envelope
        plt.plot(
            x_axis,
            lowest_envelope,
            color="black",
            label="Min Envelope",
            linewidth=1,
            alpha=0.7,
        )

        # Line 3: The specific key trace (Correct or Best Candidate)
        plt.plot(
            x_axis,
            corr_matrix[target_key],
            color="blue",
            linestyle="--",
            label=label_prefix,
            linewidth=1.5,
        )

        plt.title(f"CPA Correlation Analysis: Byte {byte_index:02d}")
        plt.xlabel("Sample Point")
        plt.ylabel("Correlation Coefficient")
        plt.legend(loc="upper right", frameon=True)
        plt.grid(True, alpha=0.2)
        plt.axhline(0, color="black", lw=1, alpha=0.3)
        plt.tight_layout()
        plt.show()

    # def plot_cpa_results(self, byte_index: int):
    #     # Find the specific result for this byte index
    #     corr_matrix = next((res for idx, res in self.results if idx == byte_index), None)
    #
    #     if corr_matrix is None:
    #         print(f"Error: No results found for Byte {byte_index}")
    #         return
    #
    #     plt.figure(figsize=(12, 6))
    #     samples = corr_matrix.shape[1]
    #     x_axis = np.arange(samples)
    #
    #     # --- Calculate Envelopes ---
    #     # Instead of picking one key, we pick the highest/lowest value at EVERY sample point
    #     # Resulting shape: (samples,)
    #     highest_line = np.max(corr_matrix, axis=0)
    #     lowest_line = np.min(corr_matrix, axis=0)
    #
    #     # --- Plotting ---
    #     # 1. Highest Coefficient Line (The upper boundary of all 256 traces)
    #     plt.plot(x_axis, highest_line, color="red", label="Max Correlation Envelope", linewidth=1)
    #
    #     # 2. Lowest Correlation Line (The lower boundary/most negative)
    #     plt.plot(x_axis, lowest_line, color="black", label="Min Correlation Envelope", linewidth=1)
    #
    #     # 3. Correct Key Line (The specific trace for the known key)
    #     correct_k = self.correct_keys_dict.get(byte_index)
    #     if correct_k is not None:
    #         correct_trace = corr_matrix[correct_k]
    #         plt.plot(x_axis, correct_trace, color="blue", linestyle="--",
    #                  label=f"Correct Key ({correct_k:02X})", linewidth=1.2, alpha=0.8)
    #
    #     plt.title(f"CPA Correlation Envelopes - Byte {byte_index:02d}")
    #     plt.xlabel("Sample Point")
    #     plt.ylabel("Correlation Coefficient")
    #     plt.legend(loc="upper right")
    #     plt.grid(True, alpha=0.2)
    #     plt.axhline(0, color='black', lw=1, alpha=0.3)
    #     plt.show()

    # def plot_cpa_results(self, byte_index: int):
    #     """
    #     Plots exactly 3 lines: Highest, Lowest, and Correct key.
    #     """
    #     # Find the specific result for this byte index
    #     corr_matrix = next((res for idx, res in self.results if idx == byte_index), None)
    #
    #     if corr_matrix is None:
    #         print(f"Error: No results found for Byte {byte_index}")
    #         return
    #
    #     plt.figure(figsize=(12, 6))
    #     samples = corr_matrix.shape[1]
    #     x_axis = np.arange(samples)
    #
    #     # --- Calculations ---
    #     # Get the max absolute peak reached by each of the 256 keys
    #     max_peaks = np.max(np.abs(corr_matrix), axis=1)
    #
    #     best_k = np.argmax(max_peaks)
    #     worst_k = np.argmin(max_peaks)
    #     correct_k = self.correct_keys_dict.get(byte_index)
    #
    #     # --- Plotting ---
    #
    #     # 1. Plot Lowest Correlation (Black/Dashed)
    #     plt.plot(x_axis, corr_matrix[worst_k], color="black", linestyle=":",
    #              label=f"Lowest Correlation (Key: {worst_k:02X})", alpha=0.6)
    #
    #     # 2. Plot Highest Correlation (Red)
    #     plt.plot(x_axis, corr_matrix[best_k], color="red", linewidth=1.5,
    #              label=f"Highest Correlation (Key: {best_k:02X})")
    #
    #     # 3. Plot Correct Key (Blue) - Only if it's NOT the same as the best_k
    #     if correct_k is not None:
    #         if correct_k != best_k:
    #             plt.plot(x_axis, corr_matrix[correct_k], color="blue", linestyle="--",
    #                      label=f"Correct Key ({correct_k:02X})", linewidth=1.5)
    #             plt.title(f"Byte {byte_index:02d}: Correct Key is NOT Rank 1", color="red")
    #         else:
    #             plt.title(f"Byte {byte_index:02d}: Correct Key is Rank 1", color="green")
    #     else:
    #         plt.title(f"CPA Results for Byte {byte_index:02d} (Target Key Unknown)")
    #
    #     plt.xlabel("Sample Point (Time)")
    #     plt.ylabel("Correlation Coefficient")
    #     plt.legend(loc="upper right")
    #     plt.grid(True, alpha=0.2)
    #     plt.axhline(0, color='black', lw=1, alpha=0.3)  # Reference line at 0
    #     plt.show()

    # def plot_cpa_results(self, byte_index: int):
    #     """
    #     Plots CPA results for a specific byte.
    #     :param byte_index: The index of the byte to visualize (e.g., 0 for Byte 00)
    #     """
    #     # Find the specific result for this byte index
    #     corr_matrix = next((res for idx, res in self.results if idx == byte_index), None)
    #
    #     if corr_matrix is None:
    #         print(f"Error: No results found for Byte {byte_index}")
    #         return
    #
    #     plt.figure(figsize=(12, 6))
    #     samples = corr_matrix.shape[1]
    #     x_axis = np.arange(samples)
    #
    #     # 1. Background: Plot all 256 hypotheses in very light grey
    #     # for i in range(256):
    #     #     plt.plot(x_axis, corr_matrix[i], color="grey", alpha=0.05, linewidth=0.5)
    #
    #     # --- Calculations ---
    #     # Find the max absolute peak for every key
    #     max_peaks = np.max(np.abs(corr_matrix), axis=1)
    #
    #     # Highest Correlation (Best Candidate)
    #     best_k = np.argmax(max_peaks)
    #
    #     # Lowest Correlation (The "quietest" hypothesis)
    #     worst_k = np.argmin(max_peaks)
    #
    #     # Correct Key (from the provided list)
    #     correct_k = self.correct_keys_dict.get(byte_index)
    #
    #     # --- Plotting Specific Lines ---
    #
    #     # Highest (Red)
    #     plt.plot(x_axis, corr_matrix[best_k], color="red", label=f"Highest (Key: {best_k:02X})", linewidth=1.2)
    #
    #     # Lowest (Black/Dashed)
    #     plt.plot(x_axis, corr_matrix[worst_k], color="black", linestyle=":", label=f"Lowest (Key: {worst_k:02X})",
    #              alpha=0.7)
    #
    #     # Correct Key (Blue - only if it exists and isn't already the 'Best')
    #     if correct_k is not None:
    #         if correct_k == best_k:
    #             plt.gca().set_title(f"Byte {byte_index:02d}: Success! Correct Key is Highest.", color="green")
    #         else:
    #             plt.plot(x_axis, corr_matrix[correct_k], color="blue", linestyle="--",
    #                      label=f"Correct Key ({correct_k:02X})", linewidth=1.2)
    #             plt.gca().set_title(f"Byte {byte_index:02d}: Key Ranking In Progress", color="orange")
    #     else:
    #         plt.title(f"CPA Results for Byte {byte_index:02d}")
    #
    #     plt.xlabel("Sample Point")
    #     plt.ylabel("Correlation")
    #     plt.legend(loc="upper right", frameon=True)
    #     plt.grid(True, alpha=0.2)
    #     plt.tight_layout()
    #     plt.show()


# import numpy as np


# from matplotlib import pyplot as plt
#
#
# class CorrelationPlotter:
#
#     def __init__(self, result: np.ndarray):
#         self._result = result
#
#     def plot_cpa_results(self, correct_key=None):
#         """
#         Plots CPA correlation results.
#         :param _result: Numpy array of shape (256, samples)
#         :param correct_key: Optional integer of the known correct key to highlight
#         """
#         plt.figure(figsize=(12, 6))
#
#         samples = self._result.shape[1]
#         x_axis = np.arange(samples)
#
#         # Plot all 256 hypotheses in light grey
#         for i in range(256):
#             plt.plot(x_axis, self._result[i], color="grey", alpha=0.1, linewidth=0.5)
#
#         # Identify the best candidate (highest absolute peak)
#         best_candidate = np.argmax(np.max(np.abs(self._result), axis=1))
#
#         # Highlight the best candidate
#         plt.plot(
#             x_axis,
#             self._result[best_candidate],
#             color="red",
#             linewidth=1,
#             label=f"Best Candidate (Key: {best_candidate:#04x})",
#         )
#
#         # If we know the correct key and it's different from the best candidate, highlight it
#         if correct_key is not None and correct_key != best_candidate:
#             plt.plot(
#                 x_axis,
#                 self._result[correct_key],
#                 color="blue",
#                 linewidth=1,
#                 linestyle="--",
#                 label=f"Correct Key ({correct_key:#04x})",
#             )
#
#         plt.title("CPA Correlation Results (256 Key Hypotheses)")
#         plt.xlabel("Sample Point")
#         plt.ylabel("Correlation")
#         plt.legend(loc="upper right")
#         plt.grid(True, alpha=0.2)
#         plt.show()
#
#     def plot_cpa_results2(self, correct_key=None):
#         plt.figure(figsize=(12, 6))
#         samples = self._result.shape[1]
#         x_axis = np.arange(samples)
#
#         # 1. Background: Plot all 256 hypotheses in very light grey
#         for i in range(256):
#             plt.plot(x_axis, self._result[i], color="grey", alpha=0.05, linewidth=0.5)
#
#         # 2. Logic for Specific Traces
#         # Identify index of trace containing the absolute highest and lowest peaks
#         idx_highest = np.argmax(np.max(self._result, axis=1))
#         idx_lowest = np.argmin(np.min(self._result, axis=1))
#
#         # 3. Plotting the 3 requested lines
#
#         # Trace with the Highest Correlation (often the "Best Candidate")
#         plt.plot(x_axis, self._result[idx_highest], color="red",
#                  linewidth=1.5, label=f"Highest Corr (Key: {idx_highest:#04x})")
#
#         # Trace with the Lowest Correlation
#         plt.plot(x_axis, self._result[idx_lowest], color="green",
#                  linewidth=1.5, label=f"Lowest Corr (Key: {idx_lowest:#04x})")
#
#         # Correct Key (if provided)
#         if correct_key is not None:
#             # Use a dashed line to make it visible even if it overlaps with others
#             plt.plot(x_axis, self._result[correct_key], color="blue",
#                      linewidth=2, linestyle="--", label=f"Correct Key ({correct_key:#04x})")
#
#         plt.title("CPA Correlation: Highest vs Lowest vs Correct")
#         plt.xlabel("Sample Point")
#         plt.ylabel("Correlation")
#         plt.legend(loc="upper right", frameon=True, shadow=True)
#         plt.grid(True, alpha=0.3)
#         plt.tight_layout()
#         plt.show()
