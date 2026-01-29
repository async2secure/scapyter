import numpy as np
import pandas as pd
from IPython.display import display


class KeyRankVisualizer:
    def __init__(self, full_correlation_results: list[tuple[int, np.ndarray]]):
        """
        :param full_correlation_results: List of (byte_index, correlation_array)
                                         where correlation_array is shape (256, samples)
        """
        # Store as is; we will iterate through the tuples
        self.results = full_correlation_results

    @staticmethod
    def _get_top_candidates(corr_matrix: np.ndarray, top_n: int = 5):
        # Calculate the max absolute correlation for each of the 256 key hypotheses
        max_peaks = np.max(np.abs(corr_matrix), axis=1)
        ranked_indices = np.argsort(max_peaks)[::-1]

        return [(ranked_indices[r], max_peaks[ranked_indices[r]]) for r in range(top_n)]

    def get_full_key_guess(self) -> bytes:
        """Returns the Rank 1 candidate for all bytes provided, sorted by byte index."""
        # Sort results by the byte index (the first element of the tuple) to ensure correct order
        sorted_results = sorted(self.results, key=lambda x: x[0])

        key = []
        for _, corr_matrix in sorted_results:
            max_peaks = np.max(np.abs(corr_matrix), axis=1)
            key.append(np.argmax(max_peaks))
        return bytes(key)

    def display_rank_table(self, top_n: int = 5):
        """Generates and explicitly displays the table in Jupyter."""
        data = {}

        # Unpack the tuple directly in the loop
        for byte_num, corr_matrix in self.results:
            candidates = self._get_top_candidates(corr_matrix, top_n)
            # Use the actual byte_num from the tuple for the column header
            data[f"Byte {byte_num:02d}"] = [f"{k:02X} ({v:.3f})" for k, v in candidates]

        df = pd.DataFrame(data)
        df.index = [f"Rank {i + 1}" for i in range(top_n)]

        # Apply styling (Green highlight removed as requested)
        styled = df.style.set_caption("CPA Key Hypothesis Ranking").set_table_styles(
            [
                {
                    "selector": "th",
                    "props": [("background-color", "#4CAF50"), ("color", "white")],
                }
            ]
        )

        display(styled)
        return None
