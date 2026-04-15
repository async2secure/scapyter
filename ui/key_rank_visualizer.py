from domain.value_object import CpaByteResult


import numpy as np
import pandas as pd
from IPython.display import display


class KeyRankVisualizer:
    def __init__(self, full_correlation_results: list[CpaByteResult]):
        """
        :param full_correlation_results:
            List of CpaByteResult objects
        """
        self.results = full_correlation_results

    @staticmethod
    def _get_top_candidates(corr_matrix: np.ndarray, key_candidates, top_n: int = 5):
        """
        Rank hypotheses by strongest correlation peak.
        """
        max_peaks = np.max(np.abs(corr_matrix), axis=1)
        ranked_indices = np.argsort(max_peaks)[::-1]

        return [
            (key_candidates.values[i], float(max_peaks[i]))
            for i in ranked_indices[:top_n]
        ]

    def get_full_key_guess(self) -> bytes:
        """
        Returns best key guess per byte (mapped to real values).
        """
        sorted_results = sorted(self.results, key=lambda r: r.byte_index)

        key = []

        for r in sorted_results:
            max_peaks = np.max(np.abs(r.corr_matrix), axis=1)
            best_idx = int(np.argmax(max_peaks))

            key.append(r.key_candidates.values[best_idx])

        return bytes(key)

    def display_rank_table(self, top_n: int = 5):
        """
        Displays ranking table using actual key values.
        """
        data = {}

        for r in self.results:
            candidates = self._get_top_candidates(
                r.corr_matrix, r.key_candidates, top_n
            )

            data[f"Byte {r.byte_index:02d}"] = [
                f"{val:02X} ({score:.3f})" for val, score in candidates
            ]

        df = pd.DataFrame(data)
        df.index = [f"Rank {i + 1}" for i in range(top_n)]

        display(df)
        return df
