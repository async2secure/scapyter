from pathlib import Path

import numpy as np

from scapyter.domain.tvla.value_objects import TvlaResult


class TvlaResultRepository:
    def save(
        self,
        path: str | Path,
        result: TvlaResult,
    ) -> None:
        path = Path(path)
        path.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        np.savez(
            path,
            t_scores=result.t_scores,
            trace_start=result.trace_start,
            trace_end=result.trace_end,
            sample_start=result.sample_start,
            sample_end=result.sample_end,
        )

    def load(
        self,
        path: str | Path,
    ) -> TvlaResult:
        path = Path(path)

        if not path.exists():
            raise FileNotFoundError(f"TVLA result not found: {path}")

        with np.load(path) as data:
            return TvlaResult(
                t_scores=data["t_scores"].copy(),
                trace_start=int(data["trace_start"]),
                trace_end=int(data["trace_end"]),
                sample_start=int(data["sample_start"]),
                sample_end=int(data["sample_end"]),
            )

    def save_progression(
        self,
        path: str | Path,
        results: list[TvlaResult],
    ) -> None:
        if not results:
            raise ValueError("Cannot save an empty TVLA progression.")

        path = Path(path)
        path.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        np.savez(
            path,
            t_scores=np.stack([result.t_scores for result in results]),
            trace_start=np.array(
                [result.trace_start for result in results],
                dtype=np.int64,
            ),
            trace_end=np.array(
                [result.trace_end for result in results],
                dtype=np.int64,
            ),
            sample_start=np.array(
                [result.sample_start for result in results],
                dtype=np.int64,
            ),
            sample_end=np.array(
                [result.sample_end for result in results],
                dtype=np.int64,
            ),
        )

    def load_progression(
        self,
        path: str | Path,
    ) -> list[TvlaResult]:
        path = Path(path)

        if not path.exists():
            raise FileNotFoundError(f"TVLA progression not found: {path}")

        with np.load(path) as data:
            t_scores = data["t_scores"]
            trace_start = data["trace_start"]
            trace_end = data["trace_end"]
            sample_start = data["sample_start"]
            sample_end = data["sample_end"]

            return [
                TvlaResult(
                    t_scores=t_scores[index].copy(),
                    trace_start=int(trace_start[index]),
                    trace_end=int(trace_end[index]),
                    sample_start=int(sample_start[index]),
                    sample_end=int(sample_end[index]),
                )
                for index in range(len(t_scores))
            ]
