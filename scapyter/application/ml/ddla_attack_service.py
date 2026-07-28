# application/ddla_attack.py
from tqdm import tqdm

from scapyter.application.ml.preprocessing.preprocessing_project_file_reader import (
    ProcessedProjectFileReader,
)
from scapyter.domain.leakage.leakage import LeakageModel
from scapyter.domain.ml.distinguishers import NonProfiledDistinguisher
from scapyter.domain.ml.value_objects import AttackMetric, AttackResult
from scapyter.domain.value_object import DataSource, Range
from scapyter.infrastructure.ml.stream_dataset import StreamDataset


class DDLAAttackService:

    def __init__(
        self,
        distinguisher: NonProfiledDistinguisher,
        leakage_model: LeakageModel,
        data_source: DataSource,
        project_file_reader: ProcessedProjectFileReader,
    ):
        self.distinguisher = distinguisher
        self._leakage_model = leakage_model
        self._project_file_reader = project_file_reader
        self._data_source = data_source

    def run(
        self,
        byte_location: int,
        trace_range: Range,
        sample_range: Range,
        split_percentage: float = 0.7,
    ) -> AttackResult:
        results = []

        all_indices = list(range(trace_range.start, trace_range.end))

        split = int(len(all_indices) * split_percentage)

        train_indices = all_indices[:split]
        validation_indices = all_indices[split:]

        self._project_file_reader.fit(
            trace_range=Range(
                train_indices[0],
                train_indices[-1] + 1,
            ),
            sample_range=sample_range,
        )

        train_dataset = StreamDataset(
            repo=self._project_file_reader,
            indices=train_indices,
            trace_range=trace_range,
            sample_range=sample_range,
        )

        validation_dataset = StreamDataset(
            repo=self._project_file_reader,
            indices=validation_indices,
            trace_range=trace_range,
            sample_range=sample_range,
        )

        for key_guess in tqdm(
            range(256),
            desc=f"DDLA byte {byte_location}",
        ):
            training = self.distinguisher.evaluate(
                train_dataset,
                validation_dataset=validation_dataset,
                leakage_model=self._leakage_model,
                key_guess=key_guess,
                byte_location=byte_location,
                data_source=self._data_source,
            )

            results.append(
                AttackMetric(
                    key_guess=key_guess,
                    training=training,
                )
            )

        # 3. rank keys (application logic)
        results.sort(key=lambda x: x.accuracy, reverse=True)

        return AttackResult(results)
