# infrastructure/ml/pytorch_ddla.py

from __future__ import annotations

from typing import Callable

import torch
from torch import nn
from torch.utils.data import DataLoader

from .hypothetical_label_dataset import HypotheticalLabelDataset
from .strategy.loss_strategy import LossStrategy
from .stream_dataset import StreamDataset
from .trainer import PyTorchTrainer
from ...domain.leakage.leakage import LeakageModel
from ...domain.ml.distinguishers import NonProfiledDistinguisher
from ...domain.ml.value_objects import TrainingResult
from ...domain.value_object import DataSource


class PyTorchDdlaAdapter(NonProfiledDistinguisher):
    """
    PyTorch implementation of the NonProfiledDistinguisher.

    Responsibilities
    ----------------
    - Convert domain objects into PyTorch tensors.
    - Create the neural network.
    - Build the DataLoader.
    - Delegate training to PyTorchTrainer.
    """

    def __init__(
        self,
        trainer: PyTorchTrainer,
        model_factory: Callable[[], torch.nn.Module],
        loss_strategy: LossStrategy,
        optimizer_factory: Callable[[nn.Module], torch.optim.Optimizer],
        *,
        batch_size: int = 256,
        device: str | None = None,
    ) -> None:
        self._trainer = trainer
        self._model_factory = model_factory
        self._loss_strategy = loss_strategy
        self._optimizer_factory = optimizer_factory
        self._batch_size = batch_size

        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"

        self._device = torch.device(device)

    def evaluate(
        self,
        train_dataset: StreamDataset,
        validation_dataset: StreamDataset | None,
        leakage_model: LeakageModel,
        byte_location: int,
        key_guess: int,
        data_source: DataSource,
    ) -> TrainingResult:

        num_classes: int = self._model_factory().num_classes()

        train_label_dataset = HypotheticalLabelDataset(
            base_dataset=train_dataset,
            leakage_model=leakage_model,
            key_guess=key_guess,
            byte_location=byte_location,
            data_source=data_source,
            num_classes=num_classes,
            loss_strategy=self._loss_strategy,
        )

        train_loader = DataLoader(
            train_label_dataset,
            batch_size=self._batch_size,
            shuffle=True,
        )

        validation_loader = None

        if validation_dataset is not None:
            validation_label_dataset = HypotheticalLabelDataset(
                base_dataset=validation_dataset,
                leakage_model=leakage_model,
                key_guess=key_guess,
                byte_location=byte_location,
                data_source=data_source,
                loss_strategy=self._loss_strategy,
                num_classes=num_classes,
            )

            validation_loader = DataLoader(
                validation_label_dataset,
                batch_size=self._batch_size,
                shuffle=False,
            )

        model = self._model_factory()

        optimizer = self._optimizer_factory(model)
        return self._trainer.fit(
            model=model,
            train_loader=train_loader,
            validation_loader=validation_loader,
            device=self._device,
            optimizer=optimizer,
            loss_strategy=self._loss_strategy,
        )
