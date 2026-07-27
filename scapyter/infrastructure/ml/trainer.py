import torch
from torch import nn
from torch.utils.data import DataLoader

from infrastructure.ml.strategy.loss_strategy import LossStrategy
from scapyter.domain.ml.value_objects import TrainingResult, EpochMetrics


class PyTorchTrainer:
    """
    Generic PyTorch training engine.

    Responsibilities
    ----------------
    - Train any nn.Module
    - Evaluate on an optional validation set
    - Compute loss and accuracy
    - Handle device placement
    """

    def __init__(
        self,
        epochs: int = 10,
    ):
        self.epochs = epochs

    def fit(
        self,
        model: nn.Module,
        train_loader: DataLoader,
        device: torch.device,
        optimizer: torch.optim.Optimizer,
        loss_strategy: LossStrategy,
        validation_loader: DataLoader | None = None,
    ) -> TrainingResult:

        model = model.to(device)

        history: list[EpochMetrics] = []

        for epoch in range(self.epochs):

            train_loss, train_accuracy = self._train_epoch(
                model=model,
                loader=train_loader,
                optimizer=optimizer,
                device=device,
                loss_strategy=loss_strategy,
            )

            validation_loss = None
            validation_accuracy = None

            if validation_loader is not None:
                validation_loss, validation_accuracy = self._validate(
                    model=model,
                    loader=validation_loader,
                    loss_strategy=loss_strategy,
                    device=device,
                )

            history.append(
                EpochMetrics(
                    epoch=epoch + 1,
                    train_loss=train_loss,
                    train_accuracy=train_accuracy,
                    validation_loss=validation_loss,
                    validation_accuracy=validation_accuracy,
                )
            )

        return TrainingResult(history=history)

    @staticmethod
    def _train_epoch(
        model: nn.Module,
        loader: DataLoader,
        loss_strategy: LossStrategy,
        optimizer: torch.optim.Optimizer,
        device: torch.device,
    ) -> tuple[float, float]:

        model.train()

        total_loss = 0.0
        correct = 0
        total = 0

        for x, y in loader:

            x = x.to(device)
            y = y.to(device)

            optimizer.zero_grad()

            logits = model(x)

            loss = loss_strategy.loss(
                logits,
                y,
            )

            loss.backward()

            optimizer.step()

            total_loss += loss.item()

            predictions = loss_strategy.predictions(logits)

            targets = loss_strategy.decode_targets(y)

            correct += (predictions == targets).sum().item()

            total += targets.size(0)

        average_loss = total_loss / len(loader)
        accuracy = correct / total

        return average_loss, accuracy

    @torch.no_grad()
    def _validate(
        self,
        model: nn.Module,
        loader: DataLoader,
        loss_strategy: LossStrategy,
        device: torch.device,
    ) -> tuple[float, float]:

        model.eval()

        total_loss = 0.0
        correct = 0
        total = 0

        for x, y in loader:

            x = x.to(device)
            y = y.to(device)

            logits = model(x)

            loss = loss_strategy.loss(
                logits,
                y,
            )

            total_loss += loss.item()

            predictions = loss_strategy.predictions(logits)

            targets = loss_strategy.decode_targets(y)

            correct += (predictions == targets).sum().item()

            total += targets.size(0)

        average_loss = total_loss / len(loader)
        accuracy = correct / total

        return average_loss, accuracy
