import torch
from torch import nn
from torch.utils.data import DataLoader

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
        lr: float = 1e-3,
        weight_decay: float = 0.0,
    ):
        self.epochs = epochs
        self.lr = lr
        self.weight_decay = weight_decay

    def fit(
        self,
        model: nn.Module,
        train_loader: DataLoader,
        device: torch.device,
        validation_loader: DataLoader | None = None,
    ) -> TrainingResult:

        model = model.to(device)

        criterion = nn.CrossEntropyLoss()

        optimizer = torch.optim.Adam(
            model.parameters(),
            lr=self.lr,
            weight_decay=self.weight_decay,
        )

        history: list[EpochMetrics] = []

        for epoch in range(self.epochs):

            train_loss, train_accuracy = self._train_epoch(
                model=model,
                loader=train_loader,
                criterion=criterion,
                optimizer=optimizer,
                device=device,
            )

            validation_loss = None
            validation_accuracy = None

            if validation_loader is not None:
                validation_loss, validation_accuracy = self._validate(
                    model=model,
                    loader=validation_loader,
                    criterion=criterion,
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
        criterion: nn.Module,
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

            loss = criterion(logits, y)

            loss.backward()

            optimizer.step()

            total_loss += loss.item()

            predictions = logits.argmax(dim=1)

            correct += (predictions == y).sum().item()
            total += y.size(0)

        average_loss = total_loss / len(loader)
        accuracy = correct / total

        return average_loss, accuracy

    @torch.no_grad()
    def _validate(
        self,
        model: nn.Module,
        loader: DataLoader,
        criterion: nn.Module,
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

            loss = criterion(logits, y)

            total_loss += loss.item()

            predictions = logits.argmax(dim=1)

            correct += (predictions == y).sum().item()
            total += y.size(0)

        average_loss = total_loss / len(loader)
        accuracy = correct / total

        return average_loss, accuracy
