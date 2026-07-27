import torch
from torch import nn
import torch.nn.functional as F

from infrastructure.ml.strategy.loss_strategy import LossStrategy


class MSEStrategy(LossStrategy):

    def __init__(self):
        self._criterion = nn.MSELoss()

    def loss(
        self,
        logits: torch.Tensor,
        targets: torch.Tensor,
    ) -> torch.Tensor:
        return self._criterion(logits, targets)

    def encode_target(
        self,
        label: int,
        num_classes: int,
    ) -> torch.Tensor:

        label = torch.tensor(
            int(label),
            dtype=torch.long,
        )

        return F.one_hot(
            label,
            num_classes=num_classes,
        ).float()

    def predictions(
        self,
        logits: torch.Tensor,
    ) -> torch.Tensor:

        return torch.argmax(
            logits,
            dim=1,
        )

    def decode_targets(
        self,
        targets: torch.Tensor,
    ) -> torch.Tensor:

        return torch.argmax(
            targets,
            dim=1,
        )
