from __future__ import annotations

import torch
from torch import nn
import torch.nn.functional as F

from infrastructure.ml.strategy.loss_strategy import LossStrategy


class MSEStrategy(LossStrategy):

    def __init__(self):
        self._criterion = nn.MSELoss()

    def predictions(
        self,
        logits: torch.Tensor,
    ) -> torch.Tensor:
        return torch.argmax(
            logits,
            dim=1,
        )

    def loss(self, logits, targets):
        return self._criterion(logits, targets)

    def encode_target(
        self,
        label: int,
        num_classes: int,
    ) -> torch.Tensor:

        return F.one_hot(
            torch.tensor(label),
            num_classes=num_classes,
        ).float()

    def decode_targets(
        self,
        targets: torch.Tensor,
    ) -> torch.Tensor:

        return targets.argmax(dim=1)
