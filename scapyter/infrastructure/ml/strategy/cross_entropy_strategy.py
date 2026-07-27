from __future__ import annotations

import torch
from torch import nn

from infrastructure.ml.strategy.loss_strategy import LossStrategy


class CrossEntropyStrategy(LossStrategy):

    def __init__(self):
        self._criterion = nn.CrossEntropyLoss()

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
        return torch.tensor(
            label,
            dtype=torch.long,
        ).squeeze()

    def decode_targets(
        self,
        targets: torch.Tensor,
    ) -> torch.Tensor:
        return targets
