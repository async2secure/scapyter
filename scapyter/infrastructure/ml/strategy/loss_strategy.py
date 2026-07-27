from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Protocol

import torch
from torch import nn


class LossStrategy(Protocol):
    """
    Encapsulates everything required by a training objective.

    Responsibilities
    ----------------
    - Create the loss function.
    - Encode a raw class into the representation expected by the loss.
    - Convert encoded targets back into class indices for metrics.
    """

    # @abstractmethod
    # def create_criterion(self) -> nn.Module:
    #     pass

    @abstractmethod
    def loss(
        self,
        logits: torch.Tensor,
        targets: torch.Tensor,
    ) -> torch.Tensor:
        raise NotImplementedError

    @abstractmethod
    def predictions(
        self,
        logits: torch.Tensor,
    ) -> torch.Tensor:
        raise NotImplementedError

    @abstractmethod
    def encode_target(
        self,
        label: int,
        num_classes: int,
    ) -> torch.Tensor:
        raise NotImplementedError

    @abstractmethod
    def decode_targets(
        self,
        targets: torch.Tensor,
    ) -> torch.Tensor:
        """
        Convert targets into flat class indices.

        Used for computing accuracy.
        """
        raise NotImplementedError
