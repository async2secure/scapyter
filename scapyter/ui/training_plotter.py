from __future__ import annotations

import matplotlib.pyplot as plt

from scapyter.domain.ml.value_objects import TrainingResult


class TrainingPlotter:
    """Plots training history."""

    @staticmethod
    def plot(
        result: TrainingResult,
        *,
        figsize: tuple[int, int] = (8, 4),
    ) -> None:

        history = result.history

        epochs = [m.epoch for m in history]

        train_loss = [m.train_loss for m in history]
        train_accuracy = [m.train_accuracy for m in history]

        has_validation = history[0].validation_loss is not None

        fig, axes = plt.subplots(1, 2, figsize=figsize)

        # Loss
        axes[0].plot(
            epochs,
            train_loss,
            label="Train",
            marker="o",
        )

        if has_validation:
            validation_loss = [m.validation_loss for m in history]
            axes[0].plot(
                epochs,
                validation_loss,
                label="Validation",
                marker="o",
            )

        axes[0].set_title("Loss")
        axes[0].set_xlabel("Epoch")
        axes[0].set_ylabel("Cross Entropy")
        axes[0].grid(True)
        axes[0].legend()

        # Accuracy
        axes[1].plot(
            epochs,
            train_accuracy,
            label="Train",
            marker="o",
        )

        if has_validation:
            validation_accuracy = [m.validation_accuracy for m in history]
            axes[1].plot(
                epochs,
                validation_accuracy,
                label="Validation",
                marker="o",
            )

        axes[1].set_title("Accuracy")
        axes[1].set_xlabel("Epoch")
        axes[1].set_ylabel("Accuracy")
        axes[1].set_ylim(0, 1)
        axes[1].grid(True)
        axes[1].legend()

        plt.tight_layout()
        plt.show()
