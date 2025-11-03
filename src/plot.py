import os

import matplotlib.pyplot as plt


def save_performance_plot(history, history_fine_tune, initial_epochs, save_path):
    """Saves the combined training and validation performance plot."""

    # Ensure the directory for the save_path exists
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    acc = history.history["accuracy"] + history_fine_tune.history["accuracy"]
    val_acc = (
        history.history["val_accuracy"] + history_fine_tune.history["val_accuracy"]
    )
    loss = history.history["loss"] + history_fine_tune.history["loss"]
    val_loss = history.history["val_loss"] + history_fine_tune.history["val_loss"]

    plt.figure(figsize=(10, 5))

    plt.subplot(1, 2, 1)
    plt.plot(acc, label="Training Accuracy")
    plt.plot(val_acc, label="Validation Accuracy")
    plt.axvline(
        initial_epochs - 1, linestyle="--", color="k", label="Start Fine-Tuning"
    )
    plt.legend(loc="lower right")
    plt.title("Combined Training and Validation Accuracy")
    plt.xlabel("Epoch")

    plt.subplot(1, 2, 2)
    plt.plot(loss, label="Training Loss")
    plt.plot(val_loss, label="Validation Loss")
    plt.axvline(
        initial_epochs - 1, linestyle="--", color="k", label="Start Fine-Tuning"
    )
    plt.legend(loc="upper right")
    plt.title("Combined Training and Validation Loss")
    plt.xlabel("Epoch")

    plt.tight_layout()
    plt.savefig(save_path)
    print(f"✅ Combined performance plot saved to: {save_path}")
