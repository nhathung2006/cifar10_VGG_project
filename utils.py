import csv
import json
import random
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import torch
from torch import nn

from config import (
    ACCURACY_PLOT_PATH,
    CONFUSION_MATRIX_PATH,
    DATA_DIR,
    HISTORY_CSV_PATH,
    LOSS_PLOT_PATH,
    OUTPUT_DIR,
)


def ensure_directories() -> None:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def set_seed(seed: int) -> None:
    """
    Cố định seed để kết quả giữa các lần chạy ổn định hơn.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    if torch.backends.cudnn.is_available():
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def get_device() -> torch.device:
    """
    Ưu tiên CUDA, sau đó Apple MPS, cuối cùng là CPU.
    """
    if torch.cuda.is_available():
        return torch.device("cuda")

    if (
        hasattr(torch.backends, "mps")
        and torch.backends.mps.is_available()
    ):
        return torch.device("mps")

    return torch.device("cpu")


def save_checkpoint(
    path: Path,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    val_loss: float,
    val_accuracy: float,
) -> None:
    """
    Lưu model tốt nhất cùng optimizer và thông tin epoch.
    """
    path.parent.mkdir(parents=True, exist_ok=True)

    checkpoint = {
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "val_loss": val_loss,
        "val_accuracy": val_accuracy,
    }
    torch.save(checkpoint, path)


def load_checkpoint(
    path: Path,
    model: nn.Module,
    device: torch.device,
    optimizer: torch.optim.Optimizer | None = None,
) -> dict[str, Any]:
    """
    Nạp checkpoint an toàn do chính project này tạo ra.
    """
    if not path.exists():
        raise FileNotFoundError(
            f"Không tìm thấy checkpoint: {path}\n"
            "Hãy chạy python train.py trước."
        )

    try:
        checkpoint = torch.load(
            path,
            map_location=device,
            weights_only=True,
        )
    except TypeError:
        # Tương thích với một số bản PyTorch cũ.
        checkpoint = torch.load(
            path,
            map_location=device,
        )

    model.load_state_dict(checkpoint["model_state_dict"])

    if optimizer is not None and "optimizer_state_dict" in checkpoint:
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

    return checkpoint


def save_history_csv(history: list[dict[str, float]]) -> None:
    if not history:
        return

    HISTORY_CSV_PATH.parent.mkdir(parents=True, exist_ok=True)

    with HISTORY_CSV_PATH.open(
        "w",
        newline="",
        encoding="utf-8",
    ) as file:
        writer = csv.DictWriter(
            file,
            fieldnames=list(history[0].keys()),
        )
        writer.writeheader()
        writer.writerows(history)


def plot_history(history: list[dict[str, float]]) -> None:
    """
    Lưu hai biểu đồ riêng:
    - loss
    - accuracy
    """
    if not history:
        return

    epochs = [int(item["epoch"]) for item in history]
    train_losses = [item["train_loss"] for item in history]
    val_losses = [item["val_loss"] for item in history]
    train_accuracies = [item["train_accuracy"] for item in history]
    val_accuracies = [item["val_accuracy"] for item in history]

    plt.figure(figsize=(9, 6))
    plt.plot(epochs, train_losses, label="Train loss")
    plt.plot(epochs, val_losses, label="Validation loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("VGG16 CIFAR-10 - Loss")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(LOSS_PLOT_PATH, dpi=160)
    plt.close()

    plt.figure(figsize=(9, 6))
    plt.plot(epochs, train_accuracies, label="Train accuracy")
    plt.plot(epochs, val_accuracies, label="Validation accuracy")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy (%)")
    plt.title("VGG16 CIFAR-10 - Accuracy")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(ACCURACY_PLOT_PATH, dpi=160)
    plt.close()


def save_confusion_matrix(
    confusion_matrix: torch.Tensor,
    class_names: tuple[str, ...],
) -> None:
    matrix = confusion_matrix.numpy()

    plt.figure(figsize=(10, 8))
    plt.imshow(matrix)
    plt.title("Confusion matrix - CIFAR-10")
    plt.xlabel("Predicted label")
    plt.ylabel("True label")
    plt.xticks(
        range(len(class_names)),
        class_names,
        rotation=45,
        ha="right",
    )
    plt.yticks(
        range(len(class_names)),
        class_names,
    )

    threshold = matrix.max() / 2.0 if matrix.size else 0
    for row in range(matrix.shape[0]):
        for column in range(matrix.shape[1]):
            value = int(matrix[row, column])
            plt.text(
                column,
                row,
                str(value),
                ha="center",
                va="center",
                color="white" if value > threshold else "black",
                fontsize=8,
            )

    plt.colorbar()
    plt.tight_layout()
    plt.savefig(CONFUSION_MATRIX_PATH, dpi=170)
    plt.close()


def save_json(path: Path, data: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)

    with path.open("w", encoding="utf-8") as file:
        json.dump(
            data,
            file,
            ensure_ascii=False,
            indent=2,
        )
