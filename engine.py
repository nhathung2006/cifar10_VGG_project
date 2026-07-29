from typing import Optional

import torch
from torch import nn
from torch.utils.data import DataLoader


def train_one_epoch(
    model: nn.Module,
    data_loader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
) -> tuple[float, float]:
    """
    Huấn luyện mô hình trong một epoch.
    Trả về:
    - loss trung bình
    - accuracy (%)
    """
    model.train()

    total_loss = 0.0
    total_correct = 0
    total_samples = 0

    for images, labels in data_loader:
        images = images.to(
            device,
            non_blocking=device.type == "cuda",
        )
        labels = labels.to(
            device,
            non_blocking=device.type == "cuda",
        )

        optimizer.zero_grad(set_to_none=True)

        logits = model(images)
        loss = criterion(logits, labels)

        loss.backward()
        optimizer.step()

        batch_size = labels.size(0)
        total_loss += loss.item() * batch_size
        total_correct += (logits.argmax(dim=1) == labels).sum().item()
        total_samples += batch_size

    average_loss = total_loss / total_samples
    accuracy = 100.0 * total_correct / total_samples

    return average_loss, accuracy


@torch.inference_mode()
def evaluate(
    model: nn.Module,
    data_loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
) -> tuple[float, float]:
    """
    Đánh giá loss và accuracy.
    """
    model.eval()

    total_loss = 0.0
    total_correct = 0
    total_samples = 0

    for images, labels in data_loader:
        images = images.to(
            device,
            non_blocking=device.type == "cuda",
        )
        labels = labels.to(
            device,
            non_blocking=device.type == "cuda",
        )

        logits = model(images)
        loss = criterion(logits, labels)

        batch_size = labels.size(0)
        total_loss += loss.item() * batch_size
        total_correct += (logits.argmax(dim=1) == labels).sum().item()
        total_samples += batch_size

    average_loss = total_loss / total_samples
    accuracy = 100.0 * total_correct / total_samples

    return average_loss, accuracy


@torch.inference_mode()
def evaluate_detailed(
    model: nn.Module,
    data_loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    num_classes: int,
) -> tuple[float, float, torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Đánh giá chi tiết:
    - loss
    - accuracy tổng
    - confusion matrix
    - số dự đoán đúng theo lớp
    - tổng số mẫu theo lớp
    """
    model.eval()

    total_loss = 0.0
    total_correct = 0
    total_samples = 0

    confusion_matrix = torch.zeros(
        (num_classes, num_classes),
        dtype=torch.int64,
    )
    class_correct = torch.zeros(num_classes, dtype=torch.int64)
    class_total = torch.zeros(num_classes, dtype=torch.int64)

    for images, labels in data_loader:
        images = images.to(
            device,
            non_blocking=device.type == "cuda",
        )
        labels = labels.to(
            device,
            non_blocking=device.type == "cuda",
        )

        logits = model(images)
        loss = criterion(logits, labels)
        predictions = logits.argmax(dim=1)

        batch_size = labels.size(0)
        total_loss += loss.item() * batch_size
        total_correct += (predictions == labels).sum().item()
        total_samples += batch_size

        labels_cpu = labels.cpu()
        predictions_cpu = predictions.cpu()

        class_total += torch.bincount(
            labels_cpu,
            minlength=num_classes,
        )
        class_correct += torch.bincount(
            labels_cpu[predictions_cpu == labels_cpu],
            minlength=num_classes,
        )

        flattened_indices = labels_cpu * num_classes + predictions_cpu
        confusion_matrix += torch.bincount(
            flattened_indices,
            minlength=num_classes * num_classes,
        ).reshape(num_classes, num_classes)

    average_loss = total_loss / total_samples
    accuracy = 100.0 * total_correct / total_samples

    return (
        average_loss,
        accuracy,
        confusion_matrix,
        class_correct,
        class_total,
    )
