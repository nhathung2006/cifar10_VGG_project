from torch import nn

from config import (
    CHECKPOINT_PATH,
    CLASS_NAMES,
    NUM_CLASSES,
    TEST_METRICS_PATH,
)
from data import create_test_loader
from engine import evaluate_detailed
from model import VGG16CIFAR10
from utils import (
    ensure_directories,
    get_device,
    load_checkpoint,
    save_confusion_matrix,
    save_json,
)


def main() -> None:
    ensure_directories()

    device = get_device()
    print(f"Thiết bị đang sử dụng: {device}")

    test_loader = create_test_loader(device)

    model = VGG16CIFAR10(
        num_classes=NUM_CLASSES,
        dropout=0.5,
    ).to(device)
    checkpoint = load_checkpoint(
        path=CHECKPOINT_PATH,
        model=model,
        device=device,
    )

    criterion = nn.CrossEntropyLoss()

    (
        test_loss,
        test_accuracy,
        confusion_matrix,
        class_correct,
        class_total,
    ) = evaluate_detailed(
        model=model,
        data_loader=test_loader,
        criterion=criterion,
        device=device,
        num_classes=NUM_CLASSES,
    )

    per_class_accuracy: dict[str, float] = {}

    print("\n===== ĐÁNH GIÁ TEST =====")
    print(f"Checkpoint epoch: {checkpoint['epoch']}")
    print(f"Test loss       : {test_loss:.4f}")
    print(f"Test accuracy   : {test_accuracy:.2f}%")

    print("\nAccuracy theo từng lớp:")
    for class_index, class_name in enumerate(CLASS_NAMES):
        total = int(class_total[class_index].item())
        correct = int(class_correct[class_index].item())
        accuracy = 100.0 * correct / total if total > 0 else 0.0

        per_class_accuracy[class_name] = accuracy
        print(
            f"- {class_name:10s}: "
            f"{accuracy:6.2f}% ({correct}/{total})"
        )

    save_confusion_matrix(
        confusion_matrix=confusion_matrix,
        class_names=CLASS_NAMES,
    )

    save_json(
        TEST_METRICS_PATH,
        {
            "checkpoint_epoch": int(checkpoint["epoch"]),
            "test_loss": float(test_loss),
            "test_accuracy": float(test_accuracy),
            "per_class_accuracy": per_class_accuracy,
            "confusion_matrix": confusion_matrix.tolist(),
        },
    )

    print("\nĐã lưu confusion matrix và metrics trong outputs/.")


if __name__ == "__main__":
    main()
