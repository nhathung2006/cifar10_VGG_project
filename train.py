import torch
from torch import nn
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau

from config import (
    CHECKPOINT_PATH,
    CLASS_NAMES,
    EARLY_STOPPING_PATIENCE,
    EPOCHS,
    LEARNING_RATE,
    LR_FACTOR,
    LR_PATIENCE,
    MIN_LEARNING_RATE,
    NUM_CLASSES,
    SEED,
    TEST_METRICS_PATH,
    WEIGHT_DECAY,
)
from data import create_dataloaders
from engine import evaluate, train_one_epoch
from model import VGG16CIFAR10
from utils import (
    ensure_directories,
    get_device,
    load_checkpoint,
    plot_history,
    save_checkpoint,
    save_history_csv,
    save_json,
    set_seed,
)


def main() -> None:
    ensure_directories()
    set_seed(SEED)

    device = get_device()
    print(f"Thiết bị đang sử dụng: {device}")

    train_loader, val_loader, test_loader = create_dataloaders(device)

    print(f"Số ảnh train     : {len(train_loader.dataset)}")
    print(f"Số ảnh validation: {len(val_loader.dataset)}")
    print(f"Số ảnh test      : {len(test_loader.dataset)}")

    model = VGG16CIFAR10(
        num_classes=NUM_CLASSES,
    ).to(device)

    # CrossEntropyLoss nhận trực tiếp logits, không thêm Softmax trong model.
    criterion = nn.CrossEntropyLoss()

    optimizer = Adam(
        model.parameters(),
        lr=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY,
    )

    scheduler = ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=LR_FACTOR,
        patience=LR_PATIENCE,
        min_lr=MIN_LEARNING_RATE,
    )

    history: list[dict[str, float]] = []
    best_val_loss = float("inf")
    epochs_without_improvement = 0

    for epoch in range(1, EPOCHS + 1):
        train_loss, train_accuracy = train_one_epoch(
            model=model,
            data_loader=train_loader,
            criterion=criterion,
            optimizer=optimizer,
            device=device,
        )

        val_loss, val_accuracy = evaluate(
            model=model,
            data_loader=val_loader,
            criterion=criterion,
            device=device,
        )

        # Giảm learning rate khi validation loss dừng cải thiện.
        scheduler.step(val_loss)
        current_lr = optimizer.param_groups[0]["lr"]

        epoch_result = {
            "epoch": float(epoch),
            "learning_rate": float(current_lr),
            "train_loss": float(train_loss),
            "train_accuracy": float(train_accuracy),
            "val_loss": float(val_loss),
            "val_accuracy": float(val_accuracy),
        }
        history.append(epoch_result)

        print(
            f"Epoch {epoch:02d}/{EPOCHS} | "
            f"LR: {current_lr:.7f} | "
            f"Train loss: {train_loss:.4f} | "
            f"Train acc: {train_accuracy:.2f}% | "
            f"Val loss: {val_loss:.4f} | "
            f"Val acc: {val_accuracy:.2f}%"
        )

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            epochs_without_improvement = 0

            save_checkpoint(
                path=CHECKPOINT_PATH,
                model=model,
                optimizer=optimizer,
                epoch=epoch,
                val_loss=val_loss,
                val_accuracy=val_accuracy,
            )

            print(
                "  -> Đã lưu model tốt nhất:",
                CHECKPOINT_PATH.name,
            )
        else:
            epochs_without_improvement += 1
            print(
                "  -> Validation loss chưa cải thiện "
                f"{epochs_without_improvement}/"
                f"{EARLY_STOPPING_PATIENCE} epoch."
            )

        if epochs_without_improvement >= EARLY_STOPPING_PATIENCE:
            print("Dừng sớm để hạn chế overfitting.")
            break

    save_history_csv(history)
    plot_history(history)

    # Nạp lại model tốt nhất trước khi đánh giá test.
    checkpoint = load_checkpoint(
        path=CHECKPOINT_PATH,
        model=model,
        device=device,
    )

    test_loss, test_accuracy = evaluate(
        model=model,
        data_loader=test_loader,
        criterion=criterion,
        device=device,
    )

    test_metrics = {
        "best_epoch": int(checkpoint["epoch"]),
        "best_val_loss": float(checkpoint["val_loss"]),
        "best_val_accuracy": float(checkpoint["val_accuracy"]),
        "test_loss": float(test_loss),
        "test_accuracy": float(test_accuracy),
        "classes": list(CLASS_NAMES),
    }
    save_json(TEST_METRICS_PATH, test_metrics)

    print("\n===== KẾT QUẢ CUỐI CÙNG =====")
    print(f"Best epoch : {checkpoint['epoch']}")
    print(f"Test loss  : {test_loss:.4f}")
    print(f"Test acc   : {test_accuracy:.2f}%")
    print(f"Checkpoint : {CHECKPOINT_PATH}")
    print("Đã lưu biểu đồ và kết quả trong thư mục outputs/.")


if __name__ == "__main__":
    # Cần thiết trên Windows khi DataLoader dùng nhiều worker.
    main()
