from pathlib import Path

import torch
from PIL import Image
from torch.utils.data import DataLoader
from torchvision import datasets

from config import (
    BATCH_SIZE,
    CHECKPOINT_PATH,
    CLASS_NAMES,
    DATA_DIR,
    NUM_CLASSES,
    NUM_WORKERS,
    OUTPUT_DIR,
)
from data import get_eval_transform
from model import VGG16CIFAR10
from utils import get_device, load_checkpoint


CONFUSION_PAIRS = (
    ("bird", "airplane"),
    ("dog", "cat"),
    ("cat", "dog"),
    ("airplane", "ship"),
    ("bird", "deer"),
    ("cat", "frog"),
)
EXAMPLES_PER_PAIR = 9


def main() -> None:
    device = get_device()
    test_dataset = datasets.ImageFolder(
        root=DATA_DIR / "test",
        transform=get_eval_transform(),
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=device.type == "cuda",
        persistent_workers=NUM_WORKERS > 0,
    )

    model = VGG16CIFAR10(
        num_classes=NUM_CLASSES,
        dropout=0.5,
    ).to(device)
    load_checkpoint(
        path=CHECKPOINT_PATH,
        model=model,
        device=device,
    )
    model.eval()

    pair_counts = {
        pair: 0
        for pair in CONFUSION_PAIRS
    }
    class_names = tuple(CLASS_NAMES)

    with torch.inference_mode():
        sample_index = 0
        for images, labels in test_loader:
            images = images.to(
                device,
                non_blocking=device.type == "cuda",
            )
            predictions = model(images).argmax(dim=1).cpu()
            labels = labels.cpu()

            for batch_index in range(labels.size(0)):
                true_name = class_names[int(labels[batch_index])]
                predicted_name = class_names[int(predictions[batch_index])]
                pair = (true_name, predicted_name)

                if pair not in pair_counts:
                    sample_index += 1
                    continue
                if pair_counts[pair] >= EXAMPLES_PER_PAIR:
                    sample_index += 1
                    continue

                source_path = Path(test_dataset.samples[sample_index][0])
                output_dir = OUTPUT_DIR / "confusion_examples" / (
                    f"{true_name}_{predicted_name}"
                )
                output_dir.mkdir(parents=True, exist_ok=True)
                output_path = output_dir / f"{pair_counts[pair] + 1:02d}.png"

                with Image.open(source_path) as image:
                    image.convert("RGB").save(output_path)

                pair_counts[pair] += 1
                sample_index += 1

    missing_pairs = [
        f"{true_name}_{predicted_name} ({count}/{EXAMPLES_PER_PAIR})"
        for (true_name, predicted_name), count in pair_counts.items()
        if count < EXAMPLES_PER_PAIR
    ]
    if missing_pairs:
        raise RuntimeError(
            "Không đủ ảnh cho các cặp nhầm lẫn: "
            + ", ".join(missing_pairs)
        )

    print("Đã xuất ảnh vào outputs/confusion_examples/")
    for true_name, predicted_name in CONFUSION_PAIRS:
        print(f"- {true_name}_{predicted_name}: {EXAMPLES_PER_PAIR} ảnh")


if __name__ == "__main__":
    main()
