from pathlib import Path

import matplotlib.pyplot as plt
import torch
from PIL import Image

from config import CLASS_NAMES, CIFAR10_MEAN, CIFAR10_STD, OUTPUT_DIR, SEED
from data import create_dataloaders
from utils import get_device, set_seed


IMAGE_COUNT = 50
IMAGE_SIZE = (32, 32)


def main() -> None:
    set_seed(SEED)
    device = get_device()
    train_loader, _, _ = create_dataloaders(device)

    output_dir = OUTPUT_DIR / "training_inputs"
    output_dir.mkdir(parents=True, exist_ok=True)

    mean = torch.tensor(CIFAR10_MEAN, dtype=torch.float32).view(3, 1, 1)
    std = torch.tensor(CIFAR10_STD, dtype=torch.float32).view(3, 1, 1)
    grid_images = []
    grid_labels = []
    saved_count = 0

    for images, labels in train_loader:
        print(f"Kích thước tensor batch: {tuple(images.shape)}")

        for batch_index in range(images.size(0)):
            if saved_count >= IMAGE_COUNT:
                break

            input_tensor = images[batch_index].clone().cpu()
            if tuple(input_tensor.shape) != (3, 32, 32):
                raise ValueError(
                    f"Kích thước tensor không đúng: {tuple(input_tensor.shape)}"
                )

            image_tensor = (input_tensor * std + mean).clamp(0.0, 1.0)
            image_array = (
                image_tensor.permute(1, 2, 0)
                .mul(255)
                .round()
                .to(torch.uint8)
                .numpy()
            )

            class_name = CLASS_NAMES[int(labels[batch_index])]
            output_path = output_dir / f"{saved_count + 1:03d}_{class_name}.png"
            Image.fromarray(image_array, mode="RGB").save(output_path)

            saved_image_size = Image.open(output_path).size
            print(
                f"{output_path.name}: tensor {tuple(input_tensor.shape)}, "
                f"PNG size {saved_image_size}"
            )
            if saved_image_size != IMAGE_SIZE:
                raise ValueError(
                    f"Ảnh không đúng kích thước {IMAGE_SIZE}: {saved_image_size}"
                )

            grid_images.append(image_array)
            grid_labels.append(class_name)
            saved_count += 1

        if saved_count >= IMAGE_COUNT:
            break

    if saved_count != IMAGE_COUNT:
        raise RuntimeError(
            f"Chỉ xuất được {saved_count}/{IMAGE_COUNT} ảnh training."
        )

    figure, axes = plt.subplots(5, 10, figsize=(10, 5))
    for index, (image_array, class_name) in enumerate(
        zip(grid_images, grid_labels)
    ):
        axis = axes.flat[index]
        axis.imshow(image_array, interpolation="nearest")
        axis.set_title(class_name, fontsize=6)
        axis.axis("off")

    figure.tight_layout(pad=0.2)
    figure.savefig(OUTPUT_DIR / "training_inputs_grid.png", dpi=100)
    plt.close(figure)


if __name__ == "__main__":
    main()
