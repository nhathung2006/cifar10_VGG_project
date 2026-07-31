import argparse
from pathlib import Path

import torch
from PIL import Image

from config import CHECKPOINT_PATH, CLASS_NAMES, NUM_CLASSES
from data import get_prediction_transform
from model import VGG16CIFAR10
from utils import get_device, load_checkpoint


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Dự đoán một ảnh bằng VGG16 CIFAR-10."
    )
    parser.add_argument(
        "--image",
        type=Path,
        required=True,
        help="Đường dẫn đến ảnh cần dự đoán.",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=3,
        help="Số kết quả có xác suất cao nhất cần hiển thị.",
    )
    return parser.parse_args()


@torch.inference_mode()
def predict(image_path: Path, top_k: int) -> None:
    if not image_path.exists():
        raise FileNotFoundError(
            f"Không tìm thấy ảnh: {image_path}"
        )

    device = get_device()

    model = VGG16CIFAR10(
        num_classes=NUM_CLASSES,
        #dropout=0.5,
    ).to(device)
    load_checkpoint(
        path=CHECKPOINT_PATH,
        model=model,
        device=device,
    )
    model.eval()

    image = Image.open(image_path).convert("RGB")
    transform = get_prediction_transform()

    input_tensor = transform(image).unsqueeze(0).to(device)

    logits = model(input_tensor)

    # Softmax chỉ dùng khi cần hiển thị xác suất ở bước suy luận.
    probabilities = torch.softmax(logits, dim=1)[0]

    top_k = max(1, min(top_k, NUM_CLASSES))
    top_probabilities, top_indices = torch.topk(
        probabilities,
        k=top_k,
    )

    print(f"Ảnh: {image_path}")
    print(f"Thiết bị: {device}")
    print("\nKết quả dự đoán:")

    for rank, (probability, class_index) in enumerate(
        zip(top_probabilities, top_indices),
        start=1,
    ):
        class_name = CLASS_NAMES[int(class_index.item())]
        percentage = float(probability.item()) * 100.0

        print(
            f"{rank}. {class_name:10s}: "
            f"{percentage:.2f}%"
        )


def main() -> None:
    args = parse_args()
    predict(
        image_path=args.image,
        top_k=args.top_k,
    )


if __name__ == "__main__":
    main()
