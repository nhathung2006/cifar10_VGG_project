import random
from typing import Tuple

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms

from config import (
    BATCH_SIZE,
    CIFAR10_MEAN,
    CIFAR10_STD,
    CLASS_NAMES,
    DATA_DIR,
    NUM_WORKERS,
    SEED,
    VAL_RATIO,
)


def _validate_imagefolder_structure() -> None:
    """
    Kiểm tra thư mục ImageFolder và mapping nhãn của train/test.
    """
    train_dir = DATA_DIR / "train"
    test_dir = DATA_DIR / "test"
    expected_classes = list(CLASS_NAMES)

    for split_name, split_dir in (
        ("train", train_dir),
        ("test", test_dir),
    ):
        if not split_dir.is_dir():
            raise FileNotFoundError(
                f"Không tìm thấy thư mục {split_name}: {split_dir}"
            )

        actual_classes = sorted(
            path.name
            for path in split_dir.iterdir()
            if path.is_dir()
        )
        if actual_classes != expected_classes:
            missing_classes = [
                name for name in expected_classes if name not in actual_classes
            ]
            extra_classes = [
                name for name in actual_classes if name not in expected_classes
            ]
            raise ValueError(
                f"Sai lớp trong thư mục {split_name}: {split_dir}. "
                f"Thiếu: {missing_classes or 'không có'}. "
                f"Thừa: {extra_classes or 'không có'}. "
                f"Đang có: {actual_classes}. "
                f"Mong đợi: {expected_classes}."
            )

    train_dataset = datasets.ImageFolder(root=train_dir)
    test_dataset = datasets.ImageFolder(root=test_dir)
    expected_mapping = {
        class_name: class_index
        for class_index, class_name in enumerate(expected_classes)
    }

    if train_dataset.classes != test_dataset.classes:
        raise ValueError(
            "Danh sách lớp của train và test không giống nhau: "
            f"train={train_dataset.classes}, test={test_dataset.classes}."
        )

    for split_name, dataset in (
        ("train", train_dataset),
        ("test", test_dataset),
    ):
        if dataset.classes != expected_classes:
            raise ValueError(
                f"Thứ tự lớp của {split_name} không đúng: "
                f"{dataset.classes}. Mong đợi: {expected_classes}."
            )
        if dataset.class_to_idx != expected_mapping:
            raise ValueError(
                f"Mapping nhãn của {split_name} không đúng: "
                f"{dataset.class_to_idx}. Mong đợi: {expected_mapping}."
            )


def get_train_transform() -> transforms.Compose:
    return transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandAugment(num_ops=2, magnitude=7),
        transforms.ToTensor(),
        transforms.Normalize(CIFAR10_MEAN, CIFAR10_STD),
        transforms.RandomErasing(
            p=0.25,
            scale=(0.02, 0.15),
        ),
    ])


def get_eval_transform() -> transforms.Compose:
    """
    Validation và test không dùng augmentation.
    """
    return transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize(CIFAR10_MEAN, CIFAR10_STD),
        ]
    )


def get_prediction_transform() -> transforms.Compose:
    """
    Transform cho ảnh bên ngoài khi chạy predict.py.
    Resize bảo đảm ảnh có kích thước 32 x 32.
    """
    return transforms.Compose(
        [
            transforms.Resize((32, 32)),
            transforms.ToTensor(),
            transforms.Normalize(CIFAR10_MEAN, CIFAR10_STD),
        ]
    )


def seed_worker(worker_id: int) -> None:
    """
    Giúp các DataLoader worker có kết quả tái lập.
    """
    del worker_id
    worker_seed = torch.initial_seed() % (2**32)
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def create_stratified_indices(
    targets: list[int],
    val_ratio: float,
    seed: int,
) -> Tuple[list[int], list[int]]:
    """
    Chia train/validation theo từng lớp để mỗi lớp có tỷ lệ tương đương.

    CIFAR-10 train có 5.000 ảnh/lớp.
    Với val_ratio=0.1:
    - Train: 4.500 ảnh/lớp
    - Validation: 500 ảnh/lớp
    """
    targets_array = np.asarray(targets)
    rng = np.random.default_rng(seed)

    train_indices: list[int] = []
    val_indices: list[int] = []

    for class_id in np.unique(targets_array):
        class_indices = np.where(targets_array == class_id)[0]
        rng.shuffle(class_indices)

        val_count = int(len(class_indices) * val_ratio)
        val_indices.extend(class_indices[:val_count].tolist())
        train_indices.extend(class_indices[val_count:].tolist())

    rng.shuffle(train_indices)
    rng.shuffle(val_indices)

    return train_indices, val_indices


def create_dataloaders(
    device: torch.device,
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """
    Đọc CIFAR-10 dạng ImageFolder và tạo ba DataLoader:
    - train
    - validation
    - test
    """
    _validate_imagefolder_structure()

    # Dataset này dùng để lấy nhãn phục vụ chia dữ liệu.
    split_dataset = datasets.ImageFolder(root=DATA_DIR / "train")

    train_indices, val_indices = create_stratified_indices(
        targets=split_dataset.targets,
        val_ratio=VAL_RATIO,
        seed=SEED,
    )

    # Tạo hai instance riêng để train và validation dùng transform khác nhau.
    full_train_dataset = datasets.ImageFolder(
        root=DATA_DIR / "train",
        transform=get_train_transform(),
    )

    full_val_dataset = datasets.ImageFolder(
        root=DATA_DIR / "train",
        transform=get_eval_transform(),
    )

    test_dataset = datasets.ImageFolder(
        root="/kaggle/working/cifar10_data/cifar10/test",
        transform=get_eval_transform(),
    )

    train_dataset = Subset(full_train_dataset, train_indices)
    val_dataset = Subset(full_val_dataset, val_indices)

    loader_generator = torch.Generator()
    loader_generator.manual_seed(SEED)

    common_options = {
        "batch_size": BATCH_SIZE,
        "num_workers": NUM_WORKERS,
        "pin_memory": device.type == "cuda",
        "persistent_workers": NUM_WORKERS > 0,
        "worker_init_fn": seed_worker,
    }

    train_loader = DataLoader(
        train_dataset,
        shuffle=True,
        generator=loader_generator,
        **common_options,
    )

    val_loader = DataLoader(
        val_dataset,
        shuffle=False,
        **common_options,
    )

    test_loader = DataLoader(
        test_dataset,
        shuffle=False,
        **common_options,
    )

    return train_loader, val_loader, test_loader


def create_test_loader(device: torch.device) -> DataLoader:
    """
    Chỉ tạo test loader, dùng cho evaluate.py.
    """
    _validate_imagefolder_structure()

    test_dataset = datasets.ImageFolder(
        root=DATA_DIR / "test",
        transform=get_eval_transform(),
    )

    return DataLoader(
        test_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=device.type == "cuda",
        persistent_workers=NUM_WORKERS > 0,
        worker_init_fn=seed_worker,
    )
