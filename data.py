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
    DATA_DIR,
    NUM_WORKERS,
    SEED,
    VAL_RATIO,
)


def _validate_local_cifar10() -> None:
    """
    Kiểm tra bộ CIFAR-10 dạng gốc trước khi tạo bất kỳ Dataset nào.
    """
    cifar10_dir = DATA_DIR / "cifar-10-batches-py"
    required_files = (
        "batches.meta",
        "data_batch_1",
        "data_batch_2",
        "data_batch_3",
        "data_batch_4",
        "data_batch_5",
        "test_batch",
    )
    missing_files = [
        cifar10_dir / filename
        for filename in required_files
        if not (cifar10_dir / filename).is_file()
    ]

    if missing_files:
        missing_paths = "\n".join(f"  - {path}" for path in missing_files)
        raise FileNotFoundError(
            "Không tìm thấy đầy đủ dữ liệu CIFAR-10 cục bộ.\n"
            f"Thư mục chương trình đang tìm: {cifar10_dir}\n"
            f"File còn thiếu:\n{missing_paths}\n"
            "Cấu trúc thư mục mong đợi:\n"
            f"{DATA_DIR}/\n"
            "└── cifar-10-batches-py/\n"
            "    ├── batches.meta\n"
            "    ├── data_batch_1\n"
            "    ├── data_batch_2\n"
            "    ├── data_batch_3\n"
            "    ├── data_batch_4\n"
            "    ├── data_batch_5\n"
            "    └── test_batch"
        )


def get_train_transform() -> transforms.Compose:
    """
    Data augmentation chỉ áp dụng cho tập train.
    """
    return transforms.Compose(
        [
            transforms.RandomCrop(size=32, padding=4),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.ToTensor(),
            transforms.Normalize(CIFAR10_MEAN, CIFAR10_STD),
        ]
    )


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
    Tải CIFAR-10 và tạo ba DataLoader:
    - train
    - validation
    - test
    """
    _validate_local_cifar10()

    # Dataset này dùng để lấy nhãn phục vụ chia dữ liệu.
    split_dataset = datasets.CIFAR10(
        root=DATA_DIR,
        train=True,
        download=False,
    )

    train_indices, val_indices = create_stratified_indices(
        targets=split_dataset.targets,
        val_ratio=VAL_RATIO,
        seed=SEED,
    )

    # Tạo hai instance riêng để train và validation dùng transform khác nhau.
    full_train_dataset = datasets.CIFAR10(
        root=DATA_DIR,
        train=True,
        download=False,
        transform=get_train_transform(),
    )

    full_val_dataset = datasets.CIFAR10(
        root=DATA_DIR,
        train=True,
        download=False,
        transform=get_eval_transform(),
    )

    test_dataset = datasets.CIFAR10(
        root=DATA_DIR,
        train=False,
        download=False,
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
    _validate_local_cifar10()

    test_dataset = datasets.CIFAR10(
        root=DATA_DIR,
        train=False,
        download=False,
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
