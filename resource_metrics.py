import io
import time
from typing import Any

import torch
from torch import nn


def check_thop_available() -> None:
    """Báo lỗi rõ ràng trước khi bắt đầu train nếu thiếu thop."""
    try:
        import thop  # noqa: F401
    except ImportError as error:
        raise RuntimeError(
            "Thiếu thư viện thop. Hãy cài đặt bằng: "
            "pip install 'thop>=0.1.1' trước khi chạy train.py."
        ) from error


def synchronize_cuda(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def get_parameter_counts(model: nn.Module) -> tuple[int, int]:
    total = sum(parameter.numel() for parameter in model.parameters())
    trainable = sum(
        parameter.numel()
        for parameter in model.parameters()
        if parameter.requires_grad
    )
    return total, trainable


def get_peak_vram_mb(device: torch.device) -> float | None:
    if device.type != "cuda":
        return None
    return torch.cuda.max_memory_allocated(device) / (1024 ** 2)


def profile_model(model: nn.Module) -> tuple[float, float]:
    check_thop_available()
    from thop import profile

    model = model.to("cpu")
    model.eval()
    sample = torch.zeros(1, 3, 32, 32)
    macs, _ = profile(model, inputs=(sample,), verbose=False)
    macs = float(macs)
    # Theo quy ước sử dụng ở đây, một MAC tương đương hai FLOPs.
    flops = 2.0 * macs
    return macs, flops


def get_model_size_mb(model: nn.Module) -> float:
    buffer = io.BytesIO()
    torch.save(model.state_dict(), buffer)
    return buffer.tell() / (1024 ** 2)


def build_resource_metrics(
    model: nn.Module,
    training_time_seconds: float,
    inference_time_seconds: float,
    peak_vram_mb: float | None,
    best_epoch: int,
    test_loss: float,
    test_accuracy: float,
) -> dict[str, Any]:
    total_parameters, trainable_parameters = get_parameter_counts(model)
    macs, flops = profile_model(model)
    return {
        "parameters": total_parameters,
        "trainable_parameters": trainable_parameters,
        "training_time_seconds": training_time_seconds,
        "inference_time_seconds": inference_time_seconds,
        "peak_vram_mb": peak_vram_mb,
        "macs": macs,
        "flops": flops,
        "model_size_mb": get_model_size_mb(model),
        "best_epoch": best_epoch,
        "test_loss": test_loss,
        "test_accuracy": test_accuracy,
        "input_shape": [1, 3, 32, 32],
        "macs_flops_convention": "1 MAC = 2 FLOPs",
    }


def now() -> float:
    return time.perf_counter()
