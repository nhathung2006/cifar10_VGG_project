import torch
from torch import nn


class LeNetCIFAR10(nn.Module):
    """
    LeNet được điều chỉnh cho ảnh CIFAR-10:
    - Đầu vào: 3 x 32 x 32
    - Đầu ra: 10 logits
    """

    def __init__(self, num_classes: int = 10) -> None:
        super().__init__()

        self.features = nn.Sequential(
            # 3 x 32 x 32 -> 6 x 28 x 28
            nn.Conv2d(
                in_channels=3,
                out_channels=6,
                kernel_size=5,
                stride=1,
                padding=0,
            ),
            nn.ReLU(inplace=True),

            # 6 x 28 x 28 -> 6 x 14 x 14
            nn.MaxPool2d(kernel_size=2, stride=2),

            # 6 x 14 x 14 -> 16 x 10 x 10
            nn.Conv2d(
                in_channels=6,
                out_channels=16,
                kernel_size=5,
                stride=1,
                padding=0,
            ),
            nn.ReLU(inplace=True),

            # 16 x 10 x 10 -> 16 x 5 x 5
            nn.MaxPool2d(kernel_size=2, stride=2),
        )

        self.classifier = nn.Sequential(
            nn.Flatten(),

            # 16 x 5 x 5 = 400
            nn.Linear(16 * 5 * 5, 120),
            nn.ReLU(inplace=True),

            nn.Linear(120, 84),
            nn.ReLU(inplace=True),

            # Trả về logits, không thêm Softmax khi train
            nn.Linear(84, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        return self.classifier(x)


if __name__ == "__main__":
    model = LeNetCIFAR10(num_classes=10)
    sample = torch.randn(8, 3, 32, 32)
    logits = model(sample)

    print(model)
    print(f"Input shape : {tuple(sample.shape)}")
    print(f"Output shape: {tuple(logits.shape)}")
    print(
        "Trainable parameters:",
        sum(p.numel() for p in model.parameters() if p.requires_grad),
        )