import torch
from torch import nn


class VGG16CIFAR10(nn.Module):
    """
    VGG16-BN được điều chỉnh cho ảnh CIFAR-10 32 x 32.

    Các block có số convolution lần lượt là 2-2-3-3-3 và trả về logits.
    """

    def __init__(
        self,
        num_classes: int = 10,
    ) -> None:
        super().__init__()

        block_channels = (64, 128, 256, 512, 512)
        block_layers = (2, 2, 3, 3, 3)
        layers: list[nn.Module] = []
        in_channels = 3

        for out_channels, num_convolutions in zip(
            block_channels,
            block_layers,
        ):
            for _ in range(num_convolutions):
                layers.extend(
                    [
                        nn.Conv2d(
                            in_channels=in_channels,
                            out_channels=out_channels,
                            kernel_size=3,
                            padding=1,
                        ),
                        nn.BatchNorm2d(out_channels),
                        nn.ReLU(inplace=True),
                    ]
                )
                in_channels = out_channels

            layers.append(nn.MaxPool2d(kernel_size=2, stride=2))

        self.features = nn.Sequential(*layers)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(512, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = self.avgpool(x)
        return self.classifier(x)


if __name__ == "__main__":
    model = VGG16CIFAR10(num_classes=10, )
    sample = torch.randn(8, 3, 32, 32)
    logits = model(sample)

    print(model)
    print(f"Input shape : {tuple(sample.shape)}")
    print(f"Output shape: {tuple(logits.shape)}")
    print(
        "Trainable parameters:",
        sum(p.numel() for p in model.parameters() if p.requires_grad),
        )
