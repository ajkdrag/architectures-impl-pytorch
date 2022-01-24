from multiprocessing import pool
import torch
from torch import nn


class MobileNetBlock(nn.Module):
    def __init__(self, in_channels, out_channels, **kwargs) -> None:
        super().__init__()
        self.dconv_block = nn.Sequential(
            nn.Conv2d(
                in_channels, in_channels, **kwargs, groups=in_channels, bias=False
            ),
            nn.BatchNorm2d(in_channels),
            nn.ReLU(),
        )
        self.conv_block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
        )

    def forward(self, x):
        return self.conv_block(self.dconv_block(x))


class MobileNet(nn.Module):
    def __init__(self, in_channels=3, num_classes=1000) -> None:
        super().__init__()

        self.stem = nn.Sequential(
            nn.Conv2d(
                in_channels,
                out_channels=32,
                kernel_size=3,
                stride=2,
                padding=1,
                bias=False,
            ),
            nn.BatchNorm2d(32),
            nn.ReLU(),
        )

        self.trunk = nn.Sequential(
            MobileNetBlock(32, 64, kernel_size=3, stride=1, padding=1),
            MobileNetBlock(64, 128, kernel_size=3, stride=2, padding=1),
            MobileNetBlock(128, 128, kernel_size=3, stride=1, padding=1),
            MobileNetBlock(128, 256, kernel_size=3, stride=2, padding=1),
            MobileNetBlock(256, 256, kernel_size=3, stride=1, padding=1),
            MobileNetBlock(256, 512, kernel_size=3, stride=2, padding=1),
            MobileNetBlock(512, 512, kernel_size=3, stride=1, padding=1),
            MobileNetBlock(512, 512, kernel_size=3, stride=1, padding=1),
            MobileNetBlock(512, 512, kernel_size=3, stride=1, padding=1),
            MobileNetBlock(512, 512, kernel_size=3, stride=1, padding=1),
            MobileNetBlock(512, 512, kernel_size=3, stride=1, padding=1),
            MobileNetBlock(512, 1024, kernel_size=3, stride=2, padding=1),
            MobileNetBlock(1024, 1024, kernel_size=3, stride=1, padding=1),
        )

        self.pooler = nn.AvgPool2d(kernel_size=7, stride=1)
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(in_features=1024, out_features=num_classes)
        )

    def forward(self, x):
        head = self.stem(x)
        trunk = self.trunk(head)
        print(trunk.shape)
        pooled = self.pooler(trunk)
        print(pooled.shape)
        out = self.classifier(pooled)

        return out


if __name__ == "__main__":
    net = MobileNet(num_classes=10)
    print(net)
    with torch.no_grad():
        net.eval()
        ip = torch.rand((1, 3, 224, 224))
        got = net(ip)
        print(got.shape)

