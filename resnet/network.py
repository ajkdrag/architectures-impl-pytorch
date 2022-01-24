import torch
from torch import nn


class ConvBlock(nn.Module):
    def __init__(self, **kwargs) -> None:
        super().__init__()
        self.conv_block = nn.Sequential(
            nn.Conv2d(**kwargs), nn.BatchNorm2d(kwargs["out_channels"]), nn.ReLU()
        )

    def forward(self, x):
        return self.conv_block(x)


class IdentityBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1) -> None:
        super().__init__()

        self.conv_block = nn.Sequential(
            ConvBlock(
                in_channels=in_channels,
                out_channels=out_channels // 4,
                kernel_size=1,
                stride=stride,
            ),
            ConvBlock(
                in_channels=out_channels // 4,
                out_channels=out_channels // 4,
                kernel_size=3,
                stride=1,
                padding=1,
            ),
            ConvBlock(
                in_channels=out_channels // 4,
                out_channels=out_channels,
                kernel_size=1,
                stride=1,
            ),
            nn.BatchNorm2d(out_channels),
        )

    def forward(self, x):
        left_branch = self.conv_block(x)
        return nn.ReLU()(torch.add(left_branch, x))


class ProjectionBlock(IdentityBlock):
    def __init__(self, in_channels, out_channels, stride) -> None:
        super().__init__(in_channels, out_channels, stride)

        self.conv_block_2 = nn.Sequential(
            ConvBlock(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=1,
                stride=stride,
            ),
            nn.BatchNorm2d(out_channels),
        )

    def forward(self, x):
        left_branch = self.conv_block(x)
        right_branch = self.conv_block_2(x)
        return nn.ReLU()(torch.add(left_branch, right_branch))


class ResnetBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride, reps) -> None:
        super().__init__()

        layers = [ProjectionBlock(in_channels, out_channels, stride)]
        for _ in range(reps):
            layers.append(IdentityBlock(out_channels, out_channels, 1))

        self.resnet_block = nn.Sequential(*layers)

    def forward(self, x):
        return self.resnet_block(x)


class Renset(nn.Module):
    def __init__(self, in_channels=3, num_classes=1000) -> None:
        super().__init__()

        self.stem = nn.Sequential(
            ConvBlock(
                in_channels=in_channels,
                out_channels=64,
                kernel_size=7,
                stride=2,
                padding=3,
            ),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
        )

        self.trunk = nn.Sequential(
            ResnetBlock(in_channels=64, out_channels=256, stride=1, reps=3),
            ResnetBlock(in_channels=256, out_channels=512, stride=2, reps=4),
            ResnetBlock(in_channels=512, out_channels=1024, stride=2, reps=6),
            ResnetBlock(in_channels=1024, out_channels=2048, stride=2, reps=3),
        )

        self.pooler = nn.AdaptiveAvgPool2d(output_size=(1, 1))
        self.classifier = nn.Sequential(
            nn.Flatten(), nn.Linear(in_features=2048, out_features=num_classes)
        )

    def forward(self, x):
        head = self.stem(x)
        print(head.shape)
        trunk = self.trunk(head)
        print(trunk.shape)
        pooled = self.pooler(trunk)
        print(pooled.shape)
        out = self.classifier(pooled)
        return out


if __name__ == "__main__":
    net = Renset(num_classes=10)
    print(net)
    with torch.no_grad():
        net.eval()
        ip = torch.rand((1, 3, 224, 224))
        got = net(ip)
        print(got.shape)

