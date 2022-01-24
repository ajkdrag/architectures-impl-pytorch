import torch
from torch import nn


class Shuffle(nn.Module):
    def __init__(self, num_groups) -> None:
        super().__init__()
        self.num_groups = num_groups

    def forward(self, x):
        n, c, h, w = x.size()
        return (
            x.view(n, self.num_groups, c // self.num_groups, h, w)
            .permute(0, 2, 1, 3, 4)
            .reshape(n, c, h, w)
        )


class GConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, groups) -> None:
        super().__init__()
        self.gconv = nn.Sequential(
            nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size=kernel_size,
                groups=groups,
                bias=False,
            ),
            nn.BatchNorm2d(out_channels),
        )

    def forward(self, x):
        return self.gconv(x)


class DConvBlock(nn.Module):
    def __init__(self, in_channels, kernel_size, stride, padding) -> None:
        super().__init__()
        self.dconv = nn.Sequential(
            nn.Conv2d(
                in_channels,
                in_channels,
                kernel_size=kernel_size,
                stride=stride,
                padding=padding,
                groups=in_channels,
                bias=False,
            ),
            nn.BatchNorm2d(in_channels),
        )

    def forward(self, x):
        return self.dconv(x)


class ShufflenetBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride, groups) -> None:
        super().__init__()
        self.stride = stride
        mid_channels = out_channels // 4
        self.bottleneck = nn.Sequential(
            GConvBlock(in_channels, mid_channels, 1, groups),
            nn.ReLU(),
            Shuffle(groups),
            DConvBlock(mid_channels, kernel_size=3, stride=stride, padding=1),
            nn.ReLU(),
            GConvBlock(mid_channels, out_channels, 1, groups),
        )

        shortcut_layers = []
        if stride == 2:
            shortcut_layers.append(nn.AvgPool2d(kernel_size=3, stride=2, padding=1))
        self.shortcut = nn.Sequential(*shortcut_layers)

    def forward(self, x):
        tower_1 = self.bottleneck(x)
        tower_2 = self.shortcut(x)
        out = (
            torch.cat([tower_1, tower_2], dim=1)
            if self.stride == 2
            else tower_1 + tower_2
        )
        return nn.ReLU()(out)


class ShuffleNet(nn.Module):
    def __init__(self, in_channels=3, num_classes=1000) -> None:
        super().__init__()
        groups = 3
        out_channels_list = [240, 480, 960]
        reps_list = [4, 8, 4]

        self.stem = nn.Sequential(
            nn.Conv2d(in_channels, 24, kernel_size=3, stride=2, padding=1),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
        )

        trunk_layers = []
        in_channels = 24
        for out_channels, reps in zip(out_channels_list, reps_list):
            bottleneck, in_channels = self.create_bottleneck(
                in_channels, out_channels, reps, groups
            )
            trunk_layers.append(bottleneck)

        self.trunk = nn.Sequential(*trunk_layers)
        self.pooler = nn.AdaptiveAvgPool2d((1, 1))
        self.classifier = nn.Sequential(
            nn.Flatten(), nn.Linear(in_channels, num_classes)
        )

    def create_bottleneck(self, in_channels, out_channels, reps, groups):
        layers = []
        for i in range(reps):
            stride = 2 if i == 0 else 1
            layers.append(
                ShufflenetBlock(
                    in_channels,
                    out_channels - in_channels if i == 0 else out_channels,
                    stride=stride,
                    groups=groups,
                )
            )
            in_channels = out_channels
        return nn.Sequential(*layers), out_channels

    def forward(self, x):
        stem = self.stem(x)
        print(stem.shape)
        trunk = self.trunk(stem)
        print(trunk.shape)
        pooled = self.pooler(trunk)
        print(pooled.shape)
        out = self.classifier(pooled)
        return out


if __name__ == "__main__":
    net = ShuffleNet()
    print(net)
    with torch.no_grad():
        net.eval()
        ip = torch.rand((1, 3, 224, 224))
        got = net(ip)
        print(
            "Num trainable params: ",
            sum(p.numel() for p in net.parameters() if p.requires_grad),
        )

