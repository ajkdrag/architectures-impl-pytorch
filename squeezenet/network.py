import torch
from torch import nn


class Fire(nn.Module):
    def __init__(
        self, in_channels_squeeze, out_channels_squeeze, out_channels_expand
    ) -> None:
        super().__init__()
        self.squeeze = nn.Sequential(
            nn.Conv2d(
                in_channels=in_channels_squeeze,
                out_channels=out_channels_squeeze,
                kernel_size=1,
            ),
            nn.ReLU(),
        )
        self.expand_1 = nn.Sequential(
            nn.Conv2d(
                in_channels=out_channels_squeeze,
                out_channels=out_channels_expand,
                kernel_size=1,
            ),
            nn.ReLU(),
        )
        self.expand_2 = nn.Sequential(
            nn.Conv2d(
                in_channels=out_channels_squeeze,
                out_channels=out_channels_expand,
                kernel_size=3,
                padding=1,
            ),
            nn.ReLU(),
        )

    def forward(self, x):
        squeezed = self.squeeze(x)
        expanded_1 = self.expand_1(squeezed)
        expanded_2 = self.expand_2(squeezed)
        return torch.cat([expanded_1, expanded_2], dim=1)


class SqueezeNet(nn.Module):
    def __init__(self, in_channels=3, num_classes=1000) -> None:
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv2d(
                in_channels=in_channels, out_channels=96, kernel_size=7, stride=2, padding=3
            ),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, stride=2),
        )

        self.trunk = nn.Sequential(
            Fire(96, 16, 64),
            Fire(128, 16, 64),
            Fire(128, 32, 128),
            nn.MaxPool2d(kernel_size=3, stride=2, ceil_mode=True),
            Fire(256, 32, 128),
            Fire(256, 48, 192),
            Fire(384, 48, 192),
            Fire(384, 64, 256),
            nn.MaxPool2d(kernel_size=3, stride=2, ceil_mode=True),
            Fire(512, 64, 256),
            nn.Conv2d(in_channels=512, out_channels=num_classes, kernel_size=1),
        )

        self.classifier = nn.Sequential(
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten()
        )

    def forward(self, x):
        stem = self.stem(x)
        print(stem.shape)
        trunk = self.trunk(stem)
        print(trunk.shape)
        out = self.classifier(trunk)
        print(out.shape)
        return out


if __name__ == "__main__":
    net = SqueezeNet(num_classes=1000)
    print(net)
    with torch.no_grad():
        net.eval()
        ip = torch.rand((1, 3, 224, 224))
        got = net(ip)
        print(
            "Num trainable params: ",
            sum(p.numel() for p in net.parameters() if p.requires_grad),
        )

