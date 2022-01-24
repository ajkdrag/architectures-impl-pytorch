import torch
from torch import nn


class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size) -> None:
        super().__init__()
        self.conv_block = nn.Sequential(
            nn.BatchNorm2d(num_features=in_channels),
            nn.ReLU(),
            nn.Conv2d(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=kernel_size,
                padding="same",
            ),
        )

    def forward(self, x):
        return self.conv_block(x)


class DenseBlock(nn.Module):
    def __init__(self, in_channels, out_channels, reps) -> None:
        super().__init__()
        self.layers = nn.ModuleList()
        for idx in range(reps):
            self.layers.append(
                nn.Sequential(
                    ConvBlock(in_channels + idx * out_channels, out_channels * 4, 1),
                    ConvBlock(out_channels * 4, out_channels, 3),
                )
            )

    def forward(self, x):
        for layer in self.layers:
            out = layer(x)
            x = torch.cat([x, out], dim=1)
        return x


class Transition(nn.Module):
    def __init__(self, in_channels, theta) -> None:
        super().__init__()
        out_channels = int(theta * in_channels)
        self.trans_block = nn.Sequential(
            ConvBlock(in_channels, out_channels, 1),
            nn.AvgPool2d(kernel_size=2, stride=2),
        )

    def forward(self, x):
        return self.trans_block(x)


class DenseNet(nn.Module):
    def __init__(self, in_channels=3, num_classes=1000) -> None:
        super().__init__()
        k = 32
        theta = 0.5
        reps_list = [6, 12, 24, 16]

        self.stem = nn.Sequential(
            nn.Conv2d(
                in_channels=in_channels,
                out_channels=2 * k,
                kernel_size=7,
                stride=2,
                padding=3,
            ),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
        )

        self.trunk = nn.ModuleList()

        dense_in = 2 * k
        dense_out = -1
        for idx, reps in enumerate(reps_list):
            self.trunk.append(
                DenseBlock(in_channels=dense_in, out_channels=k, reps=reps)
            )
            if idx != len(reps_list) - 1:
                dense_out = dense_in + reps * k
                self.trunk.append(Transition(in_channels=dense_out, theta=theta))
                dense_in = int(dense_out * theta)

        self.pooler = nn.AdaptiveAvgPool2d((1, 1))
        self.classifier = nn.Sequential(
            nn.Flatten(), nn.Linear(in_features=dense_out, out_features=num_classes)
        )

    def forward(self, x):
        x = self.stem(x)
        print("stem", x.shape)
        for layer in self.trunk:
            x = layer(x)
        print("trunk", x.shape)
        pooled = self.pooler(x)
        print("pooled", pooled.shape)
        out = self.classifier(pooled)
        return out


if __name__ == "__main__":
    net = DenseNet()
    with torch.no_grad():
        net.eval()
        ip = torch.rand((1, 3, 224, 224))
        got = net(ip)
        print(
            "Num trainable params: ",
            sum(p.numel() for p in net.parameters() if p.requires_grad),
        )
