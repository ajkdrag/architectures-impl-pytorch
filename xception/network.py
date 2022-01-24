import torch
from torch import nn


class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, **kwargs) -> None:
        super().__init__()
        self.conv_block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, **kwargs, bias=False),
            nn.BatchNorm2d(out_channels),
        )

    def forward(self, x):
        return self.conv_block(x)


class SepConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, **kwargs) -> None:
        super().__init__()
        self.conv_block = nn.Sequential(
            nn.Conv2d(
                in_channels, in_channels, **kwargs, groups=in_channels, bias=False
            ),
            nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_channels),
        )

    def forward(self, x):
        return self.conv_block(x)


class EntryFlow(nn.Module):
    def __init__(self, in_channels) -> None:
        super().__init__()
        self.block_1 = nn.Sequential(
            ConvBlock(
                in_channels=in_channels,
                out_channels=32,
                kernel_size=3,
                stride=2,
                padding=1,
            ),
            nn.ReLU(),
            ConvBlock(in_channels=32, out_channels=64, kernel_size=3, stride=1),
        )

        self.aux_1 = ConvBlock(
            in_channels=64, out_channels=128, kernel_size=1, stride=2
        )
        self.block_2 = nn.Sequential(
            SepConvBlock(
                in_channels=64, out_channels=128, kernel_size=3, stride=1, padding=1
            ),
            nn.ReLU(),
            SepConvBlock(
                in_channels=128, out_channels=128, kernel_size=3, stride=1, padding=1
            ),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
        )
        self.aux_2 = ConvBlock(
            in_channels=128, out_channels=256, kernel_size=1, stride=2
        )
        self.block_3 = nn.Sequential(
            nn.ReLU(),
            SepConvBlock(
                in_channels=128, out_channels=256, kernel_size=3, stride=1, padding=1
            ),
            nn.ReLU(),
            SepConvBlock(
                in_channels=256, out_channels=256, kernel_size=3, stride=1, padding=1
            ),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
        )
        self.aux_3 = ConvBlock(
            in_channels=256, out_channels=728, kernel_size=1, stride=2
        )
        self.block_4 = nn.Sequential(
            nn.ReLU(),
            SepConvBlock(
                in_channels=256, out_channels=728, kernel_size=3, stride=1, padding=1
            ),
            nn.ReLU(),
            SepConvBlock(
                in_channels=728, out_channels=728, kernel_size=3, stride=1, padding=1
            ),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
        )

    def forward(self, x):
        block_1_out = self.block_1(x)
        aux_1_out = self.aux_1(block_1_out)
        block_2_out = torch.add(aux_1_out, self.block_2(block_1_out))

        aux_2_out = self.aux_2(aux_1_out)
        block_3_out = torch.add(aux_2_out, self.block_3(block_2_out))

        aux_3_out = self.aux_3(aux_2_out)
        block_4_out = torch.add(aux_3_out, self.block_4(block_3_out))

        return block_4_out


class MiddleFlow(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.middle_block = nn.ModuleList()
        for _ in range(8):
            middle_block = nn.Sequential(
                nn.ReLU(),
                SepConvBlock(
                    in_channels=728, out_channels=728, kernel_size=3, padding=1
                ),
                nn.ReLU(),
                SepConvBlock(
                    in_channels=728, out_channels=728, kernel_size=3, padding=1
                ),
                nn.ReLU(),
                SepConvBlock(
                    in_channels=728, out_channels=728, kernel_size=3, padding=1
                ),
            )
            self.middle_block.append(middle_block)

    def forward(self, x):
        for layer in self.middle_block:
            out = layer(x)
            x = torch.add(x, out)
        return x


class ExitFlow(nn.Module):
    def __init__(self) -> None:
        super().__init__()

        self.block_1 = nn.Sequential(
            nn.ReLU(),
            SepConvBlock(in_channels=728, out_channels=728, kernel_size=3, padding=1),
            nn.ReLU(),
            SepConvBlock(in_channels=728, out_channels=1024, kernel_size=3, padding=1),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
        )

        self.aux_1 = ConvBlock(
            in_channels=728, out_channels=1024, kernel_size=1, stride=2
        )

        self.block_2 = nn.Sequential(
            SepConvBlock(in_channels=1024, out_channels=1536, kernel_size=3, padding=1),
            nn.ReLU(),
            SepConvBlock(in_channels=1536, out_channels=2048, kernel_size=3, padding=1),
            nn.ReLU(),
        )

    def forward(self, x):
        block_1_out = self.block_1(x)
        aux_1_out = self.aux_1(x)

        block_2_out = self.block_2(torch.add(aux_1_out, block_1_out))
        return block_2_out


class Xception(nn.Module):
    def __init__(self, in_channels=3, num_classes=10) -> None:
        super().__init__()

        self.entry = EntryFlow(in_channels)
        self.middle = MiddleFlow()
        self.exit = ExitFlow()
        self.pooler = nn.AdaptiveAvgPool2d((1, 1))
        self.classifier = nn.Sequential(
            nn.Flatten(), nn.Linear(in_features=2048, out_features=num_classes)
        )

    def forward(self, x):
        entry = self.entry(x)
        print("entry", entry.shape)
        middle = self.middle(entry)
        print("middle", middle.shape)
        exit = self.exit(middle)
        print("exit", exit.shape)
        out = self.classifier(self.pooler(exit))
        print("out", out.shape)
        return out


if __name__ == "__main__":
    net = Xception(num_classes=1000)
    print(net)
    with torch.no_grad():
        net.eval()
        ip = torch.rand((1, 3, 299, 299))
        got = net(ip)
        print(
            "Num trainable params: ",
            sum(p.numel() for p in net.parameters() if p.requires_grad),
        )

