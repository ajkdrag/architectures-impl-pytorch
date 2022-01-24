import torch
from torch import nn


class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, **kwargs) -> None:
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, **kwargs)
        self.relu = nn.ReLU()
        self.batchnorm = nn.BatchNorm2d(out_channels)

    def forward(self, x):
        return self.relu(self.batchnorm(self.conv(x)))


class InceptionBlock(nn.Module):
    def __init__(
        self, in_channels, out_1x1, red_3x3, out_3x3, red_5x5, out_5x5, out_1x1_pool
    ) -> None:
        super().__init__()

        self.tower_1 = ConvBlock(in_channels, out_1x1, kernel_size=1)
        self.tower_2 = nn.Sequential(
            ConvBlock(in_channels, red_3x3, kernel_size=1),
            ConvBlock(red_3x3, out_3x3, kernel_size=3, stride=1, padding=1),
        )
        self.tower_3 = nn.Sequential(
            ConvBlock(in_channels, red_5x5, kernel_size=1),
            ConvBlock(red_5x5, out_5x5, kernel_size=5, stride=1, padding=2),
        )
        self.tower_4 = nn.Sequential(
            nn.MaxPool2d(kernel_size=3, stride=1, padding=1),
            ConvBlock(in_channels, out_1x1_pool, kernel_size=1, stride=1),
        )

    def forward(self, x):
        return torch.cat(
            [self.tower_1(x), self.tower_2(x), self.tower_3(x), self.tower_4(x)], dim=1
        )


class AuxClassifier(nn.Module):
    def __init__(self, num_features, num_classes):
        super().__init__()
        self.avgpool = nn.AvgPool2d(kernel_size=5, stride=3)
        self.conv = nn.Conv2d(
            in_channels=num_features, out_channels=128, kernel_size=1, stride=1,
        )
        self.relu = nn.ReLU()
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(4 * 4 * 128, 1024),
            nn.Dropout(p=0.7),
            nn.Linear(1024, num_classes),
        )

    def forward(self, x):
        x = self.avgpool(x)
        x = self.conv(x)
        x = self.relu(x)
        x = self.classifier(x)

        return x


class GoogleNet(nn.Module):
    def __init__(self, in_channels=3, num_classes=10) -> None:
        super().__init__()
        self.conv_1 = ConvBlock(in_channels, 64, kernel_size=7, stride=2, padding=3)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.conv_2 = nn.Sequential(
            ConvBlock(64, 64, kernel_size=1, stride=1),
            ConvBlock(64, 192, kernel_size=3, stride=1, padding=1),
        )
        self.inception_3a = InceptionBlock(192, 64, 96, 128, 16, 32, 32)
        self.inception_3b = InceptionBlock(256, 128, 128, 192, 32, 96, 64)

        self.inception_4a = InceptionBlock(480, 192, 96, 208, 16, 48, 64)
        self.inception_4b = InceptionBlock(512, 160, 112, 224, 24, 64, 64)
        self.inception_4c = InceptionBlock(512, 128, 128, 256, 24, 64, 64)
        self.inception_4d = InceptionBlock(512, 112, 144, 288, 32, 64, 64)
        self.inception_4e = InceptionBlock(528, 256, 160, 320, 32, 128, 128)

        self.inception_5a = InceptionBlock(832, 256, 160, 320, 32, 128, 128)
        self.inception_5b = InceptionBlock(832, 384, 192, 384, 48, 128, 128)

        self.avgpool = nn.AvgPool2d(kernel_size=7, stride=1)

        self.aux_classifier_1 = AuxClassifier(512, num_classes)
        self.aux_classifier_2 = AuxClassifier(528, num_classes)

        self.classifier = nn.Sequential(
            nn.Flatten(), nn.Dropout(p=0.4), nn.Linear(1024, num_classes)
        )

    def forward(self, x):
        x = self.conv_1(x)
        x = self.maxpool(x)
        x = self.conv_2(x)
        x = self.maxpool(x)

        x = self.inception_3a(x)
        x = self.inception_3b(x)
        x = self.maxpool(x)

        x = self.inception_4a(x)
        out_1 = self.aux_classifier_1(x)

        x = self.inception_4b(x)
        x = self.inception_4c(x)
        x = self.inception_4d(x)
        out_2 = self.aux_classifier_2(x)

        x = self.inception_4e(x)
        x = self.maxpool(x)

        x = self.inception_5a(x)
        x = self.inception_5b(x)
        x = self.avgpool(x)
        x = self.classifier(x)

        if self.training:
            return [x, out_1, out_2]
        return x


if __name__ == "__main__":
    batch_size = 3
    num_classes = 10
    x = torch.randn((batch_size, 3, 224, 224))
    y = torch.randint(0, num_classes, (batch_size,))

    net = GoogleNet(num_classes=num_classes)
    out, out_1, out_2 = net(x)

    loss = nn.CrossEntropyLoss()
    loss_1 = nn.CrossEntropyLoss()
    loss_2 = nn.CrossEntropyLoss()

    discount = 0.3

    total_loss = loss(out, y) + discount * (loss_1(out_1, y) + loss_2(out_2, y))
    print(total_loss.item())
