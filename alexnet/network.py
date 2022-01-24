from torch import nn
import torch


class AlexNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.feature_ext = nn.Sequential(
            nn.Conv2d(3, 96, kernel_size=(11, 11), stride=4, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=(3, 3), stride=(2, 2)),
            nn.Conv2d(96, 256, kernel_size=(5, 5), stride=(1, 1), padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=(3, 3), stride=(2, 2)),
            nn.Conv2d(256, 384, kernel_size=(3, 3), stride=(1, 1)),
            nn.ReLU(inplace=True),
            nn.Conv2d(384, 384, kernel_size=(3, 3), stride=(1, 1)),
            nn.ReLU(inplace=True),
            nn.Conv2d(384, 256, kernel_size=(3, 3), stride=(1, 1)),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=(3, 3), stride=(2, 2)),
        )

        self.pooling = nn.AdaptiveAvgPool2d((6, 6))  # important
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(256 * 6 * 6, 4096),
            nn.ReLU(inplace=True),
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),
            nn.Linear(4096, 10),
            nn.Dropout(0.5),
        )

    def forward(self, x):
        features = self.feature_ext(x)
        pooled = self.pooling(features)
        dense_out = self.classifier(pooled)
        return dense_out


if __name__ == "__main__":
    net = AlexNet()
    print(net)
    with torch.no_grad():
        net.eval()
        ip = torch.rand((1, 3, 227, 227))
        got = net(ip)
        print(got.shape)
