import torch
from torch import nn

# features cfg. This needs to be followed by flatten() -> linear layers
VGG16_feats = [
    64,
    64,
    "M",
    128,
    128,
    "M",
    256,
    256,
    256,
    "M",
    512,
    512,
    512,
    "M",
    512,
    512,
    512,
    "M",
]


class VGG(nn.Module):
    def __init__(self, in_channels=3, num_classes=1000) -> None:
        super().__init__()
        self.in_channels = in_channels
        self.feature_extractor = self.get_feature_extractor(VGG16_feats)
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(512 * 7 * 7, 4096),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(4096, 4096),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(4096, num_classes),
        )

    def get_feature_extractor(self, arch):
        layers = []
        in_channels = self.in_channels
        for el in arch:
            if type(el) == int:
                out_channels = el
                layers += [
                    nn.Conv2d(in_channels, out_channels, kernel_size=(3, 3), padding=(1, 1), stride=(1, 1)),
                    nn.BatchNorm2d(out_channels),
                    nn.ReLU(),
                ]
                in_channels = out_channels
            elif el == "M":
                layers += [nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2))]

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.feature_extractor(x)
        x = self.classifier(x)
        return x


if __name__ == "__main__":
    net = VGG()
    print(net)
    with torch.no_grad():
        net.eval()
        ip = torch.rand((1, 3, 224, 224))
        got = net(ip)
        print(got.shape)

