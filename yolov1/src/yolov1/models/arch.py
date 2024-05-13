import math

import torch
import torch.nn as nn
from yolov1.config import ModelConfig
from yolov1.models.backbone_factory import BackboneFactory


class YOLOOutputLayer(nn.Module):
    def __init__(self):
        super().__init__()
        self.softmax = nn.Softmax(dim=-1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # x.shape = [batch, S, S, B * 5 + C]
        classes = self.softmax(x[..., 5:])
        coordinates = self.sigmoid(x[..., :5])
        return torch.cat([coordinates, classes], dim=-1)


class OutputNet(nn.Module):
    # no expansion
    # dilation = 2
    # type B use 1x1 conv
    expansion = 1

    def __init__(self, in_planes, planes, stride=1, block_type="A"):
        super().__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(
            planes,
            planes,
            kernel_size=3,
            stride=stride,
            padding=2,
            bias=False,
            dilation=2,
        )
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(
            planes, self.expansion * planes, kernel_size=1, bias=False
        )
        self.bn3 = nn.BatchNorm2d(self.expansion * planes)
        self.downsample = nn.Sequential()
        self.relu = nn.ReLU(inplace=True)
        if stride != 1 or in_planes != self.expansion * planes or block_type == "B":
            self.downsample = nn.Sequential(
                nn.Conv2d(
                    in_planes,
                    self.expansion * planes,
                    kernel_size=1,
                    stride=stride,
                    bias=False,
                ),
                nn.BatchNorm2d(self.expansion * planes),
            )

    def forward(self, x):
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        out += self.downsample(x)
        out = self.relu(out)
        return out


class YOLOv2(nn.Module):
    model_config: ModelConfig

    def __init__(
        self,
        model_config: ModelConfig,
    ):
        super().__init__()
        self.model_config = model_config
        # S = model_config.S
        B = model_config.B
        C = model_config.nc
        # detector_hidden_sizes = model_config.detector_hidden_sizes

        # unpooled backbone
        self.backbone = BackboneFactory.create_backbone(
            model_config.backbone,
            pretrained=model_config.pretrained,
            num_classes=0,
            global_pool="",
        )

        if model_config.freeze_backbone:
            for param in self.backbone.parameters():
                param.requires_grad = False

        self.custom_conv_v1 = self._make_output_layer(
            in_channels=self.backbone.num_features
        )
        self.avgpool = nn.AvgPool2d(2)  # kernel_size = 2  , stride = 2
        self.conv_end = nn.Conv2d(
            256, (B * 5 + C), kernel_size=3, stride=1, padding=1, bias=False
        )
        self.bn_end = nn.BatchNorm2d(B * 5 + C)

    def _make_output_layer(self, in_channels):
        layers = []
        layers.append(OutputNet(in_planes=in_channels, planes=256, block_type="B"))
        layers.append(OutputNet(in_planes=256, planes=256, block_type="A"))
        layers.append(OutputNet(in_planes=256, planes=256, block_type="A"))
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.backbone(x)
        x = self.custom_conv_v1(x)
        x = self.avgpool(x)
        x = self.conv_end(x)
        x = self.bn_end(x)
        x = torch.sigmoid(x)
        x = x.permute(0, 2, 3, 1)
        return x


class YOLOv1(nn.Module):
    model_config: ModelConfig

    def __init__(
        self,
        model_config: ModelConfig,
    ):
        super().__init__()
        self.model_config = model_config

        S = model_config.S
        B = model_config.B
        C = model_config.nc
        detector_hidden_sizes = model_config.detector_hidden_sizes

        # unpooled backbone
        self.backbone = BackboneFactory.create_backbone(
            model_config.backbone,
            pretrained=model_config.pretrained,
            num_classes=0,
            global_pool="",
        )

        if model_config.freeze_backbone:
            for param in self.backbone.parameters():
                param.requires_grad = False

        backbone_out_scale = math.ceil(
            model_config.input_size[0] / self.backbone.scale_down_factor
        )
        backbone_out_units = self.backbone.num_features * (backbone_out_scale) ** 2

        # detector (head)
        linear_blocks_input = [backbone_out_units, *detector_hidden_sizes]
        linear_blocks_output = [*detector_hidden_sizes, S * S * (B * 5 + C)]

        detector = nn.Sequential()
        for ip, op in zip(linear_blocks_input, linear_blocks_output):
            detector.add_module(
                f"linear_{ip}_{op}",
                self._make_linear_block((ip, op)),
            )

        self.backbone.fc = nn.Sequential(
            nn.Flatten(),
            detector,
            nn.Unflatten(1, (S, S, B * 5 + C)),
            # YOLOOutputLayer(),
        )

    def _make_linear_block(self, linear_units: tuple[int, ...]) -> nn.Module:
        return nn.Sequential(
            nn.Linear(linear_units[0], linear_units[1], bias=False),
            nn.BatchNorm1d(linear_units[1]),
            nn.LeakyReLU(0.1),
            nn.Dropout(0.5),
        )

    def forward(self, x):
        return self.backbone(x)
