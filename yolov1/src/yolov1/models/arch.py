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
        backbone_out_units = self.backbone.num_features * \
            (backbone_out_scale) ** 2

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
