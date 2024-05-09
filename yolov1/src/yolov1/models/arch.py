import torch
import torch.nn as nn
from yolov1.models.backbone_factory import BackboneFactory
from yolov1.config import ModelConfig


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
        output_channels = model_config.backbone_output_channels
        detector_hidden_sizes = model_config.detector_hidden_sizes

        # unpooled backbone
        self.backbone = BackboneFactory.create_backbone(
            model_config.backbone,
            pretrained=model_config.pretrained,
            num_classes=0,
        )

        if model_config.freeze_backbone:
            for param in self.backbone.parameters():
                param.requires_grad = False

        self.detector = nn.Sequential(
            nn.Linear(output_channels, detector_hidden_sizes[0]),
            nn.LeakyReLU(0.1),
            nn.Dropout(0.2),
            nn.Linear(detector_hidden_sizes[0], detector_hidden_sizes[1]),
            nn.LeakyReLU(0.1),
            nn.Dropout(0.2),
            nn.BatchNorm1d(detector_hidden_sizes[1]),
            nn.Linear(detector_hidden_sizes[1], S * S * (B * 5 + C)),
            nn.Unflatten(1, (S, S, B * 5 + C)),
            YOLOOutputLayer(),
        )

    def forward(self, x):
        features = self.backbone(x)
        detections = self.detector(features)
        return detections
