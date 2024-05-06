import torch.nn as nn
from yolov1.models.backbone_factory import BackboneFactory
from yolov1.config import ModelConfig


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
        detector_hidden_sz = model_config.detector_hidden_sz

        # unpooled backbone
        self.backbone = BackboneFactory.create_backbone(
            model_config.backbone,
            output_channels=output_channels,
            pretrained=model_config.pretrained,
            num_classes=0,
            global_pool="",
        )

        self.detector = nn.Sequential(
            nn.Flatten(),
            nn.Linear(output_channels * S * S, detector_hidden_sz),
            nn.LeakyReLU(0.1),
            nn.Linear(detector_hidden_sz, S * S * (B * 5 + C)),
            nn.Unflatten(1, (S, S, B * 5 + C)),
        )

    def forward(self, x):
        features = self.backbone(x)
        detections = self.detector(features)
        return detections
