import torch.nn as nn
from yolov1.config import ModelConfig
from yolov1.models.backbone_factory import BackboneFactory
from yolov1.models.blocks import CustomConvBlock


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
        conv_channels = model_config.conv_block_channels
        detector_hidden_sz = model_config.detector_hidden_sz

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

        self.conv_block_1 = CustomConvBlock(
            in_channels=self.backbone.num_features,
            out_channels=conv_channels[0],
        )
        self.avg_pool = nn.AdaptiveAvgPool2d((S, S))
        self.head = nn.Sequential(
            nn.Flatten(),
            nn.Linear(
                conv_channels[-1] * S * S,
                detector_hidden_sz,
                bias=False,
            ),
            nn.BatchNorm1d(detector_hidden_sz),
            nn.Dropout(0.5),
            nn.LeakyReLU(0.1),
            nn.Linear(detector_hidden_sz, S * S * (B * 5 + C)),
            nn.Sigmoid(),
            nn.Unflatten(-1, (S, S, (B * 5 + C))),
        )

    def forward(self, x):
        x = self.backbone(x)
        x = self.conv_block_1(x)
        x = self.avg_pool(x)
        return self.head(x)
