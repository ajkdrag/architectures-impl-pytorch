import torch.nn as nn
from yolov1.config import ModelConfig
from yolov1.models.backbone_factory import BackboneFactory
from yolov1.models.blocks import BasicConvBlock, CustomConvBlock
from yolov1.utils.constants import Activations


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
        self.conv_block_end = BasicConvBlock(
            conv_channels[0], B * 5 + C, activation=Activations.SIGMOID
        )

    def forward(self, x):
        x = self.backbone(x)
        x = self.conv_block_1(x)
        x = self.avg_pool(x)
        x = self.conv_block_end(x)
        return x.permute((0, 2, 3, 1))
