import torch.nn as nn
from typing import Dict, Type
import timm


class BackboneFactory:
    registry: Dict[str, Type] = {}

    @classmethod
    def register(cls, name: str):
        def inner_wrapper(wrapped_class: Type):
            cls.registry[name] = wrapped_class
            return wrapped_class

        return inner_wrapper

    @classmethod
    def create_backbone(cls, name: str, output_channels: int, **kwargs):
        backbone = cls.registry[name](**kwargs)
        # Add additional convolutional layers to match the desired output channels
        num_channels = backbone.num_features
        if num_channels != output_channels:
            additional_layers = nn.Sequential(
                nn.Conv2d(
                    num_channels,
                    output_channels,
                    kernel_size=3,
                    padding=1,
                ),
                nn.BatchNorm2d(output_channels),
                nn.LeakyReLU(0.1),
            )
            backbone = nn.Sequential(backbone, additional_layers)

        return backbone


@BackboneFactory.register("resnet34")
def resnet34(pretrained: bool = True, **kwargs):
    model = timm.create_model("resnet34", pretrained=pretrained, **kwargs)
    model.num_features = 512
    return model


@BackboneFactory.register("resnet18")
def resnet18(pretrained: bool = True, **kwargs):
    model = timm.create_model("resnet18", pretrained=pretrained, **kwargs)
    model.num_features = 512
    return model
