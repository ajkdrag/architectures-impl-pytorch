from typing import Optional

import torch
import torch.nn as nn
from yolov1.utils.constants import Activations


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


class BasicConvBlock(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        activation: Optional[Activations] = Activations.NONE,
    ):
        super().__init__()
        self.conv1 = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=3,
            padding=1,
            bias=False,
        )
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.activation = activation.value

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        return self.activation(x)


class CustomConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv_block_1 = BasicConvBlock(
            in_channels,
            out_channels,
            activation=Activations.RELU,
        )

        self.conv_block_2 = BasicConvBlock(
            out_channels,
            out_channels,
            activation=Activations.RELU,
        )

    def forward(self, x):
        x = self.conv_block_1(x)
        return self.conv_block_2(x)
