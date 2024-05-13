from enum import Enum

import torch.nn as nn


class Activations(Enum):
    RELU = nn.ReLU(inplace=True)
    TANH = nn.Tanh()
    SIGMOID = nn.Sigmoid()
    NONE = nn.Identity()
