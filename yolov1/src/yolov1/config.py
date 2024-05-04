from typing import List

import yaml
from pydantic import BaseModel


class DataConfig(BaseModel):
    root: str
    train: str
    val: str
    names: List[str]


class TrainingConfig(BaseModel):
    batch_size: int
    num_workers: int
    learning_rate: float
    epochs: int


class ModelConfig(BaseModel):
    backbone: str
    pretrained: bool
    backbone_output_channels: int
    detector_hidden_sz: int
    input_size: tuple
    S: int
    B: int
    nc: int


class YOLOConfig(BaseModel):
    data: DataConfig
    training: TrainingConfig
    model: ModelConfig


def parse_config(config_file):
    with open(config_file, "r") as f:
        config_dict = yaml.safe_load(f)

    cfg = YOLOConfig(**config_dict)
    return cfg
