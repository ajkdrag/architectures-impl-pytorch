from typing import List, Tuple

import yaml
from pydantic import BaseModel


class AugmentationsConfig(BaseModel):
    horizontal_flip: float
    vertical_flip: float
    rotate: float
    brightness_contrast: float
    shift_scale_rotate: float
    random_crop: float
    random_crop_dims: Tuple[float, float]


class DataConfig(BaseModel):
    root: str
    train: str
    val: str
    names: List[str]
    augmentations: AugmentationsConfig


class TrainingConfig(BaseModel):
    batch_size: int
    num_workers: int
    learning_rate: float
    epochs: int
    checkpoints_dir: str
    save_freq: int


class InferenceConfig(BaseModel):
    batch_size: int
    checkpoint: str
    source: str


class ModelConfig(BaseModel):
    backbone: str
    pretrained: bool
    backbone_output_channels: int
    detector_hidden_sz: int
    input_size: tuple
    conf_th: float
    S: int
    B: int
    nc: int


class YOLOConfig(BaseModel):
    data: DataConfig
    training: TrainingConfig
    inference: InferenceConfig
    model: ModelConfig


def parse_config(config_file):
    with open(config_file, "r") as f:
        config_dict = yaml.safe_load(f)

    cfg = YOLOConfig(**config_dict)
    return cfg
