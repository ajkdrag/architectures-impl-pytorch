from typing import Optional, List, Tuple

import yaml
from pydantic import BaseModel


class LossConfig(BaseModel):
    l_coord: Optional[float] = 5.0
    l_obj: Optional[float] = 1.0
    l_noobj: Optional[float] = 0.5
    l_class: Optional[float] = 1.0


class AugmentationsConfig(BaseModel):
    horizontal_flip: Optional[float] = 0.5
    vertical_flip: Optional[float] = 0.0
    brightness_contrast: Optional[float] = 0.8
    shift_scale_rotate: Optional[float] = 0.0
    random_crop: Optional[float] = 0.6
    random_crop_dims: Tuple[float, float]


class DataConfig(BaseModel):
    root: str
    train: str
    val: str
    names: List[str]
    augmentations: AugmentationsConfig


class TrainingConfig(BaseModel):
    epochs: int
    dls_kwargs: Optional[dict] = {}
    optim_kwargs: Optional[dict] = {}
    checkpoints_dir: str
    save_freq: int
    loss: Optional[LossConfig] = LossConfig()


class InferenceConfig(BaseModel):
    checkpoint: str
    source: str
    dls_kwargs: Optional[dict] = dict()
    conf_th: float


class ModelConfig(BaseModel):
    backbone: str
    pretrained: bool
    freeze_backbone: bool
    backbone_output_channels: int
    detector_hidden_sizes: Tuple[int, ...]
    input_size: tuple
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
