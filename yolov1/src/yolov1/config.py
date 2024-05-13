from typing import List, Optional, Tuple

import yaml
from pydantic import BaseModel


class LossConfig(BaseModel):
    l_coord: Optional[float] = 5.0
    l_obj: Optional[float] = 1.0
    l_noobj: Optional[float] = 0.5
    l_class: Optional[float] = 1.0


class AugmentationsConfig(BaseModel):
    apply: Optional[bool] = True
    horizontal_flip: Optional[float] = 0.5
    color_jitter: Optional[float] = 0.5
    shift_scale_rotate: Optional[float] = 0.0
    random_crop: Optional[float] = 0.6
    random_crop_dims: Tuple[float, float]
    gaussian_noise: Optional[float] = 0.5


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
    val_freq: int
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
    detector_hidden_sizes: Tuple[int, ...]
    input_size: Tuple[int, int]
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
