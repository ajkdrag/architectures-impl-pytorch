import yaml
from pydantic import BaseModel


class DataConfig(BaseModel):
    train_data: str
    val_data: str


class TrainingConfig(BaseModel):
    batch_size: int
    num_workers: int
    learning_rate: float
    epochs: int


class ModelConfig(BaseModel):
    backbone: str
    pretrained: bool


class YOLOConfig(BaseModel):
    data: DataConfig
    training: TrainingConfig
    model: ModelConfig


def parse_config(config_file):
    with open(config_file, "r") as f:
        config_dict = yaml.safe_load(f)

    cfg = YOLOConfig(**config_dict)
    return cfg
