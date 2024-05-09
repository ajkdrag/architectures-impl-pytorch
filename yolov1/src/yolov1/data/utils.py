from torch.utils.data import DataLoader
from yolov1.config import YOLOConfig
from yolov1.data.dataset import YOLODataset, InferenceDataset


def get_dls(config: YOLOConfig, mode="train"):
    dataset = YOLODataset(config, mode=mode, apply_aug=True)
    dataloader = DataLoader(
        dataset,
        shuffle=True if mode == "train" else False,
        **config.training.dls_kwargs,
    )
    return dataloader


def get_dls_for_inference(config: YOLOConfig):
    dataset = InferenceDataset(config)
    dataloader = DataLoader(
        dataset,
        shuffle=False,
        **config.inference.dls_kwargs,
    )
    return dataloader
