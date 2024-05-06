from torch.utils.data import DataLoader
from yolov1.config import YOLOConfig
from yolov1.data.dataset import YOLODataset, InferenceDataset


def get_dls(config: YOLOConfig, mode="train"):
    dataset = YOLODataset(config, mode=mode)
    dataloader = DataLoader(
        dataset,
        batch_size=config.training.batch_size,
        num_workers=config.training.num_workers,
        shuffle=True if mode == "train" else False,
    )
    return dataloader


def get_dls_for_inference(config: YOLOConfig):
    dataset = InferenceDataset(config)
    dataloader = DataLoader(
        dataset,
        batch_size=config.inference.batch_size,
        shuffle=False,
    )
    return dataloader
