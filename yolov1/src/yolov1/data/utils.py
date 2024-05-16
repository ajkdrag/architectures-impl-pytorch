import random

import numpy as np
import torch
from torch.utils.data import DataLoader, WeightedRandomSampler
from yolov1.config import YOLOConfig
from yolov1.data.dataset import InferenceDataset, YOLODataset


def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def get_dls(config: YOLOConfig, mode="train", seed=0):
    dataset = YOLODataset(config, mode=mode)
    sampler = WeightedRandomSampler(
        dataset.sample_weights, len(dataset), replacement=True
    )

    dataloader = DataLoader(
        dataset,
        # shuffle=True if mode == "train" else False,
        sampler=sampler if mode == "train" else None,
        worker_init_fn=seed_worker,
        generator=torch.Generator().manual_seed(seed),
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
