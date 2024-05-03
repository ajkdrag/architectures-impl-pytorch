from pathlib import Path
from typing import List

import structlog
import torch
from PIL import Image
from yolov1.config import DataConfig, YOLOConfig
from yolov1.data.utils import get_all_files

log = structlog.get_logger()


class YOLODataset(torch.utils.data.Dataset):
    data_config: DataConfig
    transforms: any
    mode: str
    image_files: List[Path]
    label_files: List[Path]

    def __init__(self, config: YOLOConfig, transforms=None, mode="train"):
        self.data_config = config.data
        self.transforms = transforms
        self.mode = mode
        self.get_data()
        log.info("Loaded {} samples".format(len(self)))

    def get_data(self):
        subdir = (
            self.data_config.train if self.mode == "train" else self.data_config.val
        )
        path_data = Path(self.data_config.root).joinpath(subdir)
        self.image_files = sorted(get_all_files(path_data.joinpath("images")))
        self.label_files = sorted(
            get_all_files(
                path_data.joinpath("labels"),
                exts=[".txt"],
            )
        )
        if len(self.image_files) != len(self.label_files):
            log.error(
                f"#Images {len(self.image_files)} != #Labels {len(self.label_files)}"
            )
            raise ValueError("Number of images and labels not equal")

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, index):
        image_path = self.image_files[index]
        label_path = self.label_files[index]

        image = Image.open(image_path).convert("RGB")
        with open(label_path, "r") as f:
            labels = f.read().strip().split("\n")
            labels = [list(map(float, label.split())) for label in labels]
            labels = torch.tensor(labels)

        if self.transforms:
            image = self.transforms(image)

        return image, labels
