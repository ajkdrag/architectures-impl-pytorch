from pathlib import Path
from typing import List

import structlog
import torch
from PIL import Image
from yolov1.config import DataConfig, ModelConfig, YOLOConfig
from yolov1.utils.io import get_all_files
from yolov1.utils.general import encode_labels
from torchvision import transforms

log = structlog.get_logger()


class YOLODataset(torch.utils.data.Dataset):
    data_config: DataConfig
    model_config: ModelConfig
    transforms: any
    mode: str
    image_files: List[Path]
    label_files: List[Path]
    encode: bool

    def __init__(
        self,
        config: YOLOConfig,
        transforms=None,
        mode="train",
        encode=True,
    ):
        self.data_config = config.data
        self.model_config = config.model
        self.transforms = transforms
        self.mode = mode
        self.encode = encode
        self.get_data()
        log.info("Loaded {} samples".format(len(self)))

        if transforms is None:
            self.use_default_transforms()

    def use_default_transforms(self):
        self.transforms = transforms.Compose(
            [
                transforms.Resize(self.model_config.input_size),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
                ),
            ]
        )

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

        if self.encode:
            labels = encode_labels(
                labels,
                S=self.model_config.S,
                C=len(self.data_config.names),
                B=self.model_config.B,
            )

        return image, labels
