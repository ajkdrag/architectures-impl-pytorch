from pathlib import Path
from typing import List

import structlog
import torch
from PIL import Image
from yolov1.config import (
    DataConfig,
    ModelConfig,
    YOLOConfig,
)
from yolov1.utils.io import get_all_files
from yolov1.utils.general import encode_labels
from torchvision import transforms

log = structlog.get_logger()


class YOLODataset(torch.utils.data.Dataset):
    config: YOLOConfig
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
        self.config = config
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
                transforms.Resize(self.config.model.input_size),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
                ),
            ]
        )

    def get_data(self):
        subdir = (
            self.config.data.train if self.mode == "train" else self.data_config.val
        )
        path_data = Path(self.config.data.root).joinpath(subdir)
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
                S=self.config.model.S,
                C=self.config.model.nc,
                B=self.config.model.B,
            )

        return image, labels


class InferenceDataset(YOLODataset):
    def __init__(
        self,
        config: YOLOConfig,
        transforms=None,
    ):
        super().__init__(config, transforms)

    def __getitem__(self, index):
        image_path = self.image_files[index]
        image = Image.open(image_path).convert("RGB")
        if self.transforms:
            image = self.transforms(image)
        return image

    def get_data(self):
        source = self.config.inference.source
        if Path(source).is_file():
            self.image_files = [Path(source)]
        elif Path(source).is_dir():
            self.image_files = get_all_files(Path(source))
        else:
            raise ValueError(f"{source} is not a file or a dir")
