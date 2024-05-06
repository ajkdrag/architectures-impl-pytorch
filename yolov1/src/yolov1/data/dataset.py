from pathlib import Path
from typing import List

import numpy as np
import structlog
import torch
from PIL import Image
from yolov1.config import (
    YOLOConfig,
)
from yolov1.data.augmentations import (
    apply_pipeline,
    create_augmentation_pipeline,
    create_transforms,
)
from yolov1.utils.general import encode_labels
from yolov1.utils.io import get_all_files
from torchvision.transforms.functional import pil_to_tensor

log = structlog.get_logger()


class YOLODataset(torch.utils.data.Dataset):
    config: YOLOConfig
    transforms: any
    augmentations: any
    mode: str
    image_files: List[Path]
    label_files: List[Path]
    encode: bool

    def __init__(
        self,
        config: YOLOConfig,
        mode="train",
        encode=True,
    ):
        self.config = config
        self.mode = mode
        self.encode = encode
        self.transforms = create_transforms(config)
        self.augmentations = (
            create_augmentation_pipeline(config) if mode == "train" else None
        )
        self.get_data()

    def get_data(self):
        subdir = (
            self.config.data.train if self.mode == "train" else self.config.data.val
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

        log.info("Loaded {} samples".format(len(self.image_files)))

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, index):
        image_path = self.image_files[index]
        label_path = self.label_files[index]

        image = np.array(Image.open(image_path).convert("RGB"), dtype=np.float32)
        with open(label_path, "r") as f:
            labels = f.read().strip().split("\n")
            labels = [list(map(float, label.split())) for label in labels]
            labels = torch.tensor(labels)

        bboxes, class_labels = labels[:, 1:], labels[:, 0]

        if self.augmentations:
            image, bboxes, class_labels = apply_pipeline(
                self.augmentations,
                image,
                bboxes,
                class_labels,
            )

        image, bboxes, class_labels = apply_pipeline(
            self.transforms,
            image,
            bboxes,
            class_labels,
        )

        transformed_labels = torch.cat([class_labels, bboxes], dim=1)

        if self.encode:
            transformed_labels = encode_labels(
                transformed_labels,
                S=self.config.model.S,
                C=self.config.model.nc,
                B=self.config.model.B,
            )

        return image, transformed_labels


class InferenceDataset(YOLODataset):
    def __init__(
        self,
        config: YOLOConfig,
    ):
        super().__init__(config, mode="test")

    def __getitem__(self, index):
        image_path = self.image_files[index]
        image = np.array(Image.open(image_path).convert("RGB"), dtype=np.float32)
        return apply_pipeline(
            self.transforms,
            image,
            bboxes=None,
            class_labels=None,
        )

    def get_data(self):
        source = self.config.inference.source
        if Path(source).is_file():
            self.image_files = [Path(source)]
        elif Path(source).is_dir():
            self.image_files = get_all_files(Path(source))
        else:
            raise ValueError(f"{source} is not a file or a dir")
