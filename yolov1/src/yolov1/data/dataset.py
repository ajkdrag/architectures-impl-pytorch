from pathlib import Path
from typing import List

import cv2
import structlog
import torch
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

log = structlog.get_logger()


class YOLODataset(torch.utils.data.Dataset):
    config: YOLOConfig
    transforms: any
    augmentations: any
    mode: str
    image_files: List[Path]
    label_files: List[Path]
    class_weights: List[float]
    sample_weights: List[float] = None
    encode: bool
    nc: int

    def __init__(
        self,
        config: YOLOConfig,
        mode="train",
        encode=True,
    ):
        self.config = config
        self.mode = mode
        self.encode = encode
        self.class_weights = config.data.class_weights
        self.nc = len(config.data.names)
        self.transforms = create_transforms(config)
        self.augmentations = (
            create_augmentation_pipeline(config) if mode == "train" else None
        )
        self.get_data()
        self._set_class_weights()
        self._set_sample_weights()

    def _set_class_weights(self):
        if self.class_weights is None:
            # +1 for the background class
            self.class_weights = [1 / (self.nc + 1)] * self.nc
        log.info("Class weights: {}".format(self.class_weights))

    def _set_sample_weights(self):
        # for weighted random sampling
        background_cls_weight = 1 / (self.nc + 1)
        if self.sample_weights is None:
            self.sample_weights = [background_cls_weight] * len(self.image_files)
            for idx in range(len(self.image_files)):
                _, class_labels = self._read_label(idx)
                if len(class_labels) == 0:
                    continue
                majority_class = torch.mode(class_labels, 0).values.item()
                class_weight = self.class_weights[int(majority_class)]
                self.sample_weights[idx] = class_weight

    def get_data(self):
        subdir = (
            self.config.data.train if self.mode == "train" else self.config.data.val
        )
        path_data = Path(self.config.data.root).joinpath(subdir)
        self.image_files = sorted(get_all_files(path_data.joinpath("images")))
        self.label_files = [
            path_data.joinpath("labels", f.name).with_suffix(".txt")
            for f in self.image_files
        ]

        if len(self.image_files) != len(self.label_files):
            log.error(
                f"#Images {len(self.image_files)} != #Labels {len(self.label_files)}"
            )
            raise ValueError("Number of images and labels not equal")

        log.info("Loaded {} samples".format(len(self.image_files)))

    def _read_label(self, index):
        label_path = self.label_files[index]
        with open(label_path, "r") as f:
            labels = f.read().strip().split("\n")
            labels = [list(map(float, label.split())) for label in labels]
            labels = torch.tensor(labels, dtype=torch.float32)
        boxes, class_labels = labels[:, 1:], labels[:, 0]
        return boxes, class_labels

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, index):
        image_path = self.image_files[index]
        image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        boxes, class_labels = self._read_label(index)

        if self.config.data.augmentations.apply and self.augmentations:
            image, boxes, class_labels = apply_pipeline(
                self.augmentations,
                image,
                boxes,
                class_labels,
            )

        transformed_labels = torch.cat([class_labels.unsqueeze(1), boxes], dim=1)

        if self.encode:
            image, boxes, class_labels = apply_pipeline(
                self.transforms,
                image,
                boxes,
                class_labels,
            )

            transformed_labels = encode_labels(
                torch.cat([class_labels.unsqueeze(1), boxes], dim=1),
                S=self.config.model.S,
                B=self.config.model.B,
                C=self.config.model.nc,
            )

        return image, transformed_labels


class InferenceDataset(YOLODataset):
    def __init__(
        self,
        config: YOLOConfig,
    ):
        super().__init__(config, mode="test")

    def _set_sample_weights(self):
        pass

    def __getitem__(self, index):
        image_path = self.image_files[index]
        image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

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
