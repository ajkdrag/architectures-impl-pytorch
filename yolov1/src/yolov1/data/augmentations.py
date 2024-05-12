import albumentations as A
import cv2
import torch
from albumentations.pytorch import ToTensorV2
from yolov1.config import AugmentationsConfig, YOLOConfig


def create_augmentation_pipeline(config: YOLOConfig):
    config: AugmentationsConfig = config.data.augmentations
    pipeline = []

    if config.horizontal_flip > 0:
        pipeline.append(A.HorizontalFlip(p=config.horizontal_flip))

    if config.color_jitter > 0:
        pipeline.append(
            A.ColorJitter(
                brightness=0.5,
                contrast=0.5,
                saturation=0.6,
                hue=0.1,
                p=config.color_jitter,
            )
        )

    if config.random_crop > 0:
        pipeline.append(
            A.RandomSizedBBoxSafeCrop(
                *config.random_crop_dims,
                p=config.random_crop,
                erosion_rate=0.3,
            )
        )

    if config.shift_scale_rotate > 0:
        pipeline.append(
            A.ShiftScaleRotate(
                shift_limit=[-0.2, 0.2],
                scale_limit=[-0.5, 0.2],
                rotate_limit=[-10, 10],
                border_mode=cv2.BORDER_CONSTANT,
                value=[int(v * 255.0) for v in [0.485, 0.456, 0.406]],
                p=config.shift_scale_rotate,
            )
        )

    if config.gaussian_noise > 0:
        pipeline.append(
            A.GaussNoise(
                var_limit=(10.0, 50.0),
                p=config.gaussian_noise,
            )
        )

    return A.Compose(
        pipeline,
        bbox_params=A.BboxParams(
            format="yolo",
            label_fields=["class_labels"],
            min_visibility=0.3,
        ),
    )


def create_transforms(config: YOLOConfig):
    return A.Compose(
        [
            A.Resize(
                width=config.model.input_size[0],
                height=config.model.input_size[1],
            ),
            A.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
            ),
            ToTensorV2(),
        ],
        bbox_params=A.BboxParams(
            format="yolo",
            label_fields=["class_labels"],
        ),
    )


def apply_pipeline(pipeline, image, bboxes=None, class_labels=None):
    if bboxes is None:
        return pipeline(image=image, class_labels=None)["image"]

    transformed = pipeline(
        image=image,
        bboxes=bboxes,
        class_labels=class_labels,
    )
    return (
        transformed["image"],
        torch.tensor(transformed["bboxes"]),
        torch.tensor(transformed["class_labels"]),
    )
