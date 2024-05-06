import torch
import albumentations as A
from albumentations.pytorch import ToTensorV2
from yolov1.config import AugmentationsConfig, YOLOConfig


def create_augmentation_pipeline(config: YOLOConfig):
    config: AugmentationsConfig = config.data.augmentations
    pipeline = []

    if config.horizontal_flip > 0:
        pipeline.append(A.HorizontalFlip(p=config.horizontal_flip))

    if config.rotate > 0:
        pipeline.append(A.RandomRotate90(p=config.rotate))

    if config.brightness_contrast > 0:
        pipeline.append(A.RandomBrightnessContrast(
            p=config.brightness_contrast))

    if config.shift_scale_rotate > 0:
        pipeline.append(A.ShiftScaleRotate(p=config.shift_scale_rotate))

    if config.random_crop > 0:
        pipeline.append(
            A.RandomResizedCrop(
                p=config.random_crop,
                size=config.random_crop_dims,
            )
        )

    return A.Compose(
        pipeline,
        bbox_params=A.BboxParams(
            format="yolo",
            label_fields=["class_labels"],
        ),
    )


def create_transforms(config: YOLOConfig):
    return A.Compose(
        [
            A.Resize(
                width=config.model.input_size[0],
                height=config.model.input_size[1],
            ),
            # A.Normalize(
            #     mean=[0.485, 0.456, 0.406],
            #     std=[0.229, 0.224, 0.225],
            # ),
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
        torch.tensor(transformed["class_labels"]).unsqueeze(1),
    )
