from typing import Tuple

import cv2
import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image
from yolov1.utils.general import nxywh2xyxy


def draw_boxes(
    img: Image.Image,
    labels: torch.Tensor,
    color: Tuple[int, int, int] = (255, 0, 0),
    show_class: bool = True,
    show_conf: bool = True,
    display: bool = True,
):
    canvas = np.array(img)
    black, white = (0, 0, 0), (255, 255, 255)
    text_color = white if np.mean(color) < 128 else black
    class_ids = labels[:, 0].to(torch.int32)
    class_names = map(str, class_ids.tolist())
    boxes = nxywh2xyxy(labels[:, 1:], *img.size)

    for box, class_name in zip(boxes, class_names):
        box_label = []
        if show_class:
            box_label.append(class_name)
        if show_conf:
            box_label.append("1.0")

        canvas = draw_box(
            canvas,
            box,
            " ".join(box_label),
            color=color,
            text_color=text_color,
        )

    if display:
        plt.figure(figsize=(10, 10))
        plt.axis("off")
        plt.imshow(canvas)
        return None
    return canvas


def draw_box(
    image: np.ndarray,
    box: torch.Tensor,
    label: str = "",
    color: tuple = (255, 0, 0),
    text_color: tuple = (255, 255, 255),
):
    lw = max(round(sum(image.shape[:2]) / 2 * 0.003), 2)
    p1, p2 = (int(box[0]), int(box[1])), (int(box[2]), int(box[3]))
    cv2.rectangle(image, p1, p2, color, thickness=lw, lineType=cv2.LINE_AA)
    if label:
        tf = max(lw - 1, 1)  # font thickness
        w, h = cv2.getTextSize(label, 0, fontScale=lw / 3, thickness=tf)[0]
        outside = p1[1] - h >= 3
        p2 = p1[0] + w, p1[1] - h - 3 if outside else p1[1] + h + 3
        cv2.rectangle(image, p1, p2, (*color, 1), -1, cv2.LINE_AA)
        cv2.putText(
            image,
            label,
            (p1[0], p1[1] - 2 if outside else p1[1] + h + 2),
            0,
            lw / 3,
            text_color,
            thickness=tf,
            lineType=cv2.LINE_AA,
        )
    return image