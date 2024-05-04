import torch


def nxywh2xyxy(tensor_: torch.Tensor, width: int, height: int):
    """Convert normalized xywh tensor to xyxy tensor."""
    tensor_[:, 0] *= width
    tensor_[:, 1] *= height
    tensor_[:, 2] *= width
    tensor_[:, 3] *= height

    tensor_[:, 0] -= tensor_[:, 2] * 0.5
    tensor_[:, 1] -= tensor_[:, 3] * 0.5
    tensor_[:, 2] += tensor_[:, 0]
    tensor_[:, 3] += tensor_[:, 1]

    return tensor_


def encode_labels(labels, S, B, C):
    """SxS grid according to yolo algorithm"""
    encoded_labels = torch.zeros((S, S, B * 5 + C))
    for label in labels:
        class_label, x, y, w, h = label
        class_label = int(class_label)

        i, j = int(S * y), int(S * x)
        x_cell, y_cell = S * x - j, S * y - i

        encoded_labels[i, j, 0:4] = torch.tensor([x_cell, y_cell, w, h])
        # Object conf set to 1 for simplificty (replace with IoU)
        encoded_labels[i, j, 4] = 1

        # Encode the class probabilities (OHE)
        encoded_labels[i, j, 5 + class_label] = 1

    return encoded_labels
