import torch


def ncxcywh2xyxy(tensor: torch.Tensor, width: int, height: int):
    """Convert normalized cxcywh tensor to xyxy tensor."""
    tensor_ = tensor.clone()
    tensor_[:, 0] *= width
    tensor_[:, 1] *= height
    tensor_[:, 2] *= width
    tensor_[:, 3] *= height

    tensor_[:, 0] -= tensor_[:, 2] / 2
    tensor_[:, 1] -= tensor_[:, 3] / 2
    tensor_[:, 2] += tensor_[:, 0]
    tensor_[:, 3] += tensor_[:, 1]

    return tensor_


def encode_labels(labels, S, B, C):
    """SxS grid according to yolo algorithm"""
    encoded_labels = torch.zeros((S, S, B * 5 + C))
    for label in labels:
        class_label, cx, cy, w, h = label
        x, y = cx - w * 0.5, cy - h * 0.5
        class_label = int(class_label)

        i, j = int(S * y), int(S * x)
        x_cell, y_cell = S * x - j, S * y - i

        encoded_labels[i, j, 0:4] = torch.tensor([x_cell, y_cell, w, h])
        # Object conf set to 1 for simplificty (replace with IoU)
        encoded_labels[i, j, 4] = 1

        # Encode the class probabilities (OHE)
        encoded_labels[i, j, 5 + class_label] = 1

    return encoded_labels


def decode_labels(encoded_labels, S, B, C):
    boxes = []

    for i in range(S):
        for j in range(S):
            cell_output = encoded_labels[i, j]
            bbox = cell_output[:4]
            conf = cell_output[4]
            class_probs = cell_output[5:]

            x_cell, y_cell, w_cell, h_cell = bbox
            x = (j + x_cell) / S
            y = (i + y_cell) / S
            w = w_cell
            h = h_cell

            class_idx = torch.argmax(class_probs)
            class_prob = class_probs[class_idx]

            box = [
                class_idx.item(),
                x,
                y,
                w,
                h,
                conf,
                class_prob.item(),
            ]
            boxes.append(box)

    return boxes
