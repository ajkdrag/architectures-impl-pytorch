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
