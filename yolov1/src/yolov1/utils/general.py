import torch
import torch.nn as nn
import torchvision.ops as ops


def calc_iou(pred_boxes, target_boxes):
    pred_x1 = pred_boxes[..., 0] - pred_boxes[..., 2] / 2
    pred_y1 = pred_boxes[..., 1] - pred_boxes[..., 3] / 2
    pred_x2 = pred_boxes[..., 0] + pred_boxes[..., 2] / 2
    pred_y2 = pred_boxes[..., 1] + pred_boxes[..., 3] / 2

    target_x1 = target_boxes[..., 0] - target_boxes[..., 2] / 2
    target_y1 = target_boxes[..., 1] - target_boxes[..., 3] / 2
    target_x2 = target_boxes[..., 0] + target_boxes[..., 2] / 2
    target_y2 = target_boxes[..., 1] + target_boxes[..., 3] / 2

    inter_x1 = torch.max(pred_x1, target_x1)
    inter_y1 = torch.max(pred_y1, target_y1)
    inter_x2 = torch.min(pred_x2, target_x2)
    inter_y2 = torch.min(pred_y2, target_y2)

    inter_area = torch.clamp(inter_x2 - inter_x1, min=0) * torch.clamp(
        inter_y2 - inter_y1,
        min=0,
    )

    pred_area = (pred_x2 - pred_x1) * (pred_y2 - pred_y1)
    target_area = (target_x2 - target_x1) * (target_y2 - target_y1)

    union_area = pred_area + target_area - inter_area

    iou = inter_area / union_area
    return iou


def apply_nms(preds, iou_th=0.5):
    class_indices = preds[:, 0].int()
    class_probs = preds[:, -2]
    objectness = preds[:, -1]
    boxes = preds[:, 1:5].clone()

    # Convert box coordinates from (cx, cy, w, h) to (x1, y1, x2, y2)
    boxes[:, 0] -= boxes[:, 2] / 2
    boxes[:, 1] -= boxes[:, 3] / 2
    boxes[:, 2] += boxes[:, 0]
    boxes[:, 3] += boxes[:, 1]

    overall_scores = objectness * class_probs
    unique_classes = class_indices.unique()
    nms_indices_all = []

    for cls_idx in unique_classes:
        cls_mask = class_indices == cls_idx
        cls_boxes = boxes[cls_mask]
        cls_scores = overall_scores[cls_mask]

        nms_indices = ops.nms(cls_boxes, cls_scores, iou_th)
        nms_indices_all.append(cls_mask.nonzero()[nms_indices].flatten())

    nms_indices_all = torch.cat(nms_indices_all)

    return preds[nms_indices_all]


def compile_model(model: nn.Module, device, train_dl):
    model.to(torch.device(device))
    model = torch.compile(model)

    for images, _ in train_dl:
        model.train()
        _ = model(images.to(torch.device(device)))
        break
    return model


def ncxcywh2xyxy(tensor, width: int, height: int):
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


def encode_labels(labels: torch.Tensor, S, B, C) -> torch.Tensor:
    """SxS grid according to yolo algorithm
    labels: tensor of shape [N, 5]
    """
    encoded_labels = torch.zeros((S, S, B * 5 + C), dtype=torch.float32)
    if labels.numel() == 0:
        return encoded_labels

    # shapes: torch.Size([N]))
    class_labels = labels[:, 0].long()
    cx, cy = labels[:, 1], labels[:, 2]
    w, h = labels[:, 3], labels[:, 4]
    x, y = cx - w * 0.5, cy - h * 0.5

    # Calculate the cell indices and cell-rel coords
    i, j = (S * y).int(), (S * x).int()
    x_cell, y_cell = S * x - j, S * y - i

    # stacked shape: torch.Size([N, 5])
    boxes = torch.stack(
        (
            x_cell,
            y_cell,
            torch.sqrt(w),
            torch.sqrt(h),
            torch.ones_like(
                x_cell,
            ),
        ),
        dim=-1,
    )

    # Create a tensor to store the class probabilities (OHE)
    # scatter_ puts 1 on indices defined by class_labels.unsqueeze(1)
    # OHE shape: torch.Size([N, C])
    class_probs = torch.zeros((labels.shape[0], C), dtype=torch.float32)
    class_probs.scatter_(1, class_labels.unsqueeze(1), 1)

    encoded_labels[i, j, :-C] = boxes.float().repeat(1, B)
    encoded_labels[i, j, -C:] = class_probs.float()

    return encoded_labels


def decode_labels(
    encoded_labels,
    S=7,
    B=1,
    C=5,
    prob_th=0.1,
    nms=True,
    iou_th=0.5,
):
    """
    ip shape: [S, S, B*5 + C]
    op shape: [N, 7] if prob_th >= 0
    op shape: [S*S*B, 7] if prob_th == -1
    """
    if encoded_labels.numel() == 0:
        return torch.empty(size=(0, 7))
    # view shape: [S*S, B*5 + C]

    encoded_labels = encoded_labels.view(S * S, -1)
    # extracted cell boxes shape: [S*S*B, 5]
    boxes = encoded_labels[:, : B * 5].reshape(S * S * B, -1)
    # decoding OHE class probs
    class_probs, class_idx = torch.max(
        encoded_labels[:, -C:],
        dim=-1,
        keepdim=True,
    )
    class_probs = class_probs.repeat_interleave(B, dim=0)
    class_idx = class_idx.repeat_interleave(B, dim=0)
    # shape: [S*S*B, 7]
    boxes_and_probs = torch.cat([class_idx, boxes, class_probs], dim=-1)

    # decoding boxes to cxcywh format
    i = torch.arange(S, device=boxes.device).repeat_interleave(S * B)
    j = torch.arange(S, device=boxes.device).repeat(S).repeat_interleave(B)

    # w and h were square rooted during encoding.
    boxes_and_probs[:, 3:5].pow_(2)
    boxes_and_probs[:, 1] = ((boxes_and_probs[:, 1] + j) / S) + boxes_and_probs[
        :, 3
    ] * 0.5
    boxes_and_probs[:, 2] = ((boxes_and_probs[:, 2] + i) / S) + boxes_and_probs[
        :, 4
    ] * 0.5
    # p(class) = p(class | obj) * p(obj)
    prob_mask = (boxes_and_probs[:, -1] * boxes_and_probs[:, 5]) > prob_th
    valid_boxes_and_probs = boxes_and_probs[prob_mask]

    if valid_boxes_and_probs.numel() > 0 and nms:
        valid_boxes_and_probs = apply_nms(valid_boxes_and_probs, iou_th)

    return valid_boxes_and_probs


def count_parameters(model: torch.nn.Module):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
