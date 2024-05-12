from typing import Dict

import torch
import torch.nn as nn


def validate(
    model: nn.Module,
    dataloader: torch.utils.data.DataLoader,
    criterion: nn.Module,
    device: str = "cpu",
    metrics: Dict[str, callable] = None,
) -> Dict[str, float]:
    model.eval()
    running_losses = {
        "total": 0.0,
        "coord": 0.0,
        "obj": 0.0,
        "noobj": 0.0,
        "class": 0.0,
    }
    running_metrics = {
        metric_name: 0.0 for metric_name in metrics} if metrics else {}

    with torch.no_grad():
        for images, labels in dataloader:
            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)

            loss, coord_loss, obj_loss, noobj_loss, class_loss = criterion(
                outputs, labels
            )

            running_losses["total"] += loss.item()
            running_losses["coord"] += coord_loss.item()
            running_losses["obj"] += obj_loss.item()
            running_losses["noobj"] += noobj_loss.item()
            running_losses["class"] += class_loss.item()

            if metrics:
                for metric_name, metric_fn in metrics.items():
                    running_metrics[metric_name] += metric_fn(outputs, labels)

    epoch_losses = {k: v / len(dataloader) for k, v in running_losses.items()}
    epoch_metrics = {k: v / len(dataloader)
                     for k, v in running_metrics.items()}

    return epoch_losses, epoch_metrics
