from pathlib import Path
from typing import Dict

import structlog
import torch
import torch.nn as nn
import torch.optim as optim

from yolov1.config import YOLOConfig
from yolov1.data.utils import get_dls
from yolov1.eval import validate
from yolov1.models.arch import YOLOv2 as YOLOv1
from yolov1.utils.general import count_parameters
from yolov1.utils.loss import SimplifiedYOLOLossV2

log = structlog.get_logger()


def train(
    model: nn.Module,
    dataloader: torch.utils.data.DataLoader,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    device: str = "cpu",
) -> Dict[str, float]:
    model.train()
    running_losses = {
        "total": 0.0,
        "coord": 0.0,
        "obj": 0.0,
        "noobj": 0.0,
        "class": 0.0,
    }

    for images, labels in dataloader:
        images = images.to(device)
        labels = labels.to(device)

        outputs = model(images)

        loss, coord_loss, obj_loss, noobj_loss, class_loss = criterion(outputs, labels)

        # clear-fill-use
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

        running_losses["total"] += loss.item()
        running_losses["coord"] += coord_loss.item()
        running_losses["obj"] += obj_loss.item()
        running_losses["noobj"] += noobj_loss.item()
        running_losses["class"] += class_loss.item()

    epoch_losses = {k: v / len(dataloader) for k, v in running_losses.items()}

    return epoch_losses


def main(config: YOLOConfig):
    cfg_train = config.training
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_dl = get_dls(config, mode="train")
    val_dl = get_dls(config, mode="valid")

    model = YOLOv1(config.model).to(device)
    num_params = count_parameters(model)
    log.info(f"Loaded model successfully with {num_params} trainable params")

    criterion = SimplifiedYOLOLossV2(config)
    optimizer = optim.SGD(
        model.parameters(),
        **config.training.optim_kwargs,
    )

    save_freq = cfg_train.save_freq
    val_freq = cfg_train.val_freq
    epochs = cfg_train.epochs
    learning_rate = cfg_train.optim_kwargs.get("lr")

    for epoch in range(1, epochs + 1):
        if epoch == 30:
            learning_rate = 0.0001
        if epoch == 40:
            learning_rate = 0.00001
        for param_group in optimizer.param_groups:
            param_group["lr"] = learning_rate

        # train
        epoch_losses = train(model, train_dl, optimizer, criterion, device)
        log.info(f"[{epoch}/{epochs}] Train Loss: {epoch_losses}")

        # eval
        if epoch % val_freq == 0:
            val_losses, val_metrics = validate(
                model, val_dl, criterion, device, metrics=None
            )
            log.info(
                f"[{epoch}/{epochs}] Val Loss: {val_losses}, Val Metrics: {val_metrics}"
            )

        checkpoint = {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
        }

        Path(cfg_train.checkpoints_dir).mkdir(parents=True, exist_ok=True)
        checkpoint_file = f"epoch_{epoch}.pt"
        if epoch == epochs:
            checkpoint_file = f"final_{checkpoint_file}"

        if (epoch % save_freq == 0) or epoch == epochs:
            torch.save(checkpoint, f"{cfg_train.checkpoints_dir}/{checkpoint_file}")
