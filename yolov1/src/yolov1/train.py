import structlog
import torch
import torch.nn as nn
import torch.optim as optim
from pathlib import Path
from yolov1.config import YOLOConfig
from yolov1.data.utils import get_dls
from yolov1.models.arch import YOLOv1
from yolov1.utils.loss import SimplifiedYOLOLossV2
from yolov1.utils.general import count_parameters
from typing import Dict

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
    model = YOLOv1(config.model).to(device)
    num_params = count_parameters(model)
    log.info(f"Loaded model successfully with {num_params} trainable params")

    criterion = SimplifiedYOLOLossV2(config)
    optimizer = optim.Adam(
        model.parameters(),
        **config.training.optim_kwargs,
    )

    save_freq = cfg_train.save_freq
    last_epoch = cfg_train.epochs - 1

    for epoch in range(cfg_train.epochs):
        epoch_loss = train(model, train_dl, optimizer, criterion, device)
        log.info(f"Epoch [{epoch+1}/{cfg_train.epochs}], Train Loss: {epoch_loss}")

        checkpoint = {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
        }
        Path(cfg_train.checkpoints_dir).mkdir(parents=True, exist_ok=True)
        checkpoint_file = f"epoch_{epoch+1}.pt"
        if epoch == last_epoch:
            checkpoint_file = f"final_{checkpoint_file}"

        if (epoch % save_freq == 0) or epoch == last_epoch:
            torch.save(checkpoint, f"{cfg_train.checkpoints_dir}/{checkpoint_file}")
