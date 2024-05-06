import structlog
import torch
import torch.nn as nn
import torch.optim as optim
from pathlib import Path
from yolov1.config import YOLOConfig
from yolov1.data.utils import get_dls
from yolov1.models.arch import YOLOv1
from yolov1.utils.loss import SimplifiedYOLOLoss

log = structlog.get_logger()


def train(
    model: nn.Module,
    dataloader: torch.utils.data.DataLoader,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    device: str = "cpu",
) -> None:
    model.train()
    running_loss = 0.0

    for images, labels in dataloader:
        images = images.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

    return running_loss / len(dataloader)


def main(config: YOLOConfig):
    cfg_train = config.training
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_dl = get_dls(config, mode="train")
    model = YOLOv1(config.model).to(device)
    log.info("Loaded model successfully")

    criterion = SimplifiedYOLOLoss(config.model.nc)
    optimizer = optim.Adam(model.parameters(), lr=cfg_train.learning_rate)

    for epoch in range(cfg_train.epochs):
        train_loss = train(model, train_dl, optimizer, criterion, device)
        log.info(f"Epoch [{epoch+1}/{cfg_train.epochs}], Train Loss: {train_loss:.4f}")

        checkpoint = {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
        }
        Path(cfg_train.checkpoints_dir).mkdir(parents=True, exist_ok=True)
        checkpoint_file = f"epoch_{epoch+1}.pt"
        if epoch == cfg_train.epochs - 1:
            checkpoint_file = f"final_{checkpoint_file}"

        torch.save(checkpoint, f"{cfg_train.checkpoints_dir}/{checkpoint_file}")
