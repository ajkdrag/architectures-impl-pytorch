import structlog
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from yolov1.config import YOLOConfig
from yolov1.data.dataset import YOLODataset
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


def _get_dls(config: YOLOConfig, mode="train"):
    dataset = YOLODataset(config, mode=mode)
    dataloader = DataLoader(
        dataset,
        batch_size=config.training.batch_size,
        num_workers=config.training.num_workers,
        shuffle=True if mode == "train" else False,
    )
    return dataloader


def main(config: YOLOConfig):
    cfg_train = config.training
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_dl = _get_dls(config)
    model = YOLOv1(config.model).to(device)

    criterion = SimplifiedYOLOLoss(config.model.nc)
    optimizer = optim.Adam(model.parameters(), lr=cfg_train.learning_rate)

    for epoch in range(cfg_train.epochs):
        train_loss = train(model, train_dl, optimizer, criterion, device)
        log.info(
            f"Epoch [{epoch+1}/{cfg_train.epochs}], Train Loss: {train_loss:.4f}")

        checkpoint = {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
        }
        torch.save(checkpoint, f"checkpoints/epoch_{epoch+1}.pt")

    torch.save(model.state_dict(), "models/yolov1_final.pt")
