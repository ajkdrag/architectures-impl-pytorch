import lightning as pl
import structlog
import torch
from lightning.pytorch.callbacks import ModelCheckpoint
from lightning.pytorch.utilities.model_summary import ModelSummary

from yolov1.config import YOLOConfig
from yolov1.data.utils import get_dls
from yolov1.models.model import YOLOv1LightningModel

log = structlog.get_logger()
torch.backends.cudnn.benchmark = True
torch.set_float32_matmul_precision("medium")


def main(config: YOLOConfig):
    train_dl = get_dls(config, mode="train")
    val_dl = get_dls(config, mode="valid")

    pl_model = YOLOv1LightningModel(config)
    ModelSummary(pl_model)

    checkpoint_callback = ModelCheckpoint(
        dirpath=config.training.checkpoints_dir,
        filename="epoch_{epoch:02d}_{val_mAP:.2f}",
        save_top_k=1,
        save_last=True,
        monitor="val_mAP",
        mode="max",
    )

    trainer = pl.Trainer(
        max_epochs=config.training.epochs,
        callbacks=[checkpoint_callback],
        default_root_dir=config.training.checkpoints_dir,
        accelerator="gpu",
        precision="16",
        check_val_every_n_epoch=config.training.val_freq,
        log_every_n_steps=1,
    )

    trainer.fit(pl_model, train_dl, val_dl)
