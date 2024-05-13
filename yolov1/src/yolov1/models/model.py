import torch
import torch.nn as nn
import torch.optim as optim
from lightning import LightningModule
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torchmetrics.detection import MeanAveragePrecision
from yolov1.config import YOLOConfig
from yolov1.models.arch import YOLOv1
from yolov1.utils.general import decode_labels
from yolov1.utils.loss import SimplifiedYOLOLoss


class YOLOv1LightningModel(LightningModule):
    config: YOLOConfig
    model: nn.Module
    criterion: nn.Module

    def __init__(self, config):
        super().__init__()
        self.hparams.update(**vars(config))
        self.model = YOLOv1(config.model)
        self.criterion = SimplifiedYOLOLoss(config)
        self.map_metric = MeanAveragePrecision(
            box_format="cxcywh",
            iou_type="bbox",
            iou_thresholds=[0.5],
            extended_summary=False,
            average="macro",
            backend="faster_coco_eval",
        )

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        images, labels = batch
        outputs = self(images)
        losses = self.criterion(outputs, labels)
        self.log("train_loss", losses[0], prog_bar=True)
        return losses[0]

    def validation_step(self, batch, batch_idx):
        images, labels = batch
        outputs = self(images)
        loss, coord_loss, obj_loss, noobj_loss, class_loss = self.criterion(
            outputs, labels
        )

        # Decode the labels and predictions
        S, B, C = self.hparams.model.S, self.hparams.model.B, self.hparams.model.nc
        decoded_labels = torch.cat([decode_labels(label, S, B, C) for label in labels])
        decoded_preds = torch.cat(
            [decode_labels(o, S, B, C) for o in outputs.detach().cpu()]
        )
        print(decoded_preds.shape, decoded_labels.shape)
        preds = [
            dict(
                boxes=decoded_preds[..., 1:5],
                scores=decoded_preds[..., 5:6],
                labels=decoded_preds[..., 0:1],
            )
        ]
        targets = [
            dict(
                boxes=decoded_labels[..., 1:5],
                labels=decoded_preds[..., 0:1],
            )
        ]

        # Update the MeanAveragePrecision metric
        self.map_metric.update(preds, targets)
        self.log("mAP", self.map_metric, prog_bar=True)
        self.log("val_loss", loss, prog_bar=True)
        return loss

    def on_validation_epoch_end(self):
        lr = self.trainer.lr_scheduler_configs[0].scheduler.get_last_lr()[0]
        self.log("lr", lr, prog_bar=True)
        self.map_metric.reset()

    def configure_optimizers(self):
        optimizer = optim.Adam(
            self.parameters(),
            lr=self.hparams.lr,
            **self.hparams.training.optim_kwargs,
        )
        scheduler = {
            "scheduler": ReduceLROnPlateau(
                optimizer,
                factor=0.5,
                min_lr=0.000003,
            ),
            "name": "lr_scheduler",
            "monitor": "val_loss",
        }
        return [optimizer], [scheduler]
