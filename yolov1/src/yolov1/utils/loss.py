import torch.nn as nn
from yolov1.config import YOLOConfig


class SimplifiedYOLOLossV2(nn.Module):
    def __init__(self, config: YOLOConfig):
        super().__init__()
        self.num_classes = config.model.nc
        self.lambda_coord = config.training.loss.l_coord
        self.lambda_obj = config.training.loss.l_obj
        self.lambda_noobj = config.training.loss.l_noobj
        self.lambda_class = config.training.loss.l_class
        self.mse = nn.MSELoss(reduction="sum")

    def forward(self, predictions, targets):
        pred_boxes = predictions[..., :4]
        pred_conf = predictions[..., 4].unsqueeze(-1)
        pred_class = predictions[..., 5:]

        target_boxes = targets[..., :4]
        target_conf = targets[..., 4].unsqueeze(-1)
        target_class = targets[..., 5:]

        # pred_boxes_wh = pred_boxes[..., 2:].clone()
        # pred_boxes_wh = torch.sqrt(
        #     torch.abs(pred_boxes_wh)) * torch.sign(pred_boxes_wh)
        # pred_boxes = torch.cat(
        #     (pred_boxes[..., :2], pred_boxes_wh),
        #     dim=-1,
        # )
        # target_boxes[..., 2:] = torch.sqrt(target_boxes[..., 2:])

        obj_mask = target_conf.squeeze(-1) > 0

        coord_loss = self.lambda_coord * self.mse(
            pred_boxes[obj_mask],
            target_boxes[obj_mask],
        )

        obj_loss = self.lambda_obj * self.mse(
            pred_conf[obj_mask],
            target_conf[obj_mask],
        )

        noobj_loss = self.lambda_noobj * self.mse(
            pred_conf[~obj_mask],
            target_conf[~obj_mask],
        )

        class_loss = self.lambda_class * self.mse(
            pred_class[obj_mask],
            target_class[obj_mask],
        )

        total_loss = coord_loss + obj_loss + noobj_loss + class_loss

        return (
            total_loss,
            coord_loss,
            obj_loss,
            noobj_loss,
            class_loss,
        )
