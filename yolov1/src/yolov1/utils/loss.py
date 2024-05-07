import torch
import torch.nn as nn


class SimplifiedYOLOLossV2(nn.Module):
    def __init__(
        self,
        num_classes,
        lambda_coord=5,
        lambda_noobj=0.5,
    ):
        super().__init__()
        self.num_classes = num_classes
        self.lambda_coord = lambda_coord
        self.lambda_noobj = lambda_noobj

    def forward(self, predictions, targets):
        pred_boxes = predictions[..., :4]
        pred_conf = predictions[..., 4].unsqueeze(-1)
        pred_class = predictions[..., 5:]

        target_boxes = targets[..., :4]
        target_conf = targets[..., 4].unsqueeze(-1)
        target_class = targets[..., 5:]

        obj_mask = target_conf.squeeze(-1) > 0

        coord_loss = self.lambda_coord * torch.mean(
            torch.square(
                pred_boxes[obj_mask] - target_boxes[obj_mask],
            ).sum(dim=-1),
        )

        obj_loss = torch.mean(
            torch.square(
                pred_conf[obj_mask] - target_conf[obj_mask],
            ),
        )

        class_loss = torch.mean(
            torch.square(
                pred_class[obj_mask] - target_class[obj_mask],
            ).sum(dim=-1),
        )

        noobj_loss = self.lambda_noobj * torch.mean(torch.square(pred_conf[~obj_mask]))

        total_loss = coord_loss + obj_loss + class_loss + noobj_loss

        return (
            total_loss,
            coord_loss,
            obj_loss,
            class_loss,
            noobj_loss,
        )


class SimplifiedYOLOLoss(nn.Module):
    def __init__(
        self,
        num_classes,
        lambda_coord=5,
        lambda_noobj=0.5,
    ):
        super().__init__()
        self.num_classes = num_classes
        self.lambda_coord = lambda_coord
        self.lambda_noobj = lambda_noobj

    def forward(self, predictions, targets):
        # Extract predicted bounding box coordinates and confidence
        pred_boxes = predictions[..., :4]
        pred_conf = predictions[..., 4].unsqueeze(-1)

        # Extract target bounding box coordinates and confidence
        target_boxes = targets[..., :4]
        target_conf = targets[..., 4].unsqueeze(-1)

        # Compute coordinate loss
        coord_loss = self.lambda_coord * torch.sum(
            torch.square(pred_boxes - target_boxes)
        )

        # Compute confidence loss
        obj_loss = torch.sum(torch.square(pred_conf - target_conf))
        noobj_loss = self.lambda_noobj * torch.sum(
            torch.square(pred_conf) * (1 - target_conf)
        )
        conf_loss = obj_loss + noobj_loss

        # Compute class probability loss
        pred_class = predictions[..., 5:]
        target_class = targets[..., 5:]
        class_loss = torch.sum(torch.square(pred_class - target_class))

        # Total loss
        total_loss = coord_loss + conf_loss + class_loss

        return total_loss
