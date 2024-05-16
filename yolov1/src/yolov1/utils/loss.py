import torch
import torch.nn as nn
from yolov1.config import YOLOConfig
from yolov1.utils.general import calc_iou, decode_labels


class YOLOLoss(nn.Module):
    def __init__(self, config: YOLOConfig):
        super().__init__()
        self.num_classes = config.model.nc
        self.lambda_coord = config.training.loss.l_coord
        self.lambda_obj = config.training.loss.l_obj
        self.lambda_noobj = config.training.loss.l_noobj
        self.lambda_class = config.training.loss.l_class
        self.config = config
        self.mse = nn.MSELoss(reduction="sum")

    def forward(self, preds, targets):
        """
        preds, targets: [N, S, S, B*5 + C]
        """
        S, B, C = (
            self.config.model.S,
            self.config.model.B,
            self.config.model.nc,
        )
        # shape: [N, S*S*B, 7] in (class_idx, cx, cy, w, h, conf, cond_cls_prob)
        N = len(targets)
        decoded_preds = torch.stack(
            [decode_labels(x, S, B, C, prob_th=-1, nms=False) for x in preds]
        )
        decoded_targets = torch.stack(
            [decode_labels(x, S, B, C, prob_th=-1, nms=False) for x in targets]
        )
        last_dim_sz = decoded_targets.shape[-1]

        # obj mask shape: [N, S*S*B, 7]
        mask_obj_exists = decoded_targets[..., 5] > 0
        mask_obj_exists = mask_obj_exists.unsqueeze(-1).expand_as(decoded_targets)

        # shape: [N, K, 4] K is a multiple of B since we had B times

        preds_filtered = decoded_preds[mask_obj_exists].view(-1, last_dim_sz)
        targets_filtered = decoded_targets[mask_obj_exists].view(-1, last_dim_sz)
        preds_unfiltered = decoded_preds[~mask_obj_exists].view(-1, last_dim_sz)
        targets_unfiltered = decoded_targets[~mask_obj_exists].view(-1, last_dim_sz)

        # shape: [N*K]
        ious = calc_iou(
            preds_filtered[..., 1:5].reshape(-1, 4),
            targets_filtered[..., 1:5].reshape(-1, 4),
        )

        # shape: [N, K/B, B]
        ious = ious.reshape(-1, B)

        # shape: [N, K, 4]
        _, best_iou_idx = torch.max(ious, dim=-1, keepdim=False)
        best_iou_mask = torch.zeros_like(ious)
        best_iou_mask = (
            best_iou_mask.scatter(
                -1,
                best_iou_idx.unsqueeze(-1),
                1,
            )
            .reshape(-1, 1)
            .bool()
        )

        # coord loss
        pred_boxes = preds_filtered[..., 1:5]
        target_boxes = targets_filtered[..., 1:5]
        resp_pred_boxes = pred_boxes[best_iou_mask.expand_as(pred_boxes)].view(-1, 4)
        resp_target_boxes = target_boxes[best_iou_mask.expand_as(target_boxes)].view(
            -1, 4
        )

        coord_loss = self.lambda_coord * self.mse(resp_pred_boxes, resp_target_boxes)

        # obj loss
        pred_confs = preds_filtered[..., 5:6]
        target_confs = targets_filtered[..., 5:6]
        resp_pred_confs = pred_confs[best_iou_mask.expand_as(pred_confs)].view(-1, 1)
        resp_target_confs = target_confs[best_iou_mask.expand_as(target_confs)].view(
            -1, 1
        )

        obj_loss = self.lambda_obj * self.mse(resp_pred_confs, resp_target_confs)

        # noobj loss
        noobj_pred_confs = preds_unfiltered[..., 5:6]
        noobj_target_confs = targets_unfiltered[..., 5:6]
        nonresp_pred_confs = pred_confs[~best_iou_mask.expand_as(pred_confs)].view(
            -1, 1
        )
        nonresp_target_confs = torch.zeros_like(nonresp_pred_confs)
        noobj_pred_confs = torch.cat([noobj_pred_confs, nonresp_pred_confs], dim=0)
        noobj_target_confs = torch.cat(
            [noobj_target_confs, nonresp_target_confs], dim=0
        )

        noobj_loss = self.lambda_noobj * self.mse(noobj_pred_confs, noobj_target_confs)

        # class loss
        mask_cls = mask_obj_exists.reshape(N, S, S, -1)[..., 0:1].expand_as(preds)
        pred_cls_probs = preds[mask_cls][..., -C:]
        target_cls_probs = targets[mask_cls][..., -C:]

        cls_loss = self.lambda_class * self.mse(pred_cls_probs, target_cls_probs)

        total_loss = coord_loss + obj_loss + noobj_loss + cls_loss

        return (
            total_loss / N,
            coord_loss,
            obj_loss,
            noobj_loss,
            cls_loss,
        )


class SimplifiedYOLOLoss(nn.Module):
    def __init__(self, config: YOLOConfig):
        super().__init__()
        self.num_classes = config.model.nc
        self.lambda_coord = config.training.loss.l_coord
        self.lambda_obj = config.training.loss.l_obj
        self.lambda_noobj = config.training.loss.l_noobj
        self.lambda_class = config.training.loss.l_class
        self.mse = nn.MSELoss(reduction="sum")

    def forward(self, predictions, targets):
        N = targets.size()[0]
        pred_boxes = predictions[..., :4]
        pred_conf = predictions[..., 4].unsqueeze(-1)
        pred_class = predictions[..., 5:]

        target_boxes = targets[..., :4]
        target_conf = targets[..., 4].unsqueeze(-1)
        target_class = targets[..., 5:]

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
            total_loss / N,
            coord_loss,
            obj_loss,
            noobj_loss,
            class_loss,
        )
