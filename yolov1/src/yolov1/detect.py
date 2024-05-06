import structlog
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from yolov1.config import YOLOConfig
from yolov1.data.utils import get_dls_for_inference
from yolov1.models.arch import YOLOv1
from yolov1.utils.general import decode_labels
from yolov1.utils.vis import draw_boxes

log = structlog.get_logger()


def infer(
    model: nn.Module,
    dataloader: DataLoader,
    config: YOLOConfig,
    device: str = "cpu",
    draw: bool = True,
):
    model.to(device)
    model.eval()

    S, B, C = config.model.S, config.model.B, config.model.nc

    mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)

    with torch.no_grad():
        for images in dataloader:
            images = images.to(device)
            encoded_outputs = model(images)
            decoded_outputs = [decode_labels(o, S, B, C)
                               for o in encoded_outputs]
            decoded_images = images * std + mean

            results = {
                "decoded_outputs": decoded_outputs,
                "drawn": [],
            }

            if draw:
                for image in decoded_images:
                    results["drawn"].append(
                        draw_boxes(image, decoded_outputs),
                    )
            yield results


def main(config: YOLOConfig):
    model = YOLOv1(config.model)
    model.load_state_dict(
        torch.load(config.inference.checkpoint)["model_state_dict"],
    )
    log.info("Model loaded successfully")

    inference_dl = get_dls_for_inference(config)
    for outputs in infer(model, inference_dl, config):
        yield outputs


if __name__ == "__main__":
    main()
