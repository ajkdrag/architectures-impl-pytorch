import structlog
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from yolov1.config import YOLOConfig
from yolov1.data.utils import get_dls_for_inference
from yolov1.models.model import YOLOv1LightningModel
from yolov1.utils.general import decode_labels
from yolov1.utils.vis import draw_boxes_tensor

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

    with torch.no_grad():
        for images in dataloader:
            images = images.to(device)
            encoded_outputs = model(images)
            decoded_outputs = [
                decode_labels(
                    o,
                    S,
                    B,
                    C,
                    prob_th=config.inference.prob_th,
                )
                .detach()
                .cpu()
                for o in encoded_outputs
            ]

            results = {
                "decoded_outputs": decoded_outputs,
                "drawn": [],
            }

            if draw:
                for image, output in zip(images, decoded_outputs):
                    results["drawn"].append(
                        draw_boxes_tensor(
                            image.detach().cpu(),
                            output,
                            display=False,
                        ),
                    )
            yield results


def main(config: YOLOConfig):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = YOLOv1LightningModel.load_from_checkpoint(
        config.inference.checkpoint, config=config
    )
    log.info("Model loaded successfully")

    inference_dl = get_dls_for_inference(config)
    for outputs in infer(model, inference_dl, config, device):
        yield outputs


if __name__ == "__main__":
    main()
