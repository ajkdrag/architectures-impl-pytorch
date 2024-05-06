import torch
import torch.nn as nn
from torchvision import transforms
from torch.utils.data import DataLoader
from PIL import ImageDraw
import structlog
from yolov1.config import YOLOConfig
from yolov1.models.arch import YOLOv1
from yolov1.utils.general import decode_labels
from yolov1.data.utils import get_dls_for_inference


log = structlog.get_logger()


def decode_output(output, config: YOLOConfig):
    S = config.model.S
    B = 1
    C = config.model.nc

    output = output.reshape(S, S, B * 5 + C)
    boxes = []

    for i in range(S):
        for j in range(S):
            cell_output = output[i, j]
            bbox = cell_output[:4]
            conf = cell_output[4]
            class_probs = cell_output[5:]

            if conf > config.model.conf_th:
                x_cell, y_cell, w_cell, h_cell = bbox
                x = (j + x_cell) / S
                y = (i + y_cell) / S
                w = w_cell
                h = h_cell

                class_idx = torch.argmax(class_probs)
                class_prob = class_probs[class_idx]

                box = [x, y, w, h, conf, class_idx.item(), class_prob.item()]
                boxes.append(box)

    return boxes


def visualize_outputs(l_images, l_boxes, config):
    l_images = l_images.detach().cpu()
    visualized_images = []
    mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)

    for image, boxes in zip(l_images, l_boxes):
        image_tensor = image * std + mean
        image = transforms.ToPILImage()(image_tensor)

        draw = ImageDraw.Draw(image)
        for box in boxes:
            class_idx, x, y, w, h, conf, class_prob = box
            x1 = int((x - w / 2) * image.width)
            y1 = int((y - h / 2) * image.height)
            x2 = int((x + w / 2) * image.width)
            y2 = int((y + h / 2) * image.height)
            if x2 <= x1 or y2 <= y1:
                continue

            draw.rectangle((x1, y1, x2, y2), outline="red", width=2)
            draw.text(
                (x1, y1),
                f"{int(class_idx)}: {class_prob:.2f}",
                fill="red",
            )

        visualized_images.append(image)
    return visualized_images


def infer(
    model: nn.Module,
    dataloader: DataLoader,
    config: YOLOConfig,
    device: str = "cpu",
):
    model.to(device)
    model.eval()

    S, B, C = config.model.S, config.model.B, config.model.nc

    with torch.no_grad():
        for images in dataloader:
            images = images.to(device)
            encoded_outputs = model(images)
            decoded_outputs = [decode_labels(o, S, B, C) for o in encoded_outputs]
            visualized_images = visualize_outputs(images, decoded_outputs, config)

            yield visualized_images


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
